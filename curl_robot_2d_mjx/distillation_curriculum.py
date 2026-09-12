"""Host-side selection of complete rolling snapshots and continuation candidates."""

import numpy as np


def low_speed_snapshot_indices(forward, yaw, *, speed_min, speed_max,
                               straight_fraction, seed):
    """Sample 60/20/20 speed mass, preserving the requested turn distribution."""
    forward, yaw = np.asarray(forward), np.asarray(yaw)
    if forward.ndim != 1 or forward.shape != yaw.shape or not len(forward):
        raise ValueError("Snapshot commands must be nonempty matching vectors")
    if not np.isfinite([speed_min, speed_max, straight_fraction]).all():
        raise ValueError("Sampling settings must be finite")
    if not speed_min < speed_max or not 0 <= straight_fraction <= 1:
        raise ValueError("Expected a nonzero speed range and straight fraction in [0, 1]")
    if (not np.isfinite(forward).all() or not np.isfinite(yaw).all()
            or np.any(forward < speed_min - 1e-6)
            or np.any(forward > speed_max + 1e-6)):
        raise ValueError("Snapshot commands are nonfinite or outside the configured range")
    edges = np.linspace(speed_min, speed_max, 4)
    bins = np.searchsorted(edges[1:-1], forward, side="right")
    directions = (np.abs(yaw) <= 1e-3, yaw > 1e-3, yaw < -1e-3)
    turn_mass = (straight_fraction, (1 - straight_fraction) / 2,
                 (1 - straight_fraction) / 2)
    probabilities = np.zeros(len(forward), dtype=np.float64)
    groups = {}
    for b, (speed, mass) in enumerate(zip(("low", "medium", "high"), (0.6, 0.2, 0.2))):
        for direction, mask, fraction in zip(("straight", "left", "right"), directions, turn_mass):
            selected = (bins == b) & mask
            count = int(selected.sum())
            target = mass * fraction
            if target and not count:
                raise ValueError(f"No snapshots for {speed}/{direction}; increase --envs")
            if count:
                probabilities[selected] = target / count
            groups[f"{speed}/{direction}"] = {"available": count, "probability": target}
    probabilities /= probabilities.sum()
    indices = np.random.default_rng(seed).choice(len(forward), len(forward), p=probabilities)
    for b, speed in enumerate(("low", "medium", "high")):
        for direction, mask in zip(("straight", "left", "right"), directions):
            groups[f"{speed}/{direction}"]["sampled"] = int(((bins == b) & mask)[indices].sum())
    return indices.astype(np.int32), {
        "mode": "low_speed_focus", "speed_bin_edges_m_s": edges.tolist(),
        "groups": groups, "unique_snapshots": int(len(np.unique(indices))),
        "note": "Training reset sampling with replacement; evaluation remains uniform. "
                "State, history and previous action use identical indices. "
                "Step-count timeout restarts at takeover; physics and phase are preserved.",
    }


def continuation_decision(baseline, best, candidate):
    """Operational guardrails, not a statistical significance test."""
    reasons = []
    for report in (baseline, best, candidate):
        for group in (report['overall'], *report['by_speed'].values()):
            for key in ('success_rate', 'full_horizon_rate', 'forward_mae_m_s', 'yaw_mae_rad_s'):
                if group.get(key) is None or not np.isfinite(group[key]):
                    return {'safe': False, 'select': False,
                            'reasons': ['Empty/nonfinite evaluation group'], 'low_speed_improved': False}
    for reference in (baseline, best):
        if (candidate['speed_bin_edges_m_s'] != reference['speed_bin_edges_m_s']
                or candidate['criteria'] != reference['criteria']
                or candidate['overall']['episodes'] != reference['overall']['episodes']):
            return {'safe': False, 'select': False,
                    'reasons': ['Evaluation criteria/population changed'], 'low_speed_improved': False}
        # Seeds alone are not sufficient to claim equal physical states, but
        # changed commands would make even this weaker comparison invalid.
        keys = ('forward_command_m_s', 'yaw_command_rad_s', 'teacher_warmup_steps')
        if (len(candidate['per_episode']) != len(reference['per_episode'])
                or any(a[k] != b[k] for a, b in zip(candidate['per_episode'], reference['per_episode'])
                       for k in keys)):
            return {'safe': False, 'select': False,
                    'reasons': ['Evaluation commands or warmup changed'], 'low_speed_improved': False}
    # Compare to both the original policy and the best accepted candidate so
    # repeated small regressions cannot silently accumulate.
    for label, reference in (("baseline", baseline), ("best", best)):
        for group, allowed in (("overall", .02), ("medium", .03), ("high", .03)):
            old = reference["overall"] if group == "overall" else reference["by_speed"][group]
            new = candidate["overall"] if group == "overall" else candidate["by_speed"][group]
            if new["success_rate"] < old["success_rate"] - allowed:
                reasons.append(f"{group} success dropped > {allowed:.0%} against {label}")
        for metric, allowed in (("full_horizon_rate", .02),):
            if candidate["overall"][metric] < reference["overall"][metric] - allowed:
                reasons.append(f"{metric} dropped > {allowed:.0%} against {label}")
        for metric, allowed in (("forward_mae_m_s", .005), ("yaw_mae_rad_s", .003)):
            if candidate["overall"][metric] > reference["overall"][metric] + allowed:
                reasons.append(f"{metric} increased > {allowed} against {label}")
    low, old_low = candidate["by_speed"]["low"], best["by_speed"]["low"]
    improved = (low["success_rate"] > old_low["success_rate"]
                and low["forward_mae_m_s"] < old_low["forward_mae_m_s"])
    return {"safe": not reasons, "select": not reasons and improved,
            "reasons": reasons, "low_speed_improved": improved}
