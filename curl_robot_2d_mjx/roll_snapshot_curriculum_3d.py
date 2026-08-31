"""CPU-only selection of real ROLL cycles; never modify simulator states."""

import numpy as np


ROLL_PROGRESS_SOURCE = "signed_body_y_integral_from_episode_reset"
ROLL_PROGRESS_FIELDS = ("roll_phase_rad", "roll_origin_phase_rad")


def validate_roll_progress(arrays):
    count = len(arrays["qpos"])
    if str(arrays.get("roll_progress_source", "")) != ROLL_PROGRESS_SOURCE:
        raise ValueError("cycle curriculum needs signed ROLL progress tracked from reset; "
                         "recollect v2 snapshots, do not infer turns from sparse poses/time")
    for name in ROLL_PROGRESS_FIELDS:
        if name not in arrays or np.shape(arrays[name]) != (count,):
            raise ValueError(f"ROLL cycle snapshots require {name} shape ({count},)")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"nonfinite {name}")
    for episode in np.unique(arrays["episode_id"]):
        selected = arrays["episode_id"] == episode
        origins = arrays["roll_origin_phase_rad"][selected]
        if not np.allclose(origins, origins[0], rtol=0, atol=1e-8):
            raise ValueError("ROLL progress origin must be fixed at each episode reset")
        if np.any(np.diff(arrays["time_s"][selected]) <= 0):
            raise ValueError("snapshot time must strictly increase within each episode; "
                             "do not collect through an auto-reset wrapper")


def _range(values):
    return {"min": float(np.min(values)), "median": float(np.median(values)),
            "max": float(np.max(values))}


def select_roll_cycle_snapshots(arrays, config, *, require_coverage=True):
    """Use signed net turns, not absolute angular travel (rocking cancels).

    Bounded courses require every requested complete cycle/phase cell. The
    full-bank course retains partial final cycles and reports their coverage;
    it does not certify unobserved higher speeds or contact safety.
    """
    validate_roll_progress(arrays)
    turns = config.snapshot_roll_direction * (
        arrays["roll_phase_rad"] - arrays["roll_origin_phase_rad"]) / (2 * np.pi)
    keep = turns >= config.snapshot_min_turns
    if config.snapshot_max_turns is not None:
        keep &= turns < config.snapshot_max_turns
    after_cycle_filter = int(keep.sum())
    if config.snapshot_tail_fraction < 1:
        for episode in np.unique(arrays["episode_id"]):
            indices = np.flatnonzero(arrays["episode_id"] == episode)
            times = arrays["time_s"][indices]
            threshold = times.max() - config.snapshot_tail_fraction * (times.max() - times.min())
            keep[indices] &= times >= threshold
    if not keep.any():
        raise ValueError(f"no ROLL snapshots in turns [{config.snapshot_min_turns}, "
                         f"{config.snapshot_max_turns}); available directed net turns={_range(turns)}. "
                         "Collect the missing cycles/check direction or tail filter; never scale qvel.")

    selected_turns = turns[keep]
    cycle = np.floor(selected_turns).astype(np.int64)
    bins = config.snapshot_phase_bins
    phase_bin = np.minimum(((selected_turns - cycle) * bins).astype(int), bins - 1)
    phase_counts = np.bincount(phase_bin, minlength=bins)
    cycles = (range(int(config.snapshot_min_turns), int(config.snapshot_max_turns))
              if config.snapshot_max_turns is not None else np.unique(cycle))
    cells = {str(int(c)): np.bincount(phase_bin[cycle == c], minlength=bins).tolist()
             for c in cycles}
    incomplete = [c for c, counts in cells.items() if min(counts) == 0]
    coverage_complete = bool(np.all(phase_counts > 0) and
                             (config.snapshot_max_turns is None or not incomplete))
    if require_coverage and not coverage_complete:
        raise ValueError(f"incomplete ROLL cycle/phase coverage: {cells}; collect longer/more densely "
                         "(sample_every=1, warmup_steps=0). Do not replace missing cycles with startup.")

    # Equal probability per occupied (cycle, phase) cell, then uniform within
    # that cell. Slow portions must not dominate just because they last longer.
    _, inverse, counts = np.unique(cycle * bins + phase_bin, return_inverse=True,
                                  return_counts=True)
    probabilities = 1.0 / (len(counts) * counts[inverse])
    cdf = np.cumsum(probabilities).astype(np.float32)
    cdf[-1] = 1.0
    keys = ("qpos", "qvel", "ctrl", "time_s", "episode_id", *ROLL_PROGRESS_FIELDS)
    bank = {key: arrays[key][keep].copy() for key in keys}
    bank["sampling_cdf"] = cdf
    linear = np.linalg.norm(bank["qvel"][:, :3], axis=1)
    angular = np.linalg.norm(bank["qvel"][:, 3:6], axis=1)
    report = {
        "selection": "real_roll_cycles_v1", "stage": config.curriculum_stage,
        "source_policy": str(arrays["source_policy"].item()),
        "roll_progress_source": ROLL_PROGRESS_SOURCE,
        "requested_turns": [config.snapshot_min_turns, config.snapshot_max_turns],
        "direction": config.snapshot_roll_direction,
        "total_samples": len(turns), "samples_after_cycle_filter": after_cycle_filter,
        "selected_samples": len(selected_turns), "episodes": int(len(np.unique(bank["episode_id"]))),
        "available_directed_turns": _range(turns), "selected_directed_turns": _range(selected_turns),
        "selected_time_s": _range(bank["time_s"]),
        "linear_speed_m_s": _range(linear), "angular_speed_rad_s": _range(angular),
        "phase_bin_counts": phase_counts.tolist(), "cycle_phase_counts": cells,
        "coverage_complete": coverage_complete, "incomplete_cycles": incomplete,
        "sampling": "uniform_occupied_cycle_phase_then_uniform_sample",
        "velocities_modified": False,
        "per_cycle": {str(int(c)): {
            "samples": int(np.sum(cycle == c)),
            "linear_speed_m_s": _range(linear[cycle == c]),
            "angular_speed_rad_s": _range(angular[cycle == c]),
        } for c in np.unique(cycle)},
    }
    return bank, report
