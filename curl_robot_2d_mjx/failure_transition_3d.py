"""Dependency-light terminal and reset-source diagnostics for Transition."""


TRANSITION_FAILURE_MODE_NAMES_3D = ("brake", "deploy", "stabilize")
TRANSITION_SOURCE_OUTCOME_NAMES_3D = (
    "episodes", "success", "failed", "timeout", "root_height_low",
)


TRANSITION_FAILURE_CAUSE_NAMES_3D = (
    "action_nonfinite",
    "physics_nonfinite",
    "root_height_low",
    "root_height_high",
    "brake_timeout",
    "deploy_timeout",
    "stabilize_guard",
    "other",
)


def transition_failure_causes_3d(xp, *, failed, action_finite, physics_finite,
                                 root_height_low, root_height_high,
                                 brake_timeout, deploy_timeout, stabilize_guard):
    """Assign each failed episode to exactly one deterministic cause.

    Priority handles simultaneous predicates. Invalid numerics precede physical
    bounds; bounds precede mode timeouts; STABILIZE guard is last because a
    fallen stance can trigger it together with root-height loss.
    """
    predicates = tuple(xp.asarray(value, dtype=bool) for value in (
        xp.logical_not(action_finite),
        xp.logical_not(physics_finite),
        root_height_low,
        root_height_high,
        brake_timeout,
        deploy_timeout,
        stabilize_guard,
    ))
    remaining = xp.asarray(failed, dtype=bool)
    causes = {}
    for name, predicate in zip(TRANSITION_FAILURE_CAUSE_NAMES_3D[:-1], predicates):
        causes[name] = remaining & predicate
        remaining = remaining & (~predicate)
    causes["other"] = remaining
    return causes


def transition_failure_breakdown_3d(metrics, prefix="eval/episode_failure_"):
    """Build a human-readable final summary from Brax evaluation metrics."""
    rates = {name: float(metrics.get(prefix + name, 0.0))
             for name in TRANSITION_FAILURE_CAUSE_NAMES_3D}
    total = float(metrics.get("eval/episode_failed", 0.0))
    summed = sum(rates.values())
    dominant = max(rates, key=rates.get) if summed > 0 else None
    return {"rates": rates, "sum": summed, "total_failed": total,
            "consistency_error": summed - total, "dominant_cause": dominant,
            "mutually_exclusive": True}


def transition_failure_mode_metrics_3d(xp, causes, mode):
    """Cross exclusive failure causes with the mode whose action was applied."""
    result = {}
    for mode_index, mode_name in enumerate(TRANSITION_FAILURE_MODE_NAMES_3D):
        in_mode = xp.asarray(mode == mode_index, dtype=bool)
        for cause_name in TRANSITION_FAILURE_CAUSE_NAMES_3D:
            result[f"failure_{cause_name}_mode_{mode_name}"] = (
                xp.asarray(causes[cause_name], dtype=bool) & in_mode
            )
    return result


def transition_failure_mode_breakdown_3d(metrics, prefix="eval/episode_"):
    """Summarize the mutually exclusive ``cause x active mode`` matrix."""
    by_mode = {}
    cells = {}
    for mode_name in TRANSITION_FAILURE_MODE_NAMES_3D:
        rates = {}
        for cause_name in TRANSITION_FAILURE_CAUSE_NAMES_3D:
            value = float(metrics.get(
                f"{prefix}failure_{cause_name}_mode_{mode_name}", 0.0
            ))
            rates[cause_name] = value
            cells[f"{cause_name}@{mode_name}"] = value
        by_mode[mode_name] = {"rates": rates, "total": sum(rates.values())}
    summed = sum(cells.values())
    total = float(metrics.get("eval/episode_failed", 0.0))
    dominant = max(cells, key=cells.get) if summed > 0 else None
    return {
        "by_mode": by_mode,
        "sum": summed,
        "total_failed": total,
        "consistency_error": summed - total,
        "dominant_cell": dominant,
        "mutually_exclusive": True,
        "mode_definition": "mode whose policy action produced the terminal physics step",
    }


def transition_source_metrics_3d(
    xp,
    *,
    done,
    success,
    failed,
    timeout,
    root_height_low,
    source_phase_bin,
    source_cycle,
    phase_bins,
    cycles,
):
    """Create terminal pulses grouped by the immutable reset snapshot labels."""
    result = {}
    outcomes = {
        "episodes": done,
        "success": success,
        "failed": failed,
        "timeout": timeout,
        "root_height_low": root_height_low,
    }
    for group_name, source, labels in (
        ("phase_bin", source_phase_bin, range(phase_bins)),
        ("cycle", source_cycle, cycles),
    ):
        for label in labels:
            selected = xp.asarray(source == label, dtype=bool)
            for outcome_name, outcome in outcomes.items():
                result[f"source_{group_name}_{label}_{outcome_name}"] = (
                    selected & xp.asarray(outcome, dtype=bool)
                )
    return result


def _conditional_source_row(metrics, stem, prefix, episode_count):
    values = {
        name: float(metrics.get(f"{prefix}{stem}_{name}", 0.0))
        for name in TRANSITION_SOURCE_OUTCOME_NAMES_3D
    }
    exposure = values["episodes"]
    conditional = {
        name + "_rate": (values[name] / exposure if exposure > 0 else None)
        for name in TRANSITION_SOURCE_OUTCOME_NAMES_3D[1:]
    }
    conditional["outcome_consistency_error"] = (
        (values["success"] + values["failed"] + values["timeout"] - exposure)
        / exposure if exposure > 0 else None
    )
    return {
        "evaluation_fraction": exposure,
        "episodes": (
            int(round(exposure * episode_count))
            if episode_count is not None else None
        ),
        **conditional,
    }


def transition_source_breakdown_3d(
    metrics, *, phase_bins, cycles, episode_count=None,
    prefix="eval/episode_"
):
    """Return conditional outcome rates for every sampled phase and cycle."""
    by_phase = {
        str(index): _conditional_source_row(
            metrics, f"source_phase_bin_{index}", prefix, episode_count
        )
        for index in range(phase_bins)
    }
    by_cycle = {
        str(cycle): _conditional_source_row(
            metrics, f"source_cycle_{cycle}", prefix, episode_count
        )
        for cycle in cycles
    }
    observed = [
        (f"phase_bin_{label}", row["failed_rate"])
        for label, row in by_phase.items()
        if row["failed_rate"] is not None
    ]
    worst = max(observed, key=lambda item: item[1])[0] if observed else None
    phase_fraction_sum = sum(row["evaluation_fraction"] for row in by_phase.values())
    cycle_fraction_sum = sum(row["evaluation_fraction"] for row in by_cycle.values())
    return {
        "by_phase_bin": by_phase,
        "by_cycle": by_cycle,
        "worst_phase_bin_by_failure_rate": worst,
        "phase_evaluation_fraction_sum": phase_fraction_sum,
        "cycle_evaluation_fraction_sum": cycle_fraction_sum,
        "phase_coverage_consistency_error": (
            phase_fraction_sum - 1.0 if phase_bins else None
        ),
        "cycle_coverage_consistency_error": (
            cycle_fraction_sum - 1.0 if cycles else None
        ),
        "rates_are_conditional_on_group": True,
    }
