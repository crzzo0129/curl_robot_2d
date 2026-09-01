"""Mutually exclusive terminal-failure diagnostics for Transition."""


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
