"""Pure helpers for absolute transition control and its reward reference."""


def interpolated_stand_target(xp, handoff, stand, elapsed, duration):
    alpha = xp.clip(elapsed / duration, 0.0, 1.0)
    blend = alpha * alpha * (3.0 - 2.0 * alpha)
    return handoff + blend * (stand - handoff)


def limit_transition_target(xp, requested, previous, rates, dt, low, high):
    """Once per policy tick; previous starts at the last ROLL motor target."""
    requested = xp.clip(requested, low, high)
    delta = xp.asarray(rates) * dt
    return previous + xp.clip(requested - previous, -delta, delta)
