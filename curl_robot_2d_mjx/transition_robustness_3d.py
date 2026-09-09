"""Bounded force pulses, shared by NumPy tests and MJX stepping."""


def horizontal_push_force(xp, samples, step, dt, mass, config):
    """Four U[0,1) samples: enable, onset, direction, magnitude.

    Force is world-frame, at torso COM. No velocity injection or vertical force.
    Timing is discretized once to control ticks; pulse force is cleared afterward.
    """
    low, high = config.push_start_range_s
    start = xp.ceil((low + (high - low) * samples[1]) / dt)
    ticks = xp.ceil(config.push_duration_s / dt)
    active = ((samples[0] < config.push_probability)
              & (step >= start) & (step < start + ticks))
    angle = samples[2] * (2.0 * 3.141592653589793)
    magnitude = mass * config.push_acceleration_m_s2 * samples[3] * active
    return magnitude * xp.stack((xp.cos(angle), xp.sin(angle), xp.asarray(0.0)))
