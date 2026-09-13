"""Array-backend independent horizontal speed and takeover command definitions."""


def heading_displacement(xp, dx, dy, previous_heading, heading):
    delta = xp.arctan2(xp.sin(heading - previous_heading), xp.cos(heading - previous_heading))
    middle = previous_heading + 0.5 * delta
    return dx * xp.cos(middle) + dy * xp.sin(middle)


def takeover_commands(xp, initial_speed, target_speed, target_yaw, elapsed_s,
                      speed_slew, yaw_slew):
    elapsed_s = xp.maximum(elapsed_s, 0.0)
    speed = initial_speed + xp.clip(target_speed - initial_speed,
                                   -speed_slew * elapsed_s, speed_slew * elapsed_s)
    yaw = xp.clip(target_yaw, -yaw_slew * elapsed_s, yaw_slew * elapsed_s)
    transitioning = (xp.abs(speed - target_speed) > 1e-6) | (xp.abs(yaw - target_yaw) > 1e-6)
    return speed, yaw, transitioning
