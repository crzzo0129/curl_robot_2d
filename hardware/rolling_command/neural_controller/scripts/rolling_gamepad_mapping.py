"""Command mapping for the command-conditioned rolling policy only."""
import math


def rolling_command(axes, forward_axis=1, yaw_axis=3, deadzone=0.10):
    if not 0 <= deadzone < 1:
        raise ValueError('Invalid joystick deadzone')
    if min(forward_axis, yaw_axis) < 0 or len(axes) <= max(forward_axis, yaw_axis):
        raise ValueError('Missing configured joystick axis')
    values = (axes[forward_axis], axes[yaw_axis])
    if not all(math.isfinite(v) and abs(v) <= 1.001 for v in values):
        raise ValueError('Invalid joystick axis value')

    def normalized(value):
        value = max(-1., min(1., value))
        if abs(value) <= deadzone:
            return 0.
        return math.copysign((abs(value) - deadzone) / (1. - deadzone), value)

    forward, yaw = map(normalized, values)
    # Match the existing teleop axis signs. Never interpret neutral as stop.
    vx = max(0.45, min(0.75, 0.60 + 0.15 * forward))
    # The training commands are zero or |yaw| >= 0.02 rad/s.
    yaw_rate = 0. if yaw == 0 else math.copysign(0.02 + 0.05 * abs(yaw), yaw)
    return vx, 0., yaw_rate
