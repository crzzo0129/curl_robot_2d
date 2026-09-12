"""CPU replay of hardware/rolling_command/.../rolling_handoff.hpp (50 Hz)."""
import math
import numpy as np


class RollingHandoff:
    def __init__(self):
        self.ticks = self.healthy_ticks = 0
        self.alpha = 0.
        self.vx, self.yaw = .6, 0.
        self.stage = 'blending'
        self.stop_required = False

    @staticmethod
    def healthy(axis_z, roll_rate):
        return (math.isfinite(axis_z) and math.isfinite(roll_rate)
                and abs(axis_z) <= math.sin(math.radians(15)) and .5 <= abs(roll_rate) <= 12.)

    @classmethod
    def window(cls, pitch, axis_z, roll_rate, delta):
        return (cls.healthy(axis_z, roll_rate) and math.isfinite(pitch)
                and abs(pitch) <= math.pi/3 and math.isfinite(delta) and delta <= .12)

    def tick(self, stable, vx, yaw):
        self.ticks += 1
        if self.ticks <= 15:
            u = self.ticks/15
            self.alpha = u*u*(3-2*u)
            self.stage = 'blending'
            return
        self.alpha = 1.
        if self.stage in ('blending', 'settling'):
            self.healthy_ticks = self.healthy_ticks+1 if stable else 0
            self.stage = 'settling'
            if self.healthy_ticks < 15:
                self.stop_required = self.ticks >= 115
                return
            self.stage = 'command_ramp'
        self.vx = float(np.clip(vx, self.vx-.003, self.vx+.003))
        self.yaw = float(np.clip(yaw, self.yaw-.0014, self.yaw+.0014))
        self.stage = 'rolling' if abs(self.vx-vx) < 1e-9 and abs(self.yaw-yaw) < 1e-9 else 'command_ramp'


def effective_action(policy, target):
    result = np.zeros(12, dtype=np.float32)
    moving = policy.scale != 0
    result[moving] = np.clip((target[moving]-policy.center[moving])/policy.scale[moving], -1, 1)
    return result
