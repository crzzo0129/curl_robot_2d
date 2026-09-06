"""Small, event-gated single-foot repositioning probe; not a learned skill."""
from dataclasses import dataclass
import numpy as np
import mujoco

from curl_robot_2d_mjx.stand_compact_wbc_3d import (
    StandCompactWbc3D, StandCompactWbcConfig, LEGS, smootherstep, quaternion_error,
)


def triangle_margin(point, vertices):
    vertices = np.asarray(vertices)
    edges = np.roll(vertices, -1, axis=0) - vertices
    rel = np.asarray(point) - vertices
    crosses = edges[:, 0]*rel[:, 1] - edges[:, 1]*rel[:, 0]
    area = np.sum(vertices[:, 0]*np.roll(vertices[:, 1], -1) -
                  vertices[:, 1]*np.roll(vertices[:, 0], -1))
    return float(np.min(np.sign(area)*crosses / np.maximum(np.linalg.norm(edges, axis=1), 1e-12)))


def triangle_incenter(vertices):
    vertices = np.asarray(vertices)
    lengths = np.linalg.norm(np.roll(vertices, 1, axis=0) - np.roll(vertices, -1, axis=0), axis=1)
    return np.sum(vertices * lengths[:, None], axis=0) / np.maximum(lengths.sum(), 1e-12)


@dataclass(frozen=True)
class SingleFootConfig:
    step_m: float = .008
    lift_m: float = .015
    transfer_s: float = 1.0
    transfer_timeout_s: float = 4.0
    lift_s: float = .45
    move_s: float = .6
    lower_s: float = .6
    touchdown_timeout_s: float = 1.0
    support_margin_m: float = .004
    unload_force_n: float = 3.0
    touchdown_force_n: float = 3.0
    integral_gain: float = 1.5
    rounds: int = 1


class SingleFootCompactProbe:
    order = (0, 3, 1, 2)

    def __init__(self, model, config=SingleFootConfig()):
        self.model, self.config = model, config
        self.ik = StandCompactWbc3D(model, StandCompactWbcConfig())
        self.torso = model.body('torso').id
        self.phase, self.failed, self.done = 'uninitialized', False, False
        self.reason = ''
        self.events = []

    def reset(self, data):
        self.ik.reset(data)
        self.feet = data.site_xpos[self.ik.site_ids].copy()
        self.initial_feet = self.feet.copy()
        self.nominal_root = data.qpos[:7].copy()
        self.root_goal = self.nominal_root.copy()
        self.previous_root = self.root_goal.copy()
        self.command = data.ctrl.copy()
        self.integral = np.zeros(12)
        self.completed = 0
        self.elapsed = 0.
        self.ready_frames = 0
        self.support_margin = 0.
        self.peak_clearance = 0.
        self.begin_transfer(data)

    @property
    def active_leg(self):
        return self.order[self.completed % 4]

    def enter(self, phase, data):
        self.phase, self.elapsed, self.ready_frames = phase, 0., 0
        self.events.append({'time_s': float(data.time), 'phase': phase,
                            'leg': LEGS[self.active_leg], 'completed_steps': self.completed})

    def fail(self, reason, data):
        self.failed, self.reason = True, reason
        self.enter('failed', data)

    def begin_transfer(self, data):
        leg = self.active_leg
        self.previous_root = self.root_goal.copy()
        others = [i for i in range(4) if i != leg]
        center = triangle_incenter(self.feet[others, :2])
        com_offset = data.subtree_com[self.torso, :2] - data.qpos[:2]
        self.root_goal = self.nominal_root.copy()
        requested_shift = .35*(center-com_offset-self.nominal_root[:2])
        requested_shift *= min(1., .025/max(np.linalg.norm(requested_shift), 1e-12))
        self.root_goal[:2] += requested_shift
        # Small world-x inward motion; retain lateral foot spacing.
        self.foot_start = data.site_xpos[self.ik.site_ids[leg]].copy()
        self.foot_goal = self.foot_start.copy()
        target_x = self.ik.final_feet[leg, 0]
        self.foot_goal[0] += np.clip(target_x-self.foot_start[0], -self.config.step_m, self.config.step_m)
        self.peak_clearance = 0.
        self.enter('transfer', data)

    def step(self, data, dt=.02):
        if self.failed or self.done:
            return self.command.copy()
        c, leg = self.config, self.active_leg
        self.elapsed += dt
        contact, forces = self.ik.contact_state(data)
        others = [i for i in range(4) if i != leg]
        positions = data.site_xpos[self.ik.site_ids]
        self.support_margin = triangle_margin(data.subtree_com[self.torso, :2], positions[others, :2])
        velocities = np.array([self.ik._site_velocity(data, i) for i in range(4)])
        clearance = positions[leg, 2] - self.model.geom_size[self.ik.foot_geom_ids[leg], 0]
        self.peak_clearance = max(self.peak_clearance, float(clearance))
        stable = np.max(np.abs(data.qvel[3:6])) < .3
        if self.phase == 'transfer':
            ready = (self.elapsed >= c.transfer_s and self.support_margin >= c.support_margin_m
                     and np.all(forces[others] > 2.) and stable)
            self.ready_frames = self.ready_frames+1 if ready else 0
            if self.ready_frames >= 3:
                self.foot_start = positions[leg].copy()
                # Weight transfer can roll the spherical foot without sliding.
                # Plan the 8 mm swing from its measured unload location.
                self.foot_goal = self.foot_start.copy()
                self.foot_goal[0] += np.clip(self.ik.final_feet[leg,0]-self.foot_start[0],
                                            -c.step_m, c.step_m)
                self.enter('unload', data)
            elif self.elapsed >= c.transfer_timeout_s:
                self.fail('weight_transfer_not_confirmed', data)
        elif self.phase == 'unload':
            ready = forces[leg] <= c.unload_force_n and np.all(forces[others] > 2.) and self.support_margin > 0 and stable
            self.ready_frames = self.ready_frames+1 if ready else 0
            if self.ready_frames >= 3:
                self.enter('lift', data)
            elif self.elapsed >= 1.2:
                self.fail('foot_not_unloaded', data)
        elif self.phase == 'lift':
            if self.elapsed >= c.lift_s:
                if clearance >= .008 and forces[leg] < .5 and self.support_margin > 0:
                    self.enter('move', data)
                elif self.elapsed >= c.lift_s + .8:
                    self.fail('foot_not_clear', data)
        elif self.phase == 'move' and self.elapsed >= c.move_s:
            self.enter('lower', data)
        elif self.phase == 'lower' and self.elapsed >= c.lower_s:
            self.enter('touchdown', data)
        elif self.phase == 'touchdown':
            ready = (forces[leg] >= c.touchdown_force_n and
                     np.linalg.norm(positions[leg, :2] - self.foot_goal[:2]) < .004 and
                     np.linalg.norm(velocities[leg]) < .04 and stable)
            self.ready_frames = self.ready_frames+1 if ready else 0
            if self.ready_frames >= 3:
                self.feet[leg] = positions[leg]
                self.events.append({'time_s': float(data.time), 'phase': 'step_confirmed',
                    'leg': LEGS[leg], 'actual_dx_m': float(positions[leg,0]-self.foot_start[0]),
                    'peak_clearance_m': self.peak_clearance})
                self.completed += 1
                if self.completed >= 4*c.rounds:
                    self.done = True
                    self.enter('complete', data)
                else:
                    self.begin_transfer(data)
            elif self.elapsed >= c.touchdown_timeout_s:
                self.fail('touchdown_not_confirmed', data)
        if self.failed or self.done:
            return self.command.copy()
        leg = self.active_leg
        root = self.root_goal.copy()
        if self.phase == 'transfer':
            blend = smootherstep(self.elapsed / c.transfer_s)
            root = self.previous_root + blend*(self.root_goal-self.previous_root)
        targets = self.feet.copy()
        if self.phase in ('unload', 'lift', 'move', 'lower', 'touchdown'):
            targets[leg] = self.foot_start
            if self.phase == 'unload':
                targets[leg, 2] += .004*smootherstep(self.elapsed/.4)
            elif self.phase == 'lift':
                targets[leg, 2] += .004+(c.lift_m-.004)*smootherstep(self.elapsed/c.lift_s)
            elif self.phase == 'move':
                blend = smootherstep(self.elapsed/c.move_s)
                targets[leg, :2] += blend*(self.foot_goal[:2]-self.foot_start[:2])
                targets[leg, 2] += c.lift_m
            else:
                targets[leg] = self.foot_goal
                blend = smootherstep(self.elapsed/c.lower_s) if self.phase == 'lower' else 1.
                targets[leg, 2] += c.lift_m*(1-blend) - .002*blend
        requested = self.command.copy()
        for i in range(4):
            requested[3*i:3*i+3] = self.ik._leg_inverse_kinematics(root, targets[i], i, requested)
        # Bounded integral compensates quasi-static servo load error; it acts
        # only on motor commands, never the free root or measured joint state.
        self.integral = np.clip(self.integral + c.integral_gain*dt*(
            requested - data.qpos[self.ik.qpos_indices]), -.20, .20)
        requested += self.integral
        self.command = np.clip(self.command + np.clip(requested-self.command, -.02, .02),
                               self.ik.ctrl_low, self.ik.ctrl_high)
        return self.command.copy()
