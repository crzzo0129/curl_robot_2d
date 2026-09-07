"""Small, event-gated single-foot repositioning probe; not a learned skill."""
from dataclasses import asdict, dataclass
import math
import numpy as np
import mujoco

from curl_robot_2d_mjx.stand_compact_wbc_3d import (
    StandCompactWbc3D, StandCompactWbcConfig, LEGS, smootherstep,
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
    transfer_timeout_s: float = 6.0
    transfer_confirm_s: float = .12
    unload_timeout_s: float = 4.0
    unload_confirm_s: float = .16
    unload_lift_m: float = .002
    unload_ramp_s: float = 1.2
    lift_s: float = .8
    lift_clear_timeout_s: float = 3.5
    move_s: float = .6
    lower_s: float = .6
    touchdown_timeout_s: float = 2.5
    recenter_timeout_s: float = 4.0
    recenter_confirm_s: float = .30
    support_margin_m: float = .006
    transfer_foot_force_n: float = 4.0
    force_shift_gain_m: float = .005
    force_shift_limit_m: float = .015
    unload_force_n: float = .5
    touchdown_force_n: float = 3.0
    recenter_speed_m_s: float = .012
    support_force_gain_m_n_s: float = .0002
    support_force_offset_limit_m: float = .004
    integral_gain: float = 1.5
    rounds: int = 1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.rounds, int):
            raise ValueError('rounds must be an integer')


class SingleFootCompactProbe:
    # The front pair is the verified reachable prefix. Rear-leg lifting needs
    # a stance-constrained whole-body solver; safety gates stop before forcing it.
    order = (0, 1, 3, 2)

    def __init__(self, model, config=SingleFootConfig()):
        self.model, self.config = model, config
        self.ik = StandCompactWbc3D(model, StandCompactWbcConfig())
        self.torso = model.body('torso').id
        self.phase, self.failed, self.done = 'uninitialized', False, False
        self.reason = ''
        self.events = []

    def reset(self, data):
        self.failed, self.done, self.reason = False, False, ''
        self.events = []
        self.ik.reset(data)
        self.feet = data.site_xpos[self.ik.site_ids].copy()
        self.initial_feet = self.feet.copy()
        self.nominal_root = data.qpos[:7].copy()
        self.root_goal = self.nominal_root.copy()
        self.previous_root = self.root_goal.copy()
        self.command = data.ctrl.copy()
        self.integral = np.zeros(12)
        self.support_z_offsets = np.zeros(4)
        self.body_weight_n = float(np.sum(self.model.body_mass)*9.81)
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
                            'leg': LEGS[self.active_leg] if not self.done and phase not in ('recenter', 'complete') else None,
                            'completed_steps': self.completed,
                            'target_root_z_m': float(self.root_goal[2])})

    def fail(self, reason, data):
        self.failed, self.reason = True, reason
        self.enter('failed', data)
        leg = self.active_leg
        _, forces = self.ik.contact_state(data)
        self.events[-1].update(reason=reason, normal_force_n=float(forces[leg]),
            foot_position_error_m=(data.site_xpos[self.ik.site_ids[leg]]-self.foot_goal).tolist(),
            support_margin_m=self.support_margin)

    def begin_transfer(self, data):
        leg = self.active_leg
        self.support_z_offsets[:] = 0.
        self.integral[3*leg:3*leg+3] = 0.
        self.previous_root = self.root_goal.copy()
        others = [i for i in range(4) if i != leg]
        center = triangle_incenter(self.feet[others, :2])
        com_offset = data.subtree_com[self.torso, :2] - data.qpos[:2]
        self.root_goal = self.nominal_root.copy()
        height_progress = (self.completed + 1)/(4*self.config.rounds)
        self.root_goal[2] += height_progress*(self.ik.final_root[2]-self.nominal_root[2])
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
        stable = (np.max(np.abs(data.qvel[3:6])) < .20 and
                  np.linalg.norm(data.qvel[:2]) < .04)
        if self.phase == 'transfer':
            # Correct measured support error slowly while all feet remain down.
            # The initial geometric shift alone does not account for servo load
            # error or changes to the standing keyframe.
            if (self.elapsed >= c.transfer_s and
                    (self.support_margin < c.support_margin_m or
                     forces[leg] > c.transfer_foot_force_n)):
                center = triangle_incenter(positions[others, :2])
                away = center - positions[leg, :2]
                away /= max(np.linalg.norm(away), 1e-12)
                excess = max(0., forces[leg]/c.transfer_foot_force_n - 1.)
                force_bias = away*min(c.force_shift_limit_m, c.force_shift_gain_m*excess)
                error = center + force_bias - data.subtree_com[self.torso, :2]
                correction = error * min(1., .01*dt/max(np.linalg.norm(error), 1e-12))
                shift = self.root_goal[:2] + correction - self.nominal_root[:2]
                self.root_goal[:2] = self.nominal_root[:2] + shift*min(1., .07/max(np.linalg.norm(shift), 1e-12))
            ready = (self.elapsed >= c.transfer_s and self.support_margin >= c.support_margin_m
                     and forces[leg] <= c.transfer_foot_force_n
                     and np.all(forces[others] > 2.) and stable)
            self.ready_frames = self.ready_frames+1 if ready else 0
            if self.ready_frames*dt >= c.transfer_confirm_s:
                self.foot_start = positions[leg].copy()
                # Weight transfer can roll the spherical foot without sliding.
                # Plan the 8 mm swing from its measured unload location.
                self.foot_goal = self.foot_start.copy()
                self.foot_goal[0] += np.clip(self.ik.final_feet[leg,0]-self.foot_start[0],
                                            -c.step_m, c.step_m)
                self.integral[3*leg:3*leg+3] = 0.
                self.enter('lift', data)
            elif self.elapsed >= c.transfer_timeout_s:
                self.fail('weight_transfer_not_confirmed', data)
        elif self.phase == 'unload':
            ready = (forces[leg] <= c.unload_force_n and
                     np.all(forces[others] > 2.) and
                     self.support_margin >= c.support_margin_m and stable)
            self.ready_frames = self.ready_frames+1 if ready else 0
            if self.ready_frames*dt >= c.unload_confirm_s:
                self.enter('lift', data)
            elif self.elapsed >= c.unload_timeout_s:
                self.fail('foot_not_unloaded', data)
        elif self.phase == 'lift':
            if forces[leg] > c.unload_force_n or self.support_margin < c.support_margin_m:
                center = triangle_incenter(positions[others, :2])
                away = center - positions[leg, :2]
                away /= max(np.linalg.norm(away), 1e-12)
                excess = max(0., forces[leg]/c.unload_force_n - 1.)
                desired_com = center + away*min(c.force_shift_limit_m,
                                                 c.force_shift_gain_m*excess)
                error = desired_com - data.subtree_com[self.torso, :2]
                correction = error*min(1., .006*dt/max(np.linalg.norm(error), 1e-12))
                shift = self.root_goal[:2] + correction - self.nominal_root[:2]
                self.root_goal[:2] = self.nominal_root[:2] + shift*min(
                    1., .07/max(np.linalg.norm(shift), 1e-12))
            if self.elapsed >= c.lift_s:
                if clearance >= .008 and forces[leg] < .5 and self.support_margin > 0:
                    self.enter('move', data)
                elif self.elapsed >= c.lift_clear_timeout_s:
                    self.fail('foot_not_clear', data)
        elif self.phase == 'move' and self.elapsed >= c.move_s:
            self.enter('lower', data)
        elif self.phase == 'lower' and self.elapsed >= c.lower_s:
            self.enter('touchdown', data)
        elif self.phase == 'touchdown':
            ready = (forces[leg] >= c.touchdown_force_n and
                     np.linalg.norm(positions[leg, :2] - self.foot_goal[:2]) < .004 and
                     np.linalg.norm(velocities[leg]) < .04 and
                     self.support_margin > 0 and np.all(forces[others] > 2.) and stable)
            self.ready_frames = self.ready_frames+1 if ready else 0
            if self.ready_frames >= 3:
                self.feet[leg] = positions[leg]
                self.events.append({'time_s': float(data.time), 'phase': 'step_confirmed',
                    'leg': LEGS[leg], 'actual_dx_m': float(positions[leg,0]-self.foot_start[0]),
                    'peak_clearance_m': self.peak_clearance})
                self.completed += 1
                self.enter('recenter', data)
            elif self.elapsed >= c.touchdown_timeout_s:
                self.fail('touchdown_not_confirmed', data)
        elif self.phase == 'recenter':
            delta = self.nominal_root[:2] - self.root_goal[:2]
            self.root_goal[:2] += delta*min(1., c.recenter_speed_m_s*dt/max(np.linalg.norm(delta), 1e-12))
            ready = (self.elapsed >= .5 and np.all(forces > 2.) and stable and
                     np.linalg.norm(delta) < .005)
            self.ready_frames = self.ready_frames+1 if ready else 0
            if self.ready_frames*dt >= c.recenter_confirm_s:
                if self.completed >= 4*c.rounds:
                    self.done = True
                    self.enter('complete', data)
                else:
                    self.begin_transfer(data)
            elif self.elapsed >= c.recenter_timeout_s:
                self.fail('four_foot_recenter_not_confirmed', data)
        if self.failed or self.done:
            return self.command.copy()
        leg = self.active_leg
        root = self.root_goal.copy()
        if self.phase == 'touchdown':
            # Once the foot is down, return load to it gradually. Holding the
            # three-foot COM target can leave the landed foot almost unloaded.
            delta = self.nominal_root[:2] - self.root_goal[:2]
            self.root_goal[:2] += delta*min(1., c.recenter_speed_m_s*dt/max(np.linalg.norm(delta), 1e-12))
            root = self.root_goal.copy()
        if self.phase == 'transfer':
            blend = smootherstep(self.elapsed / c.transfer_s)
            root = self.previous_root + blend*(self.root_goal-self.previous_root)
        targets = self.feet.copy()
        if self.phase in ('lift', 'move', 'lower', 'touchdown'):
            support = [i for i in range(4) if i != leg]
            desired_force = self.body_weight_n/3.
            for i in support:
                # A lower world-z target extends that support leg and raises
                # its normal load. The small bounded trim supplies the force
                # distribution that pure body-position IK cannot determine.
                self.support_z_offsets[i] = np.clip(
                    self.support_z_offsets[i] -
                    c.support_force_gain_m_n_s*dt*(desired_force-forces[i]),
                    -c.support_force_offset_limit_m, c.support_force_offset_limit_m)
                targets[i, 2] += self.support_z_offsets[i]
        if self.phase in ('unload', 'lift', 'move', 'lower', 'touchdown'):
            targets[leg] = self.foot_start
            if self.phase == 'unload':
                targets[leg, 2] += c.unload_lift_m*smootherstep(self.elapsed/c.unload_ramp_s)
            elif self.phase == 'lift':
                targets[leg, 2] += c.lift_m*smootherstep(self.elapsed/c.lift_s)
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
