"""Cartesian whole-body controller for stand-to-compact transition.

This is a deterministic, non-learning controller.  A small equality-
constrained inverse kinematics keeps measured stance feet stationary while a
state machine relocates one foot at a time.  Only twelve joint
position/velocity targets are emitted; the floating base is never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


LEGS = ("front_left", "front_right", "rear_left", "rear_right")
JOINT_NAMES = tuple(
    f"{leg}_{joint}"
    for leg in LEGS
    for joint in ("hip_abduction", "hip", "knee")
)


@dataclass(frozen=True)
class StandCompactWbcConfig:
    control_timestep_s: float = 0.02
    stand_settle_s: float = 0.25
    weight_transfer_s: float = 0.30
    swing_s: float = 0.55
    touchdown_timeout_s: float = 0.25
    final_capture_timeout_s: float = 1.0
    required_hold_s: float = 0.10
    foot_lift_m: float = 0.025
    support_shift_fraction: float = 0.30
    support_shift_limit_m: float = 0.025
    touchdown_contact_frames: int = 3
    touchdown_force_n: float = 3.0
    touchdown_vertical_speed_m_s: float = 0.08
    touchdown_position_tolerance_m: float = 0.025
    base_position_gain: float = 3.0
    base_linear_damping: float = 1.2
    base_orientation_gain: float = 4.0
    base_angular_damping: float = 1.5
    swing_position_gain: float = 8.0
    posture_gain: float = 2.5
    base_linear_weight: float = 8.0
    base_angular_weight: float = 12.0
    swing_weight: float = 30.0
    posture_weight: float = 1.0
    regularization_weight: float = 0.03
    stance_constraint_damping: float = 1.0e-7
    stance_position_gain: float = 2.0
    maximum_stance_correction_m_s: float = 0.01
    maximum_joint_velocity_rad_s: float = 1.2
    command_lookahead_s: float = 0.10
    maximum_command_step_rad: float = 0.035
    joint_position_tolerance_rad: float = 0.02
    joint_velocity_tolerance_rad_s: float = 0.05
    root_height_tolerance_m: float = 0.01
    root_linear_velocity_tolerance_m_s: float = 0.02
    root_angular_velocity_tolerance_rad_s: float = 0.10
    orientation_tolerance_rad: float = 0.05
    maximum_tilt_rad: float = 0.35
    maximum_angular_speed_rad_s: float = 1.5

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if name == "support_shift_fraction":
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("support_shift_fraction must be in [0, 1]")
            elif name == "touchdown_contact_frames":
                if not isinstance(value, int) or value < 1:
                    raise ValueError("touchdown_contact_frames must be positive")
            elif not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class WbcOutput:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    phase: str
    swing_legs: tuple[int, ...] | None
    phase_progress: float
    expected_stance_feet: tuple[int, ...]
    measured_contact: np.ndarray
    qp_stance_residual_m_s: float
    successful: bool
    failed: bool
    failure_reason: str
    compact_hold_s: float


def smootherstep(value: float) -> float:
    u = float(np.clip(value, 0.0, 1.0))
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def smootherstep_derivative(value: float) -> float:
    u = float(np.clip(value, 0.0, 1.0))
    return 30.0 * u**2 * (1.0 - u) ** 2


def swing_bump(value: float) -> float:
    u = float(np.clip(value, 0.0, 1.0))
    return 64.0 * u**3 * (1.0 - u) ** 3


def swing_bump_derivative(value: float) -> float:
    u = float(np.clip(value, 0.0, 1.0))
    return 192.0 * u**2 * (1.0 - u) ** 2 * (1.0 - 2.0 * u)


def quaternion_error(desired: np.ndarray, measured: np.ndarray) -> np.ndarray:
    result = np.empty(3)
    mujoco.mju_subQuat(result, desired, measured)
    return result


def solve_equality_qp(
    task_matrix: np.ndarray,
    task_target: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_target: np.ndarray,
    *,
    regularization: float,
    constraint_damping: float,
) -> np.ndarray:
    """Solve min ||Ax-b||² + reg||x||² subject to Cx=d."""

    variables = task_matrix.shape[1]
    hessian = task_matrix.T @ task_matrix + regularization * np.eye(variables)
    gradient = task_matrix.T @ task_target
    if not len(constraint_matrix):
        return np.linalg.solve(hessian, gradient)
    constraints = constraint_matrix.shape[0]
    kkt = np.block([
        [hessian, constraint_matrix.T],
        [constraint_matrix, -constraint_damping * np.eye(constraints)],
    ])
    rhs = np.concatenate((gradient, constraint_target))
    solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=1.0e-10)
    return solution[:variables]


class StandCompactWbc3D:
    """Event-driven 12-joint stand-to-low-speed-compact controller."""

    swing_groups = ((0,), (3,), (1,), (2,))

    def __init__(self, model: mujoco.MjModel, config=StandCompactWbcConfig()):
        config.validate()
        if (model.nq, model.nv, model.nu) != (19, 18, 12):
            raise ValueError("controller requires the 19/18/12 rollingquad model")
        self.model = model
        self.config = config
        self.joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES])
        self.qpos_indices = np.asarray(model.jnt_qposadr[self.joint_ids], dtype=int)
        self.dof_indices = np.asarray(model.jnt_dofadr[self.joint_ids], dtype=int)
        self.actuator_ids = np.asarray([
            model.actuator(f"{name}_servo").id for name in JOINT_NAMES
        ])
        if not np.array_equal(self.actuator_ids, np.arange(12)):
            raise ValueError("unexpected actuator order")
        self.site_ids = np.asarray([model.site(f"{leg}_foot_site").id for leg in LEGS])
        self.foot_geom_ids = np.asarray([model.geom(f"{leg}_foot_proxy").id for leg in LEGS])
        self.floor_geom_id = model.geom("floor").id
        self.ctrl_low = model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_high = model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.stand_target = model.key("stand").ctrl[self.actuator_ids].copy()
        self.compact_target = model.key("compact").ctrl[self.actuator_ids].copy()
        self.compact_key_qpos = model.key("compact").qpos.copy()
        self._template_compact_feet = self._key_feet("compact")
        self._ik_data = mujoco.MjData(model)
        self._initialized = False

    def _key_feet(self, key_name: str) -> np.ndarray:
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key(key_name).id)
        mujoco.mj_forward(self.model, data)
        return data.site_xpos[self.site_ids].copy()

    def reset(self, data: mujoco.MjData) -> None:
        mujoco.mj_forward(self.model, data)
        self.start_root = data.qpos[:7].copy()
        self.start_feet = data.site_xpos[self.site_ids].copy()
        self.foot_targets = self.start_feet.copy()
        relative = self._template_compact_feet - self.compact_key_qpos[:3]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.start_root[3:7])
        rotation = rotation.reshape(3, 3)
        self.final_feet = self.start_root[:3] + relative @ rotation.T
        self.final_feet[:, 2] = self.start_feet[:, 2]
        clearance = self.compact_key_qpos[2] - float(np.mean(self._template_compact_feet[:, 2]))
        self.final_root = self.start_root.copy()
        self.final_root[2] = float(np.mean(self.start_feet[:, 2])) + clearance
        self.phase = "settle_stand"
        self.phase_time = 0.0
        self.swing_index = 0
        self.active_group = None
        self.previous_root_goal = self.start_root.copy()
        self.current_root_goal = self.start_root.copy()
        self.swing_start = {}
        self.touchdown_frames = 0
        self.compact_hold_s = 0.0
        self.failed = False
        self.failure_reason = ""
        self.successful = False
        self.last_command = data.qpos[self.qpos_indices].copy()
        self._initialized = True

    def contact_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        contact = np.zeros(4, dtype=bool)
        normal_force = np.zeros(4)
        for index in range(data.ncon):
            item = data.contact[index]
            pair = (int(item.geom[0]), int(item.geom[1]))
            if self.floor_geom_id not in pair:
                continue
            other = pair[1] if pair[0] == self.floor_geom_id else pair[0]
            matches = np.flatnonzero(self.foot_geom_ids == other)
            if not len(matches):
                continue
            leg = int(matches[0])
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            contact[leg] = True
            normal_force[leg] += max(float(force[0]), 0.0)
        return contact, normal_force

    def _support_goal(self, group: tuple[int, ...]) -> np.ndarray:
        stance = [index for index in range(4) if index not in group]
        centroid = np.mean(self.foot_targets[stance, :2], axis=0)
        offset = self.config.support_shift_fraction * (centroid - self.start_root[:2])
        norm = float(np.linalg.norm(offset))
        if norm > self.config.support_shift_limit_m:
            offset *= self.config.support_shift_limit_m / norm
        result = self.start_root.copy()
        result[:2] += offset
        return result

    def _advance_state(self, data: mujoco.MjData, contact: np.ndarray, force: np.ndarray, dt: float) -> None:
        self.phase_time += dt
        if self.phase == "settle_stand" and self.phase_time >= self.config.stand_settle_s:
            self.active_group = self.swing_groups[self.swing_index]
            self.previous_root_goal = self.current_root_goal.copy()
            self.current_root_goal = self._support_goal(self.active_group)
            self.phase = "transfer_weight"
            self.phase_time = 0.0
        elif self.phase == "transfer_weight" and self.phase_time >= self.config.weight_transfer_s:
            self.swing_start = {
                leg: data.site_xpos[self.site_ids[leg]].copy()
                for leg in self.active_group
            }
            self.phase = "swing"
            self.phase_time = 0.0
        elif self.phase == "swing" and self.phase_time >= self.config.swing_s:
            self.phase = "confirm_touchdown"
            self.phase_time = 0.0
            self.touchdown_frames = 0
        elif self.phase == "confirm_touchdown":
            landed = True
            for leg in self.active_group:
                site_velocity = self._site_velocity(data, leg)
                site_position = data.site_xpos[self.site_ids[leg]]
                touchdown_geometry = (
                    (contact[leg] and force[leg] >= self.config.touchdown_force_n)
                    or site_position[2] <= self.final_feet[leg, 2] + 0.003
                )
                landed = landed and (
                    touchdown_geometry
                    and abs(float(site_velocity[2])) <= self.config.touchdown_vertical_speed_m_s
                    and float(np.linalg.norm(site_position - self.final_feet[leg]))
                    <= self.config.touchdown_position_tolerance_m
                )
            self.touchdown_frames = self.touchdown_frames + 1 if landed else 0
            if self.touchdown_frames >= self.config.touchdown_contact_frames:
                for leg in self.active_group:
                    self.foot_targets[leg] = self.final_feet[leg]
                self.swing_index += 1
                if self.swing_index == len(self.swing_groups):
                    self.phase = "capture_compact"
                    self.active_group = None
                else:
                    self.active_group = self.swing_groups[self.swing_index]
                    self.previous_root_goal = self.current_root_goal.copy()
                    self.current_root_goal = self._support_goal(self.active_group)
                    self.phase = "transfer_weight"
                self.phase_time = 0.0
            elif self.phase_time >= self.config.touchdown_timeout_s:
                self.failed = True
                label = "+".join(LEGS[leg] for leg in self.active_group)
                self.failure_reason = f"{label} touchdown timeout"
        elif self.phase == "capture_compact":
            eligible = self._compact_eligible(data)
            self.compact_hold_s = self.compact_hold_s + dt if eligible else 0.0
            if self.compact_hold_s + 1.0e-12 >= self.config.required_hold_s:
                self.successful = True
                self.phase = "success"
            elif self.phase_time >= self.config.final_capture_timeout_s:
                self.failed = True
                self.failure_reason = "compact capture timeout"

        tilt = float(np.linalg.norm(quaternion_error(self.start_root[3:7], data.qpos[3:7])[:2]))
        if not self.successful and (
            tilt > self.config.maximum_tilt_rad
            or float(np.max(np.abs(data.qvel[3:6]))) > self.config.maximum_angular_speed_rad_s
        ):
            self.failed = True
            self.failure_reason = "body attitude safety limit"
        if self.failed:
            self.phase = "failed"

    def _site_jacobian(self, data: mujoco.MjData, leg: int) -> np.ndarray:
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacobian, None, int(self.site_ids[leg]))
        return jacobian

    def _site_velocity(self, data: mujoco.MjData, leg: int) -> np.ndarray:
        return self._site_jacobian(data, leg) @ data.qvel

    def _base_target(self, data: mujoco.MjData) -> np.ndarray:
        if self.phase == "transfer_weight" or self.phase == "swing" or self.phase == "confirm_touchdown":
            desired = self._support_goal(self.active_group)
        elif self.phase == "capture_compact" or self.phase == "success":
            desired = self.final_root
        else:
            desired = self.start_root
        linear = (
            self.config.base_position_gain * (desired[:3] - data.qpos[:3])
            - self.config.base_linear_damping * data.qvel[:3]
        )
        angular = (
            self.config.base_orientation_gain * quaternion_error(desired[3:7], data.qpos[3:7])
            - self.config.base_angular_damping * data.qvel[3:6]
        )
        return np.concatenate((np.clip(linear, -0.15, 0.15), np.clip(angular, -0.4, 0.4)))

    def _desired_root_pose(self) -> np.ndarray:
        """Return the virtual floating-base pose used only by inverse kinematics."""

        if self.phase == "settle_stand":
            return self.start_root.copy()
        if self.phase == "transfer_weight":
            blend = smootherstep(self.phase_time / self.config.weight_transfer_s)
            result = self.previous_root_goal + blend * (
                self.current_root_goal - self.previous_root_goal
            )
            # The support goals preserve orientation, so normalized linear
            # quaternion interpolation is exact here and guards future edits.
            result[3:7] /= np.linalg.norm(result[3:7])
            return result
        if self.phase in ("swing", "confirm_touchdown"):
            return self.current_root_goal.copy()
        if self.phase in ("capture_compact", "success"):
            blend = smootherstep(
                min(self.phase_time, 0.65) / 0.65
                if self.phase == "capture_compact" else 1.0
            )
            result = self.current_root_goal + blend * (
                self.final_root - self.current_root_goal
            )
            result[3:7] /= np.linalg.norm(result[3:7])
            return result
        return self.current_root_goal.copy()

    def _desired_feet(self) -> np.ndarray:
        feet = self.foot_targets.copy()
        if self.phase in ("swing", "confirm_touchdown"):
            for leg in self.active_group:
                target, _ = self._swing_target(self._ik_data, leg)
                feet[leg] = target
        return feet

    def _leg_inverse_kinematics(
        self,
        root_pose: np.ndarray,
        foot_position: np.ndarray,
        leg: int,
        seed: np.ndarray,
    ) -> np.ndarray:
        """Damped Newton IK for one independent 3-DoF leg."""

        data = self._ik_data
        data.qpos[:] = self.compact_key_qpos
        data.qpos[:7] = root_pose
        data.qpos[self.qpos_indices] = seed
        dofs = self.dof_indices[3 * leg:3 * leg + 3]
        qpos = self.qpos_indices[3 * leg:3 * leg + 3]
        site = int(self.site_ids[leg])
        for _ in range(12):
            mujoco.mj_forward(self.model, data)
            error = foot_position - data.site_xpos[site]
            if float(np.linalg.norm(error)) < 2.0e-6:
                break
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jacobian, None, site)
            block = jacobian[:, dofs]
            damping = 2.0e-6 * np.eye(3)
            delta = block.T @ np.linalg.solve(block @ block.T + damping, error)
            data.qpos[qpos] += np.clip(delta, -0.12, 0.12)
            data.qpos[qpos] = np.clip(
                data.qpos[qpos],
                self.model.jnt_range[self.joint_ids[3 * leg:3 * leg + 3], 0],
                self.model.jnt_range[self.joint_ids[3 * leg:3 * leg + 3], 1],
            )
        return data.qpos[qpos].copy()

    def _inverse_kinematics_command(self) -> np.ndarray:
        root = self._desired_root_pose()
        # _swing_target reads measured site positions for feedback.  Seed the
        # scratch data with the virtual root/current command before sampling it.
        self._ik_data.qpos[:] = self.compact_key_qpos
        self._ik_data.qpos[:7] = root
        self._ik_data.qpos[self.qpos_indices] = self.last_command
        mujoco.mj_forward(self.model, self._ik_data)
        feet = self._desired_feet()
        command = self.last_command.copy()
        for leg in range(4):
            solved = self._leg_inverse_kinematics(root, feet[leg], leg, command)
            command[3 * leg:3 * leg + 3] = solved
        if self.phase == "success":
            command = self.compact_target.copy()
        return np.clip(command, self.ctrl_low, self.ctrl_high)

    def _swing_target(self, data: mujoco.MjData, leg: int) -> tuple[np.ndarray, np.ndarray]:
        goal = self.final_feet[leg]
        if self.phase == "swing":
            raw = np.clip(self.phase_time / self.config.swing_s, 0.0, 1.0)
            blend = smootherstep(raw)
            start = self.swing_start[leg]
            target = start + blend * (goal - start)
            target[2] += self.config.foot_lift_m * swing_bump(raw)
            velocity = (
                smootherstep_derivative(raw) * (goal - start)
                + np.asarray((0.0, 0.0, self.config.foot_lift_m * swing_bump_derivative(raw)))
            ) / self.config.swing_s
        else:
            target = goal.copy()
            velocity = np.zeros(3)
            if self.phase == "confirm_touchdown":
                target[2] -= 0.003
        measured = data.site_xpos[self.site_ids[leg]]
        return target, velocity + self.config.swing_position_gain * (target - measured)

    def _posture_goal(self) -> np.ndarray:
        goal = self.stand_target.copy()
        completed = tuple(
            leg for group in self.swing_groups[:self.swing_index] for leg in group
        )
        active = () if self.active_group is None else self.active_group
        for leg in completed + tuple(active):
            goal[3 * leg:3 * leg + 3] = self.compact_target[3 * leg:3 * leg + 3]
        if self.phase in ("capture_compact", "success"):
            goal[:] = self.compact_target
        return goal

    def _expected_stance(self) -> tuple[int, ...]:
        if self.phase in ("swing", "confirm_touchdown"):
            return tuple(index for index in range(4) if index not in self.active_group)
        return (0, 1, 2, 3)

    def _solve_velocity(self, data: mujoco.MjData, contact: np.ndarray):
        rows = []
        targets = []

        def task(matrix, target, weight):
            scale = math.sqrt(weight)
            rows.append(scale * matrix)
            targets.append(scale * target)

        base_matrix = np.zeros((6, self.model.nv))
        base_matrix[:, :6] = np.eye(6)
        base_target = self._base_target(data)
        task(base_matrix[:3], base_target[:3], self.config.base_linear_weight)
        task(base_matrix[3:], base_target[3:], self.config.base_angular_weight)

        if self.phase in ("swing", "confirm_touchdown"):
            for leg in self.active_group:
                _, swing_velocity = self._swing_target(data, leg)
                task(self._site_jacobian(data, leg), swing_velocity, self.config.swing_weight)

        posture_matrix = np.zeros((12, self.model.nv))
        posture_matrix[np.arange(12), self.dof_indices] = 1.0
        posture_velocity = self.config.posture_gain * (
            self._posture_goal() - data.qpos[self.qpos_indices]
        )
        task(posture_matrix, posture_velocity, self.config.posture_weight)

        expected = self._expected_stance()
        stance_legs = [leg for leg in expected if contact[leg]]
        stance_rows = [self._site_jacobian(data, leg) for leg in stance_legs]
        constraints = np.vstack(stance_rows) if stance_rows else np.zeros((0, self.model.nv))
        desired_blocks = []
        for leg in stance_legs:
            correction = self.config.stance_position_gain * (
                self.foot_targets[leg] - data.site_xpos[self.site_ids[leg]]
            )
            norm = float(np.linalg.norm(correction))
            if norm > self.config.maximum_stance_correction_m_s:
                correction *= self.config.maximum_stance_correction_m_s / norm
            desired_blocks.append(correction)
        desired = np.concatenate(desired_blocks) if desired_blocks else np.zeros(0)
        velocity = solve_equality_qp(
            np.vstack(rows), np.concatenate(targets), constraints, desired,
            regularization=self.config.regularization_weight,
            constraint_damping=self.config.stance_constraint_damping,
        )
        maximum = float(np.max(np.abs(velocity[self.dof_indices])))
        if maximum > self.config.maximum_joint_velocity_rad_s:
            velocity *= self.config.maximum_joint_velocity_rad_s / maximum
        residual = float(np.max(np.abs(constraints @ velocity))) if len(constraints) else 0.0
        return velocity, residual, expected

    def _compact_eligible(self, data: mujoco.MjData) -> bool:
        c = self.config
        return (
            float(np.max(np.abs(data.qpos[self.qpos_indices] - self.compact_target)))
            <= c.joint_position_tolerance_rad
            and float(np.max(np.abs(data.qvel[self.dof_indices])))
            <= c.joint_velocity_tolerance_rad_s
            and abs(float(data.qpos[2] - self.final_root[2])) <= c.root_height_tolerance_m
            and float(np.max(np.abs(data.qvel[:3]))) <= c.root_linear_velocity_tolerance_m_s
            and float(np.max(np.abs(data.qvel[3:6]))) <= c.root_angular_velocity_tolerance_rad_s
            and float(np.linalg.norm(quaternion_error(self.final_root[3:7], data.qpos[3:7])))
            <= c.orientation_tolerance_rad
        )

    def step(self, data: mujoco.MjData, dt: float | None = None) -> WbcOutput:
        if not self._initialized:
            raise RuntimeError("call reset(data) before step")
        dt = self.config.control_timestep_s if dt is None else float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        contact, force = self.contact_state(data)
        if not self.failed and not self.successful:
            self._advance_state(data, contact, force, dt)

        if self.failed:
            joint_velocity = np.zeros(12)
            command = self.last_command.copy()
            residual = 0.0
            expected = self._expected_stance()
        else:
            requested = self._inverse_kinematics_command()
            delta = np.clip(
                requested - self.last_command,
                -self.config.maximum_command_step_rad,
                self.config.maximum_command_step_rad,
            )
            command = np.clip(self.last_command + delta, self.ctrl_low, self.ctrl_high)
            joint_velocity = delta / dt
            self.last_command = command.copy()
            residual = 0.0
            expected = self._expected_stance()

        progress = 1.0
        duration = 1.0
        if self.phase == "settle_stand":
            duration = self.config.stand_settle_s
        elif self.phase == "transfer_weight":
            duration = self.config.weight_transfer_s
        elif self.phase == "swing":
            duration = self.config.swing_s
        elif self.phase == "confirm_touchdown":
            duration = self.config.touchdown_timeout_s
        elif self.phase == "capture_compact":
            duration = self.config.final_capture_timeout_s
        if self.phase not in ("success", "failed"):
            progress = float(np.clip(self.phase_time / duration, 0.0, 1.0))
        return WbcOutput(
            joint_position=command,
            joint_velocity=joint_velocity,
            phase=self.phase,
            swing_legs=self.active_group,
            phase_progress=progress,
            expected_stance_feet=expected,
            measured_contact=contact,
            qp_stance_residual_m_s=residual,
            successful=self.successful,
            failed=self.failed,
            failure_reason=self.failure_reason,
            compact_hold_s=self.compact_hold_s,
        )
