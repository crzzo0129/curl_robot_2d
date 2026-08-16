"""Run a phase-based handcrafted gait on the 12-DoF Pupper model.

This is a physics feasibility diagnostic, not a training controller.  It uses
analytic sagittal-leg IK and diagonal trot phases, then saves a rollout that
can be rendered by ``scripts.render_mjx_3d_policy``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS
from curl_robot_2d_mjx.config_walking_3d import (
    Walking3DConfig,
    walking_physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import apply_physics_options_3d
from curl_robot_2d_mjx.environment_walking_3d import WALKING_MODEL_PATH_3D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "handcrafted_3d_walk"
LEG_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
TROT_PHASE_OFFSETS = (0.0, 0.5, 0.5, 0.0)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def sagittal_leg_ik(outward_x_m: float, depth_m: float) -> tuple[float, float]:
    """Return hip/knee angles for a leg whose positive x points outward."""

    geometry = PUPPER_ORIGINAL_SHELL_60_PARAMETERS
    upper = geometry.upper_length
    lower = geometry.lower_length
    radius = float(np.clip(
        math.hypot(outward_x_m, depth_m),
        abs(upper - lower) + 1.0e-6,
        upper + lower - 1.0e-6,
    ))
    knee_cos = (radius * radius - upper * upper - lower * lower) / (
        2.0 * upper * lower
    )
    knee = math.acos(float(np.clip(knee_cos, -1.0, 1.0)))
    direction = math.atan2(outward_x_m, depth_m)
    hip = direction + math.atan2(
        lower * math.sin(knee),
        upper + lower * math.cos(knee),
    )
    return hip, knee


def foot_cycle(
    phase: float,
    *,
    step_length_m: float,
    lift_m: float,
    duty_factor: float,
) -> tuple[float, float, bool]:
    """Return fore-aft offset, upward lift, and stance flag."""

    phase %= 1.0
    if phase < duty_factor:
        progress = phase / duty_factor
        return step_length_m * (0.5 - progress), 0.0, True
    progress = (phase - duty_factor) / (1.0 - duty_factor)
    blend = _smoothstep(progress)
    offset = step_length_m * (blend - 0.5)
    lift = lift_m * math.sin(math.pi * blend)
    return offset, lift, False


def _joint_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos = []
    actuators = []
    for leg in LEG_NAMES:
        for joint in ("hip_abduction", "hip", "knee"):
            name = f"{leg}_{joint}"
            qpos.append(int(model.joint(name).qposadr[0]))
            actuators.append(int(model.actuator(f"{name}_servo").id))
    return np.asarray(qpos, dtype=int), np.asarray(actuators, dtype=int)


def joint_direction_report(model: mujoco.MjModel) -> dict[str, dict[str, list[float]]]:
    """Finite-difference foot motion caused by positive hip/knee rotation."""

    data = mujoco.MjData(model)
    key_id = model.key("stand").id
    torso_id = model.body("torso").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    rotation = data.xmat[torso_id].reshape(3, 3)
    torso_position = data.xpos[torso_id].copy()
    base = {
        leg: (
            data.site_xpos[model.site(f"{leg}_foot_site").id]
            - torso_position
        )
        @ rotation
        for leg in LEG_NAMES
    }
    report: dict[str, dict[str, list[float]]] = {}
    delta = 0.05
    for leg in LEG_NAMES:
        report[leg] = {}
        for joint in ("hip", "knee"):
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            qpos_index = int(model.joint(f"{leg}_{joint}").qposadr[0])
            data.qpos[qpos_index] += delta
            mujoco.mj_forward(model, data)
            rotation = data.xmat[torso_id].reshape(3, 3)
            torso_position = data.xpos[torso_id]
            local = (
                data.site_xpos[model.site(f"{leg}_foot_site").id]
                - torso_position
            ) @ rotation
            report[leg][f"positive_{joint}_foot_derivative_xyz_m_per_rad"] = (
                np.round((local - base[leg]) / delta, 6).tolist()
            )
    return report


def gait_targets(
    time_s: float,
    nominal: np.ndarray,
    *,
    frequency_hz: float,
    step_length_m: float,
    lift_m: float,
    duty_factor: float,
    ramp_s: float,
    pitch_rad: float,
    pitch_rate_rad_s: float,
    pitch_gain_m_per_rad: float,
    pitch_rate_gain_m_s_per_rad: float,
) -> tuple[np.ndarray, tuple[bool, ...]]:
    geometry = PUPPER_ORIGINAL_SHELL_60_PARAMETERS
    nominal_hip = float(nominal[1])
    nominal_knee = float(nominal[2])
    upper = geometry.upper_length
    lower = geometry.lower_length
    nominal_outward = (
        upper * math.sin(nominal_hip)
        + lower * math.sin(nominal_hip - nominal_knee)
    )
    nominal_depth = (
        upper * math.cos(nominal_hip)
        + lower * math.cos(nominal_hip - nominal_knee)
    )
    pitch_placement = float(np.clip(
        pitch_gain_m_per_rad * pitch_rad
        + pitch_rate_gain_m_s_per_rad * pitch_rate_rad_s,
        -0.025,
        0.025,
    ))
    targets = nominal.copy()
    stance_flags: list[bool] = []
    cycle = time_s * frequency_hz
    for leg_index, (leg, phase_offset) in enumerate(
        zip(LEG_NAMES, TROT_PHASE_OFFSETS)
    ):
        fore_aft, lift, stance = foot_cycle(
            cycle + phase_offset,
            step_length_m=step_length_m,
            lift_m=lift_m,
            duty_factor=duty_factor,
        )
        stance_flags.append(stance)
        world_x_offset = fore_aft + pitch_placement
        outward_sign = 1.0 if leg.startswith("front") else -1.0
        outward = nominal_outward + outward_sign * world_x_offset
        hip, knee = sagittal_leg_ik(outward, nominal_depth - lift)
        base = 3 * leg_index
        targets[base] = 0.0
        targets[base + 1] = hip
        targets[base + 2] = knee
    blend = _smoothstep(time_s / max(ramp_s, 1.0e-6))
    return nominal + blend * (targets - nominal), tuple(stance_flags)


def run_handcrafted_gait(args: argparse.Namespace) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(WALKING_MODEL_PATH_3D))
    task = walking_physics_profile_3d("cg12", Walking3DConfig())
    apply_physics_options_3d(model, task)
    data = mujoco.MjData(model)
    key_id = model.key("stand").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    joint_qpos, actuator_ids = _joint_indices(model)
    nominal = np.asarray(model.key_ctrl[key_id, actuator_ids], dtype=float)
    torso_id = model.body("torso").id
    floor_id = model.geom("floor").id
    foot_geom_ids = {
        model.geom(f"{leg}_foot_proxy").id for leg in LEG_NAMES
    }
    control_steps = max(1, round(task.control_timestep / model.opt.timestep))
    requested_steps = round(args.duration / model.opt.timestep)
    initial_x = float(data.qpos[0])
    initial_y = float(data.qpos[1])
    qpos_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    contact_rows: list[float] = []
    maximum_tilt = 0.0
    minimum_z = float(data.qpos[2])
    nonfoot_contact_steps = 0
    failed = False
    failure_reason = ""
    target = nominal.copy()

    for physics_step in range(requested_steps):
        if physics_step % control_steps == 0:
            rotation = data.xmat[torso_id].reshape(3, 3)
            pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
            pitch_rate = float(data.qvel[4])
            target, _ = gait_targets(
                float(data.time),
                nominal,
                frequency_hz=args.frequency,
                step_length_m=args.step_length,
                lift_m=args.lift,
                duty_factor=args.duty_factor,
                ramp_s=args.ramp,
                pitch_rad=pitch,
                pitch_rate_rad_s=pitch_rate,
                pitch_gain_m_per_rad=args.pitch_gain,
                pitch_rate_gain_m_s_per_rad=args.pitch_rate_gain,
            )
            target = np.clip(
                target,
                model.actuator_ctrlrange[actuator_ids, 0],
                model.actuator_ctrlrange[actuator_ids, 1],
            )
            data.ctrl[actuator_ids] = target
        mujoco.mj_step(model, data)
        if physics_step % control_steps != 0:
            continue

        rotation = data.xmat[torso_id].reshape(3, 3)
        tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
        maximum_tilt = max(maximum_tilt, tilt)
        minimum_z = min(minimum_z, float(data.qpos[2]))
        nonfoot = 0
        foot_contacts = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if floor_id not in pair:
                continue
            robot_geom = next(iter(pair - {floor_id}), floor_id)
            if robot_geom in foot_geom_ids:
                foot_contacts += 1
            else:
                nonfoot += 1
        nonfoot_contact_steps += int(nonfoot > 0)
        qpos_rows.append(np.asarray(data.qpos).copy())
        target_rows.append(target.copy())
        contact_rows.append(float(foot_contacts))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            failed, failure_reason = True, "nonfinite"
        elif data.qpos[2] < 0.145:
            failed, failure_reason = True, "root_low"
        elif tilt > 0.72:
            failed, failure_reason = True, "upright_tilt"
        elif nonfoot_contact_steps >= round(0.12 / task.control_timestep):
            failed, failure_reason = True, "nonfoot_contact"
        if failed:
            break

    qpos = np.asarray(qpos_rows)
    displacement = float(data.qpos[0]) - initial_x
    elapsed = len(qpos_rows) * task.control_timestep
    summary: dict[str, object] = {
        "model_path": str(WALKING_MODEL_PATH_3D),
        "duration_s": elapsed,
        "requested_duration_s": args.duration,
        "failed": failed,
        "failure_reason": failure_reason,
        "root_x_displacement_m": displacement,
        "mean_forward_velocity_m_s": displacement / max(elapsed, 1.0e-6),
        "final_lateral_drift_m": float(data.qpos[1]) - initial_y,
        "minimum_root_z_m": minimum_z,
        "maximum_upright_tilt_rad": maximum_tilt,
        "mean_foot_contact_count": float(np.mean(contact_rows)),
        "controller": {
            "frequency_hz": args.frequency,
            "step_length_m": args.step_length,
            "lift_m": args.lift,
            "duty_factor": args.duty_factor,
            "ramp_s": args.ramp,
            "pitch_gain_m_per_rad": args.pitch_gain,
            "pitch_rate_gain_m_s_per_rad": args.pitch_rate_gain,
        },
        "joint_direction_report": joint_direction_report(model),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "evaluation_rollout.npz",
        qpos=qpos,
        action=np.asarray(target_rows),
        reward=np.zeros(len(qpos), dtype=float),
        mode=np.asarray("handcrafted_trot"),
    )
    (args.output / "handcrafted_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--frequency", type=float, default=1.6)
    parser.add_argument("--step-length", type=float, default=0.045)
    parser.add_argument("--lift", type=float, default=0.025)
    parser.add_argument("--duty-factor", type=float, default=0.68)
    parser.add_argument("--ramp", type=float, default=1.0)
    parser.add_argument("--pitch-gain", type=float, default=0.035)
    parser.add_argument("--pitch-rate-gain", type=float, default=0.006)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for name in ("duration", "frequency", "step_length", "lift", "ramp"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.5 < args.duty_factor < 1.0:
        parser.error("--duty-factor must lie in (0.5, 1.0)")
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps(run_handcrafted_gait(args), indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
