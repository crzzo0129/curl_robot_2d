"""Run the deterministic 12-joint stand-to-compact WBC in CPU MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d_mjx.stand_compact_wbc_3d import (
    JOINT_NAMES,
    LEGS,
    StandCompactWbc3D,
    StandCompactWbcConfig,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/rollingquad_description_2/mjcf/rollingquad.xml"


def contact_slip_speeds(model, data, foot_geom_ids, floor_id):
    speeds = np.zeros(4)
    for index in range(data.ncon):
        item = data.contact[index]
        pair = (int(item.geom[0]), int(item.geom[1]))
        if floor_id not in pair:
            continue
        other = pair[1] if pair[0] == floor_id else pair[0]
        matches = np.flatnonzero(foot_geom_ids == other)
        if not len(matches):
            continue
        body = int(model.geom_bodyid[other])
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, item.pos, body)
        velocity = jacobian @ data.qvel
        normal = item.frame.reshape(3, 3)[0]
        tangent = velocity - np.dot(velocity, normal) * normal
        leg = int(matches[0])
        speeds[leg] = max(speeds[leg], float(np.linalg.norm(tangent)))
    return speeds


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = args.physics_timestep
    model.actuator_biasprm[:, 2] = -args.servo_kd
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.qpos[2] += args.reset_clearance_m
    data.ctrl[:] = model.key("stand").ctrl
    mujoco.mj_forward(model, data)
    for _ in range(round(args.pre_settle_s / model.opt.timestep)):
        mujoco.mj_step(model, data)

    config = StandCompactWbcConfig(
        control_timestep_s=args.control_timestep,
        stand_settle_s=args.stand_settle,
        weight_transfer_s=args.weight_transfer,
        swing_s=args.swing,
        touchdown_timeout_s=args.touchdown_timeout,
        final_capture_timeout_s=args.capture_timeout,
        required_hold_s=args.required_hold,
        foot_lift_m=args.foot_lift,
        support_shift_fraction=args.support_shift_fraction,
        support_shift_limit_m=args.support_shift_limit,
        maximum_joint_velocity_rad_s=args.maximum_joint_velocity,
        command_lookahead_s=args.command_lookahead,
        maximum_command_step_rad=args.maximum_command_step,
    )
    controller = StandCompactWbc3D(model, config)
    controller.reset(data)
    physics_steps = round(config.control_timestep_s / model.opt.timestep)
    if not np.isclose(physics_steps * model.opt.timestep, config.control_timestep_s):
        raise ValueError("control timestep must align with physics timestep")

    elapsed = 0.0
    maximum_time = args.maximum_time
    maximum_slip = np.zeros(4)
    slip_distance = np.zeros(4)
    maximum_torque = 0.0
    maximum_qp_residual = 0.0
    phase_log = []
    last_phase = None
    output = None
    while elapsed < maximum_time:
        output = controller.step(data)
        data.ctrl[:] = output.joint_position
        maximum_qp_residual = max(maximum_qp_residual, output.qp_stance_residual_m_s)
        if output.phase != last_phase:
            phase_log.append({
                "time_s": elapsed,
                "phase": output.phase,
                "swing_legs": None if output.swing_legs is None else [
                    LEGS[leg] for leg in output.swing_legs
                ],
                "root_qpos": data.qpos[:7].tolist(),
                "root_velocity": data.qvel[:6].tolist(),
                "contacts": output.measured_contact.astype(int).tolist(),
            })
            last_phase = output.phase
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)
            slip = contact_slip_speeds(
                model, data, controller.foot_geom_ids, controller.floor_geom_id
            )
            maximum_slip = np.maximum(maximum_slip, slip)
            slip_distance += slip * model.opt.timestep
            maximum_torque = max(maximum_torque, float(np.max(np.abs(data.actuator_force))))
        elapsed += config.control_timestep_s
        if output.successful or output.failed:
            break

    joint_error = float(np.max(np.abs(
        data.qpos[controller.qpos_indices] - controller.compact_target
    )))
    report = {
        "success": bool(output and output.successful),
        "failed": bool(output and output.failed),
        "failure_reason": "" if output is None else output.failure_reason,
        "elapsed_s": elapsed,
        "control_timestep_s": config.control_timestep_s,
        "joint_order": list(JOINT_NAMES),
        "phase_log": phase_log,
        "maximum_qp_stance_residual_m_s": maximum_qp_residual,
        "maximum_contact_slip_m_s_per_foot": maximum_slip.tolist(),
        "integrated_contact_slip_m_per_foot": slip_distance.tolist(),
        "sum_integrated_contact_slip_m": float(np.sum(slip_distance)),
        "maximum_actuator_torque_nm": maximum_torque,
        "compact_hold_s": 0.0 if output is None else output.compact_hold_s,
        "terminal_joint_position_error_rad": joint_error,
        "terminal_joint_speed_rad_s": float(np.max(np.abs(data.qvel[controller.dof_indices]))),
        "terminal_root_linear_speed_m_s": float(np.max(np.abs(data.qvel[:3]))),
        "terminal_root_angular_speed_rad_s": float(np.max(np.abs(data.qvel[3:6]))),
        "terminal_root_qpos": data.qpos[:7].tolist(),
        "terminal_joint_qpos": data.qpos[controller.qpos_indices].tolist(),
        "terminal_foot_positions_m": data.site_xpos[controller.site_ids].tolist(),
        "target_foot_positions_m": controller.final_feet.tolist(),
        "terminal_foot_position_error_m": np.linalg.norm(
            data.site_xpos[controller.site_ids] - controller.final_feet, axis=1
        ).tolist(),
        "mujoco_warning_counts": [int(item.number) for item in data.warning],
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physics-timestep", type=float, default=0.001)
    parser.add_argument("--control-timestep", type=float, default=0.02)
    parser.add_argument("--servo-kd", type=float, default=0.2)
    parser.add_argument("--pre-settle-s", type=float, default=0.5)
    parser.add_argument("--reset-clearance-m", type=float, default=0.0005)
    parser.add_argument("--stand-settle", type=float, default=0.25)
    parser.add_argument("--weight-transfer", type=float, default=0.30)
    parser.add_argument("--swing", type=float, default=0.55)
    parser.add_argument("--touchdown-timeout", type=float, default=0.25)
    parser.add_argument("--capture-timeout", type=float, default=1.0)
    parser.add_argument("--required-hold", type=float, default=0.10)
    parser.add_argument("--foot-lift", type=float, default=0.025)
    parser.add_argument("--support-shift-fraction", type=float, default=0.30)
    parser.add_argument("--support-shift-limit", type=float, default=0.025)
    parser.add_argument("--maximum-joint-velocity", type=float, default=1.2)
    parser.add_argument("--command-lookahead", type=float, default=0.10)
    parser.add_argument("--maximum-command-step", type=float, default=0.035)
    parser.add_argument("--maximum-time", type=float, default=6.0)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
