"""Collect steady-state CEM cycle data for the stand-to-roll capture detector.

Runs the mature phase-locked CEM reference (the high-speed zero-contact
controller) for ``--cycles`` full oscillator periods after a short warm-up, and
saves full per-control-step state so a later matcher can bucket by CEM phase
theta.  Everything is saved raw (time-step data); bucketing happens downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from scripts import evaluate_3d_symmetric_cem_reference as bridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad_abd10.xml"
)
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "rollingquad_abd10_high_speed_zero_contact_refine_smoke"
    / "01_zero_contact_speed_refine"
    / "best_phase_controller.json"
)
DEFAULT_OUT = PROJECT_ROOT / "results" / "cem_cycle_data" / "cem_cycles.npz"

ABDUCTION_JOINT_NAMES = (
    "front_left_hip_abduction",
    "front_right_hip_abduction",
    "rear_left_hip_abduction",
    "rear_right_hip_abduction",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--physics-profile", default="cg20")
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--front-abduction-deg", type=float, default=-10.0)
    parser.add_argument("--rear-abduction-deg", type=float, default=10.0)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    bridge.activate_planar_geometry(bridge.PUPPER_ORIGINAL_SHELL_60_PARAMETERS)

    config = load_cem_reference(
        args.controller, reference_weight=1.0, minimum_residual_gain=0.0
    )
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    task = bridge.physics_profile_3d(args.physics_profile)
    bridge.apply_physics_options_3d(model, task)
    data = mujoco.MjData(model)

    actuator_ids = np.asarray(
        [model.actuator(f"{name}_servo").id for name in bridge.JOINT_NAMES_3D]
    )
    qpos_indices = np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in bridge.JOINT_NAMES_3D]
    )
    joint_low = np.asarray(
        [model.jnt_range[model.joint(name).id, 0] for name in bridge.JOINT_NAMES_3D]
    )
    joint_high = np.asarray(
        [model.jnt_range[model.joint(name).id, 1] for name in bridge.JOINT_NAMES_3D]
    )

    abduction_actuator_ids = np.asarray(
        [model.actuator(f"{name}_servo").id for name in ABDUCTION_JOINT_NAMES]
    )
    abduction_qpos_indices = np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in ABDUCTION_JOINT_NAMES]
    )
    abduction_ctrl = np.deg2rad(
        (
            args.front_abduction_deg,
            args.front_abduction_deg,
            args.rear_abduction_deg,
            args.rear_abduction_deg,
        )
    )
    abduction_ctrl = np.clip(
        abduction_ctrl,
        model.actuator_ctrlrange[abduction_actuator_ids, 0],
        model.actuator_ctrlrange[abduction_actuator_ids, 1],
    )

    # Servo gains for rolling and abduction actuators.
    for ids in (actuator_ids, abduction_actuator_ids):
        model.actuator_gainprm[ids, 0] = args.kp
        model.actuator_biasprm[ids, 1] = -args.kp
        model.actuator_biasprm[ids, 2] = -args.kd
        model.actuator_forcerange[ids, 0] = -args.torque_limit
        model.actuator_forcerange[ids, 1] = args.torque_limit

    torso_body_id = model.body("torso").id
    floor_geom_id = model.geom("floor").id
    foot_geom_ids = {model.geom(name).id for name in bridge.FOOT_GEOM_NAMES_3D}
    shell_geom_ids = bridge._shell_geom_ids(model, mujoco, foot_geom_ids)

    initial_planar = bridge.planar_cem_target(
        0.0, config, apply_foot_gap_projection=True
    )
    initial_ctrl = np.clip(
        bridge.map_planar_to_curl_3d_targets(initial_planar),
        joint_low,
        joint_high,
    )
    bridge._reset_data(model, data, mujoco, qpos_indices, actuator_ids, initial_ctrl)
    data.qpos[abduction_qpos_indices] = abduction_ctrl
    data.ctrl[abduction_actuator_ids] = abduction_ctrl
    mujoco.mj_forward(model, data)

    timestep = float(model.opt.timestep)
    control_dt = 0.02
    control_repeat = max(1, round(control_dt / timestep))
    record_duration = args.cycles * (2.0 * np.pi / config.oscillator_rate_rad_s)
    total_steps = max(1, round((args.warmup_s + record_duration) / control_dt))
    warmup_steps = round(args.warmup_s / control_dt)

    phase = 0.0
    rolling_phase = 0.0
    rows = []

    for step in range(total_steps):
        for _ in range(control_repeat):
            phase = float(
                bridge.advance_oscillator(
                    np, rolling_phase, phase, timestep, config, rate_scale=1.0
                )
            )
            planar = bridge.planar_cem_target(
                phase, config, apply_foot_gap_projection=True
            )
            ctrl = np.clip(
                bridge.map_planar_to_curl_3d_targets(planar),
                joint_low,
                joint_high,
            )
            data.ctrl[actuator_ids] = ctrl
            data.ctrl[abduction_actuator_ids] = abduction_ctrl
            mujoco.mj_step(model, data)
            rolling_phase += float(data.qvel[4]) * timestep

        if step >= warmup_steps:
            shell_contact, foot_contact, self_contact = bridge._contact_flags(
                data, floor_geom_id, shell_geom_ids, foot_geom_ids
            )
            rows.append(
                (
                    float(data.time),
                    data.qpos.copy(),
                    data.qvel.copy(),
                    data.xmat[torso_body_id].reshape(3, 3).copy(),
                    data.qvel[0:3].copy(),
                    data.qvel[3:6].copy(),
                    phase,
                    phase % (2.0 * np.pi),
                    rolling_phase,
                    data.ctrl.copy(),
                    np.asarray(
                        (shell_contact, foot_contact, self_contact),
                        dtype=np.float32,
                    ),
                )
            )

    recorded = len(rows)
    arrays = {
        "time": np.asarray([r[0] for r in rows], dtype=np.float64),
        "qpos": np.stack([r[1] for r in rows]),
        "qvel": np.stack([r[2] for r in rows]),
        "orientation": np.stack([r[3] for r in rows]),
        "linear_velocity": np.stack([r[4] for r in rows]),
        "angular_velocity": np.stack([r[5] for r in rows]),
        "cem_phase": np.asarray([r[6] for r in rows], dtype=np.float64),
        "cem_phase_wrapped": np.asarray([r[7] for r in rows], dtype=np.float64),
        "rolling_phase": np.asarray([r[8] for r in rows], dtype=np.float64),
        "joint_target": np.stack([r[9] for r in rows]),
        "contact": np.stack([r[10] for r in rows]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    metadata = {
        "xml": str(args.xml.resolve()),
        "controller": str(args.controller.resolve()),
        "physics_profile": args.physics_profile,
        "kp": args.kp,
        "kd": args.kd,
        "torque_limit": args.torque_limit,
        "front_abduction_deg": args.front_abduction_deg,
        "rear_abduction_deg": args.rear_abduction_deg,
        "oscillator_rate_rad_s": config.oscillator_rate_rad_s,
        "oscillator_period_s": 2.0 * np.pi / config.oscillator_rate_rad_s,
        "oscillator_coupling_per_s": config.oscillator_coupling_per_s,
        "cycles": args.cycles,
        "warmup_s": args.warmup_s,
        "control_dt": control_dt,
        "recorded_steps": recorded,
        "record_duration_s": record_duration,
        "total_rolling_turns": float(rolling_phase / (2.0 * np.pi)),
        "joint_names": list(bridge.JOINT_NAMES_3D),
        "abduction_joint_names": list(ABDUCTION_JOINT_NAMES),
    }
    (args.out.with_suffix(".json")).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Recorded {recorded} steps over {record_duration:.1f}s "
        f"(~{rolling_phase / (2.0 * np.pi):.1f} rolling turns)",
        flush=True,
    )
    print(f"  data: {args.out}", flush=True)
    print(f"  meta: {args.out.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
