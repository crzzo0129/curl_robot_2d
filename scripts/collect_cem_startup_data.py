"""Record original CEM oscillator startup at policy rate, including frame zero.

Separate from the steady-state matcher bank. CPU MuJoCo collection, run on
the cloud. The oscillator/target functions are the original controller; its
targets are held for 20 ms to match the student's action interface.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/cem_startup_data"))
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-seconds", type=float, default=2.0)
    parser.add_argument("--controller", type=Path)
    args = parser.parse_args()
    if args.episodes < 5 or not 0 < args.startup_seconds < 10:
        parser.error("episodes >=5 and startup-seconds between 0 and 10 required")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output must be empty")
    import mujoco
    from scripts.collect_cem_cycle_data import DEFAULT_CONTROLLER
    from scripts import evaluate_3d_symmetric_cem_reference as bridge
    from curl_robot_2d_mjx.cem_reference import load_cem_reference
    from curl_robot_2d_mjx.config_stand_to_roll import StandToRollConfig
    from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
    from curl_robot_2d_mjx.environment_3d import (
        model_path_3d, apply_physics_options_3d, disable_rollingquad_self_collision_3d,
        geometry_parameters_3d,
    )
    from curl_robot_2d_mjx.deployment_rolling_3d import (
        CONTROLLER_JOINT_NAMES_3D, initial_rolling_deploy_history_3d,
        push_rolling_deploy_frame_3d, rolling_deploy_frame_3d,
    )
    from curl_robot_2d_mjx.stand_to_roll_training import action_center_and_scale
    from curl_robot_2d_mjx.reset_grounding import make_floor_clearance

    task = StandToRollConfig(reset_velocity_noise_rad_s=0.0)
    model = mujoco.MjModel.from_xml_path(str(model_path_3d(task.geometry)))
    apply_physics_options_3d(model, physics_profile_3d(task.physics_profile,
                            Rolling3DConfig(geometry=task.geometry)))
    disable_rollingquad_self_collision_3d(model)
    bridge.activate_planar_geometry(bridge.PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
    controller = load_cem_reference(args.controller or DEFAULT_CONTROLLER)
    names = CONTROLLER_JOINT_NAMES_3D
    qi = np.asarray([model.jnt_qposadr[model.joint(n).id] for n in names])
    ai = np.asarray([model.actuator(n + "_servo").id for n in names])
    ji = np.asarray([model.joint(n).id for n in names])
    lo, hi = model.jnt_range[ji].T
    center, scale = action_center_and_scale(task)
    planar_ai = np.asarray([model.actuator(n + "_servo").id for n in bridge.JOINT_NAMES_3D])
    abd_ai = ai[[0, 3, 6, 9]]
    compact = model.key_qpos[model.key("compact").id].copy()
    stand = model.key_qpos[model.key("stand").id].copy()
    stand[qi[[0, 3, 6, 9]]] = 0.0
    torso = model.body("torso").id
    floor = model.geom("floor").id
    floor_clearance = make_floor_clearance(model, floor, np)
    radius = geometry_parameters_3d(task.geometry).shell_contact_radius
    repeat = round(task.control_timestep / model.opt.timestep)
    rng = np.random.default_rng(args.seed)
    samples, reports = [], []
    for episode in range(args.episodes):
        data = mujoco.MjData(model)
        alpha = rng.uniform(0.0, 0.1)
        data.qpos[:] = (1 - alpha) * compact + alpha * stand
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        data.qpos[qi] = np.clip(data.qpos[qi] + rng.uniform(-0.01, 0.01, 12), lo, hi)
        data.qvel[:] = 0.0
        data.ctrl[ai] = data.qpos[qi]
        mujoco.mj_forward(model, data)
        z_correction = task.reset_ground_clearance_m - float(floor_clearance(data))
        data.qpos[2] += z_correction
        mujoco.mj_forward(model, data)
        history = initial_rolling_deploy_history_3d(np)
        last_action = np.zeros(12)
        phase, rolling = 0.0, 0.0
        x0 = data.qpos[0]
        rows, failed = [], False
        for step in range(task.episode_length):
            rotation = data.xmat[torso].reshape(3, 3)
            frame = rolling_deploy_frame_3d(
                np, angular_velocity_body=rotation.T @ data.cvel[torso, :3],
                projected_gravity=rotation.T @ np.asarray([0., 0., -1.]),
                joint_position_offset=data.qpos[qi] - center, last_action=last_action)
            history = push_rolling_deploy_frame_3d(np, history, frame)
            phase = float(bridge.advance_oscillator(np, rolling, phase,
                          task.control_timestep, controller, rate_scale=1.0))
            planar = bridge.planar_cem_target(phase, controller, apply_foot_gap_projection=True)
            command = data.ctrl.copy()
            command[planar_ai] = bridge.map_planar_to_curl_3d_targets(planar)
            command[abd_ai] = np.deg2rad([-10., -10., 10., 10.])
            action = np.clip((np.clip(command[ai], lo, hi) - center) / scale, -1, 1)
            # Save PRE-action observation and the exact held action, no shift.
            rows.append((history.copy(), action.copy(), episode,
                         step * task.control_timestep < args.startup_seconds))
            data.ctrl[ai] = np.clip(center + scale * action, lo, hi)
            for _ in range(repeat):
                mujoco.mj_step(model, data)
                rolling += float(data.qvel[4]) * model.opt.timestep
                tilt = np.arccos(np.clip(abs(data.xmat[torso].reshape(3, 3)[1, 1]), 0, 1))
                collision = any(c.dist <= 0 and c.geom1 != floor and c.geom2 != floor
                                for c in data.contact)
                failed = (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                          or data.qpos[2] > task.terminate_root_z_max_m
                          or abs(data.qpos[1]) > task.terminate_lateral_m
                          or tilt > task.terminate_axis_tilt_rad or collision)
                if failed:
                    break
            last_action = action
            if failed:
                break
        progress = min(rolling, (data.qpos[0] - x0) / radius)
        success = not failed and progress >= 2 * np.pi
        reports.append({"episode": episode, "alpha": alpha, "success": bool(success),
                        "reset_z_correction_m": z_correction,
                        "steps": len(rows), "roll_progress": float(progress)})
        if success:
            samples.extend(rows)
        print(f"[CEM startup] episode={episode} success={success} progress={progress:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rate = float(np.mean([r["success"] for r in reports]))
    report = {"success_rate": rate, "task": asdict(task), "episodes": reports,
              "controller": controller.source, "startup_seconds": args.startup_seconds,
              "control_dt": task.control_timestep, "initial_velocity": "zero",
              "status": "passed" if rate >= 0.8 else "teacher_gate_failed"}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if rate < 0.8:
        print("Teacher startup gate failed; no BC dataset exported. Inspect controller initialization/50Hz behavior.")
        return
    np.savez_compressed(args.out / "startup_bc.npz", schema_version=np.asarray(1),
        observations=np.asarray([r[0] for r in samples], dtype=np.float32),
        actions=np.asarray([r[1] for r in samples], dtype=np.float32),
        episode_id=np.asarray([r[2] for r in samples]),
        startup=np.asarray([r[3] for r in samples]),
        action_center=center, action_scale=scale)
    print(f"Saved {args.out / 'startup_bc.npz'}; teacher success={rate:.3f}")


if __name__ == "__main__":
    main()
