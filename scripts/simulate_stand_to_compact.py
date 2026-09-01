"""Physical stand-to-compact servo test (no policy inference or qpos playback).

Run: python -m scripts.simulate_stand_to_compact
Only ctrl is interpolated at 50 Hz.  A free root, CAD-floor contact, gravity,
joint limits and the source MJCF's torque-limited PD servos remain active.
Training physics constants are read without importing JAX/Brax or patching XML.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/rollingquad_description_2/mjcf/rollingquad.xml"
TRAINING = ROOT / "scripts/train_ppo_walk3d.py"


def training_constants():
    wanted = {"PHYS_TIMESTEP", "SOLVER_ITER", "SOLVER_LS_ITER", "IMPRATIO",
              "CONE", "EULERDAMP", "N_FRAMES", "NOMINAL_H",
              "SELF_COLLISION", "WALK_COLLISION_PROXIES"}
    result = {}
    for node in ast.parse(TRAINING.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = ast.literal_eval(node.value)
    if result.keys() != wanted:
        raise ValueError(f"missing training constants: {wanted - result.keys()}")
    if not result["SELF_COLLISION"] or result["WALK_COLLISION_PROXIES"]:
        raise ValueError(
            "This test expects full CAD with the selective self-collision profile"
        )
    return result


def load_model(profile="training"):
    config = training_constants()
    if profile == "refined":
        config["PHYS_TIMESTEP"] *= 0.5
        config["N_FRAMES"] *= 2
        config["SOLVER_ITER"] = max(20, config["SOLVER_ITER"])
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.opt.timestep = config["PHYS_TIMESTEP"]
    model.opt.iterations = config["SOLVER_ITER"]
    model.opt.ls_iterations = config["SOLVER_LS_ITER"]
    model.opt.impratio = config["IMPRATIO"]
    model.opt.cone = getattr(mujoco.mjtCone, "mjCONE_" + config["CONE"].upper())
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    flag = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    if config["EULERDAMP"]:
        model.opt.disableflags &= ~flag
    else:
        model.opt.disableflags |= flag
    assert not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    assert not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])

    joints = model.actuator_trnid[:, 0]
    addresses = model.jnt_qposadr[joints]
    names = [model.joint(int(jid)).name for jid in joints]
    poses = {}
    for name in ("stand", "compact"):
        kid = model.key(name).id
        poses[name] = model.key_qpos[kid, addresses].copy()
        np.testing.assert_allclose(poses[name], model.key_ctrl[kid], atol=1e-9)
        if (np.any(poses[name] < model.actuator_ctrlrange[:, 0])
                or np.any(poses[name] > model.actuator_ctrlrange[:, 1])):
            raise ValueError(f"{name} target outside actuator limits")
    np.testing.assert_allclose(
        model.key_qpos[model.key("stand").id, 2], config["NOMINAL_H"], atol=1e-9)
    return model, config, names, addresses, poses


def font(size):
    for candidate in ("C:/Windows/Fonts/consola.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def contact_stats(model, data):
    floor = model.geom("floor").id
    bodies = set()
    minimum = 0.0
    peak_force = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        minimum = min(minimum, float(contact.dist))
        if floor in contact.geom:
            other = int(contact.geom[1] if contact.geom[0] == floor else contact.geom[0])
            bodies.add(model.body(int(model.geom_bodyid[other])).name)
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        peak_force = max(peak_force, float(np.linalg.norm(force[:3])))
    return minimum, peak_force, sorted(bodies)


def draw_frame(renderer, model, data, phase, blend, error, tau, elapsed, duration):
    panels = []
    for azimuth, elevation, title in ((135, -23, "OBLIQUE"), (90, -7, "SIDE")):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0, 0, 0.09]
        camera.distance = 0.80
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera)
        panel = Image.fromarray(renderer.render())
        ImageDraw.Draw(panel).text((12, 10), title, font=font(16), fill="white")
        panels.append(panel)
    canvas = Image.new("RGB", (960, 442), (18, 24, 33))
    canvas.paste(panels[0], (0, 58))
    canvas.paste(panels[1], (480, 58))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), "PHYSICS ON | STAND -> COMPACT | actual simulated motion",
              font=font(21), fill=(235, 242, 247))
    tilt = np.degrees(np.arccos(np.clip(data.xmat[1, 8], -1, 1)))
    draw.text((14, 34), f"t={elapsed:5.2f}s  {phase:<16} blend={blend:5.1%}  transition={duration:g}s  tilt={tilt:.1f}deg",
              font=font(16), fill=(110, 210, 233))
    draw.text((14, 386), f"root z={data.qpos[2]:.4f}m   |tau|max={tau:.3f}/3Nm   max target error={error:.4f}rad",
              font=font(17), fill="white")
    draw.text((14, 414), "Full CAD / selective self-collision | gravity ON | no root or joint pose forcing",
              font=font(14), fill=(174, 187, 201))
    return canvas


def simulate(duration, args, render):
    model, config, names, addresses, poses = load_model(args.physics_profile)
    data = mujoco.MjData(model)
    # The only state assignment is this initial reset, matching DeployEnv.
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.qpos[2] += 0.0005
    mujoco.mj_forward(model, data)
    initial_xy = data.qpos[:2].copy()
    dt = model.opt.timestep
    control_steps = config["N_FRAMES"]
    frame_steps = max(1, round(1.0 / (args.fps * dt)))
    total = args.stand_seconds + duration + args.hold_seconds
    count = round(total / dt)
    records, states, velocities, controls, torques, frames = [], [], [], [], [], []
    snapshots = {}
    floor_bodies = set()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 480)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 320)
    renderer = mujoco.Renderer(model, height=320, width=480) if render else None
    if args.viewer and render:
        from mujoco import viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(model, data)
    else:
        viewer_context = contextlib.nullcontext(None)
    blend = 0.0
    wall_start = time.monotonic()
    with viewer_context as viewer:
        for index in range(count + 1):
            elapsed = float(data.time)
            if index % control_steps == 0:
                blend = float(np.clip((elapsed - args.stand_seconds) / duration, 0, 1))
                data.ctrl[:] = poses["stand"] + blend * (poses["compact"] - poses["stand"])
            # Refresh derived rendering/contact quantities without altering state.
            mujoco.mj_forward(model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                raise RuntimeError(f"non-finite simulated state at {elapsed}s")
            angle_error = np.abs(data.ctrl - data.qpos[addresses])
            tau = float(np.max(np.abs(data.actuator_force)))
            tilt = float(np.degrees(np.arccos(np.clip(data.xmat[1, 8], -1, 1))))
            minimum, force, bodies = contact_stats(model, data)
            if elapsed >= args.stand_seconds:
                floor_bodies.update(bodies)
            sat = np.abs(data.actuator_force) >= 0.99 * model.actuator_forcerange[:, 1]
            records.append([elapsed, blend, data.qpos[2], tilt,
                            np.linalg.norm(data.qpos[:2] - initial_xy),
                            data.qvel[2], np.max(angle_error), tau,
                            data.ncon, minimum, force, np.mean(sat)])
            states.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
            controls.append(data.ctrl.copy())
            torques.append(data.actuator_force.copy())
            if renderer is not None and index % frame_steps == 0:
                phase = ("STAND / settle" if elapsed < args.stand_seconds else
                         "INTERPOLATING" if blend < 1.0 else "COMPACT / hold")
                frame = draw_frame(renderer, model, data, phase, blend,
                                   float(np.max(angle_error)), tau, elapsed, duration)
                frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
                if blend == 0:
                    snapshots["stand"] = frame
                if 0.49 <= blend <= 0.51:
                    snapshots["midpoint"] = frame
                if blend >= 1:
                    snapshots["compact"] = frame
            if viewer is not None:
                if not viewer.is_running():
                    raise RuntimeError("viewer closed before test completed")
                viewer.sync()
                time.sleep(max(0.0, wall_start + elapsed - time.monotonic()))
            if index < count:
                mujoco.mj_step(model, data)
    if renderer is not None:
        renderer.close()
    rows = np.asarray(records)
    qpos = np.asarray(states)
    qvel = np.asarray(velocities)
    ctrl = np.asarray(controls)
    tau_log = np.asarray(torques)
    transition = (rows[:, 0] >= args.stand_seconds) & (rows[:, 0] <= args.stand_seconds + duration)
    final_window = rows[:, 0] >= total - min(1.0, args.hold_seconds)
    report = {
        "transition_seconds": duration,
        "interpolation": "linear joint POSITION TARGET, updated at 50Hz; no low-pass",
        "total_seconds": total,
        "physics_engine": f"CPU MuJoCo {mujoco.__version__} (not MJX)",
        "physics_profile": args.physics_profile,
        "physics_constants": config,
        "control_dt_s": float(dt * control_steps),
        "qpos_overwrite_after_reset": False,
        "policy_inference": False,
        "randomization": False,
        "contact_scope": "full CAD / floor with selective robot self-collision, as in training",
        "joint_order": names,
        "stand_target_rad": poses["stand"].tolist(),
        "compact_target_rad": poses["compact"].tolist(),
        "start_root_height_m": float(qpos[0, 2]),
        "settled_stand_height_m": float(rows[np.argmin(np.abs(rows[:, 0] - args.stand_seconds)), 2]),
        "final_root_height_m": float(data.qpos[2]),
        "root_height_min_max_m": [float(rows[:, 2].min()), float(rows[:, 2].max())],
        "peak_tilt_deg": float(rows[:, 3].max()),
        "final_tilt_deg": float(rows[-1, 3]),
        "last_second_tilt_min_max_deg": [float(rows[final_window, 3].min()),
                                         float(rows[final_window, 3].max())],
        "last_second_peak_joint_speed_rad_s": float(
            np.abs(qvel[final_window, 6:]).max()),
        "final_xy_drift_m": float(rows[-1, 4]),
        "peak_abs_vertical_speed_m_s": float(np.abs(rows[:, 5]).max()),
        "peak_abs_torque_nm": float(rows[:, 7].max()),
        "motor_time_saturation_fraction": float(rows[:, 11].mean()),
        "peak_contact_force_per_contact_n": float(rows[:, 10].max()),
        "max_contact_penetration_m": float(-min(0, rows[:, 9].min())),
        "transition_max_target_error_rad": float(rows[transition, 6].max()),
        "final_max_compact_error_rad": float(np.abs(data.qpos[addresses] - poses["compact"]).max()),
        "final_mean_compact_error_rad": float(np.abs(data.qpos[addresses] - poses["compact"]).mean()),
        "last_second_max_target_error_rad": float(rows[final_window, 6].max()),
        "final_joint_position_rad": data.qpos[addresses].tolist(),
        "final_contact_bodies": contact_stats(model, data)[2],
        "transition_and_hold_contact_bodies": sorted(floor_bodies),
        "mujoco_warning_counts": [int(w.number) for w in data.warning],
        "finite_states": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
    }
    stem = f"transition_{duration:g}s"
    np.savez_compressed(args.output_dir / f"{stem}.npz", time=rows[:, 0],
                        qpos=qpos, qvel=qvel, ctrl=ctrl, actuator_force=tau_log,
                        joint_names=np.asarray(names), physics_metrics=rows)
    with (args.output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "blend", "root_z_m", "tilt_deg", "xy_drift_m",
                         "vz_m_s", "max_target_error_rad", "max_torque_nm", "ncon",
                         "min_contact_distance_m", "peak_contact_force_n", "saturation_fraction"])
        writer.writerows(rows[::control_steps])
    if frames:
        gif = args.output_dir / f"{stem}_physics.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=round(1000 / args.fps), loop=0, optimize=False)
        report["animation"] = str(gif)
        for label, frame in snapshots.items():
            frame.save(args.output_dir / f"{stem}_{label}.png")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="+", type=float, default=[1, 3, 5])
    parser.add_argument("--render-duration", type=float, default=3)
    parser.add_argument("--stand-seconds", type=float, default=2)
    parser.add_argument("--hold-seconds", type=float, default=3)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--viewer", action="store_true", help="show a live physics viewer for the rendered run")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--physics-profile", choices=("training", "refined"), default="training",
                        help="refined halves physics dt and raises solver iterations, keeping 50Hz ctrl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" /
                        "stand_to_compact_physics" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if any(x <= 0 for x in args.durations) or args.hold_seconds <= 0 or args.stand_seconds < 0:
        parser.error("durations/hold must be positive; stand must be non-negative")
    if args.fps <= 0 or args.fps > 50:
        parser.error("fps must be 1..50")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = [simulate(t, args, not args.no_render and t == args.render_duration)
               for t in args.durations]
    metadata = {"source_model": str(MODEL),
                "source_model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                "runs": reports}
    (args.output_dir / "report.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"RESULTS: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
