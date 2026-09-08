"""CEM roll -> measured-angle trigger -> fast interpolation -> Stand hold.

CPU MuJoCo only. Uses the explicitly requested abd10 no-self-collision XML.
No state replacement, braking policy, or learned policy after the trigger.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS
from curl_robot_2d_mjx.cem_reference import advance_oscillator, load_cem_reference
from scripts.evaluate_3d_symmetric_cem_reference import activate_planar_geometry
from scripts.view_3d_cem_reference import _target_for_phase

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/rollingquad_description_2/mjcf/rollingquad_abd10_no_self_collision.xml"
REFERENCE = ROOT / "results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json"


def near_target(pitch, angular_speed, target, lead, *, min_speed=0.5):
    """One-sided window BEFORE target, following the measured rotation direction."""
    distance = math.atan2(math.sin(target - pitch), math.cos(target - pitch))
    return abs(angular_speed) >= min_speed and 0 <= distance * np.sign(angular_speed) <= lead


def interpolate(start, target, elapsed, duration):
    alpha = float(np.clip(elapsed / duration, 0.0, 1.0))
    return (1.0 - alpha) * start + alpha * target


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=MODEL)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--trigger-pitch-deg", type=float, default=0.0,
                   help="nose-up positive; Stand torso orientation=0")
    p.add_argument("--lead-deg", type=float, default=15.0)
    p.add_argument("--deploy-duration", type=float, default=0.15)
    p.add_argument("--min-roll-duration", type=float, default=2.0)
    p.add_argument("--min-roll-turns", type=float, default=1.0)
    p.add_argument("--max-roll-duration", type=float, default=10.0)
    p.add_argument("--stand-duration", type=float, default=3.0)
    p.add_argument("--stand-keyframe", default="stand")
    p.add_argument("--out", type=Path, default=ROOT / "results/handcrafted_roll_to_stand_abd10")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", action="store_true")
    args = p.parse_args(argv)
    for name in ("deploy_duration", "max_roll_duration", "stand_duration"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be finite and positive")
    if not math.isfinite(args.trigger_pitch_deg) or not 0 < args.lead_deg < 90:
        p.error("trigger must be finite and lead must be in (0,90) degrees")
    if not (0 <= args.min_roll_duration < args.max_roll_duration) or not math.isfinite(args.min_roll_turns) or args.min_roll_turns < 0:
        p.error("invalid rolling warmup")
    return args


def run(args):
    activate_planar_geometry(PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    # Preserve the supplied XML's solver, contact masks and motor parameters.
    reference = load_cem_reference(args.reference)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
    joint_ids = np.array([model.joint(name).id for name in JOINT_NAMES_3D])
    qids = model.jnt_qposadr[joint_ids]
    aids = np.array([model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D])
    low, high = model.actuator_ctrlrange[aids].T
    data.ctrl[aids] = _target_for_phase(0., reference, 1., 0., .25, 0., .25, 0., low, high)
    data.qpos[qids] = data.ctrl[aids]  # initial reset only
    stand = model.key_ctrl[model.key(args.stand_keyframe).id].copy()
    torso = model.body("torso").id
    floor = model.geom("floor").id
    feet = {model.geom(f"{leg}_foot_proxy").id for leg in
            ("front_left", "front_right", "rear_left", "rear_right")}
    mujoco.mj_forward(model, data)
    phase = rolled = 0.0
    previous_in_window = False
    trigger = None
    start_ctrl = None
    dt = float(model.opt.timestep)
    rows, qpos_rows, ctrl_rows = [], [], []
    max_torque = 0.0
    longest_stand = current_stand = 0.0
    self_contacts = 0
    writer = renderer = None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(args.out / "roll_to_stand.mp4"), fps=30, codec="libx264")
        renderer = mujoco.Renderer(model, height=600, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = torso
    camera.distance, camera.azimuth, camera.elevation = 1.0, 110., -15.
    next_frame = 0.0
    context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer
        context = viewer.launch_passive(model, data)
    try:
        with context as window:
            if window is not None:
                window.cam.distance = 1.0
                window.cam.trackbodyid = torso
                window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            while True:
                tick = time.perf_counter()
                rotation = data.xmat[torso].reshape(3, 3)
                # Nose-up positive: +body-y angular velocity lowers the nose.
                pitch = math.atan2(rotation[2, 0], rotation[2, 2])
                if trigger is None:
                    if data.time >= args.max_roll_duration:
                        break
                    in_window = near_target(pitch, -float(data.qvel[4]), math.radians(args.trigger_pitch_deg), math.radians(args.lead_deg))
                    # Trigger on entering the early window, never midway
                    # through it merely because the warmup just expired.
                    if (data.time >= args.min_roll_duration and abs(rolled) >= args.min_roll_turns * 2 * math.pi
                            and in_window and not previous_in_window):
                        trigger = dict(time_s=float(data.time), pitch_deg=math.degrees(pitch),
                                       angular_speed_rad_s=float(data.qvel[4]), pitch_rate_rad_s=-float(data.qvel[4]), turns=rolled / (2 * math.pi),
                                       qpos=data.qpos.tolist(), qvel=data.qvel.tolist(), ctrl=data.ctrl.tolist())
                        start_ctrl = data.ctrl.copy()
                        print("[trigger] " + json.dumps(trigger), flush=True)
                    previous_in_window = in_window
                if trigger is None:
                    phase = float(advance_oscillator(np, rolled, phase, dt, reference))
                    data.ctrl[aids] = _target_for_phase(phase, reference, 1., 0., .25, 0., .25, data.time, low, high)
                    mode = 0
                else:
                    elapsed = float(data.time) - trigger["time_s"]
                    if elapsed >= args.deploy_duration + args.stand_duration:
                        break
                    data.ctrl[:] = interpolate(start_ctrl, stand, elapsed, args.deploy_duration)
                    mode = 1 if elapsed < args.deploy_duration else 2
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("nonfinite simulation")
                rolled += float(data.qvel[4]) * dt
                max_torque = max(max_torque, float(np.abs(data.actuator_force).max()))
                supported = set()
                nonfoot = 0
                for contact in data.contact[:data.ncon]:
                    if contact.dist > 0:
                        continue
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    if floor in (g1, g2):
                        geom = g2 if g1 == floor else g1
                        if geom in feet:
                            supported.add(geom)
                        else:
                            nonfoot += 1
                    else:
                        self_contacts += 1
                tilt = math.acos(float(np.clip(data.xmat[torso].reshape(3, 3)[2, 2], -1., 1.)))
                stable = mode == 2 and len(supported) >= 3 and nonfoot == 0 and tilt < .22 and np.linalg.norm(data.qvel[:3]) < .12 and np.linalg.norm(data.qvel[3:6]) < .45
                current_stand = current_stand + dt if stable else 0.
                longest_stand = max(longest_stand, current_stand)
                if len(rows) == 0 or data.time - rows[-1][0] >= .0099:
                    rows.append([data.time, mode, math.degrees(pitch), data.qvel[4], data.qpos[2], len(supported), nonfoot])
                    qpos_rows.append(data.qpos.copy())
                    ctrl_rows.append(data.ctrl.copy())
                if renderer is not None and data.time >= next_frame:
                    renderer.update_scene(data, camera=camera)
                    frame = renderer.render()
                    writer.append_data(frame)
                    if trigger is not None and not (args.out / "trigger.png").exists():
                        import imageio.v2 as imageio
                        imageio.imwrite(args.out / "trigger.png", frame)
                    next_frame += 1 / 30
                if window is not None:
                    if not window.is_running():
                        break
                    window.sync()
                    time.sleep(max(0., dt - (time.perf_counter() - tick)))
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    summary = dict(model=str(args.xml.resolve()), reference=str(args.reference.resolve()),
                   trigger=trigger, pitch_convention="nose_up_positive", trigger_target_deg=args.trigger_pitch_deg, lead_deg=args.lead_deg,
                   deploy_duration_s=args.deploy_duration, stand_keyframe=args.stand_keyframe,
                   longest_stable_stand_s=longest_stand, stable_stand_success=longest_stand >= 1.,
                   final_height_m=float(data.qpos[2]), final_tilt_deg=math.degrees(tilt),
                   peak_torque_nm=max_torque, self_contact_samples=self_contacts,
                   state_overwritten_after_reset=False, interpolation="linear position targets",
                   status="completed" if trigger is not None else "no_trigger", elapsed_s=float(data.time))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savetxt(args.out / "rollout.csv", np.asarray(rows), delimiter=",",
               header="time_s,mode,pitch_deg,omega_y,root_z,foot_contacts,nonfoot_contacts", comments="")
    np.savez_compressed(args.out / "rollout.npz", qpos=qpos_rows, ctrl=ctrl_rows, times=np.asarray(rows)[:, 0])
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
