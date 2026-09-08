"""Hand-crafted stand -> compact transition with a paired-foot choreography.

Choreography (foot-space, in the BODY frame; forward=+X, left=+Y):

  1. FRONT pair (FL, FR) lift TOGETHER, shift backward + inward.
  2. REAR pair (RL, RR) lift TOGETHER, shift forward + outward.
  3. Final tuck: every foot folds to the compact pose, but the FRONT pair
     finishes LAST.

Foot positions are commanded in the body frame and converted to world with the
current simulated root pose; a damped-Newton per-leg IK turns them into the 12
joint targets.  Only ctrl is written at 50 Hz -- the floating base, gravity,
contacts and the torque-limited PD servos stay live, exactly like
simulate_stand_to_compact.py, so the report reflects real physics, not playback.

Run:
    python -m scripts.manual_stand_to_compact
    python -m scripts.manual_stand_to_compact --no-render
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets" / "rollingquad_description_2" / "mjcf" / "rollingquad_abd10.xml"

LEGS = ("front_left", "front_right", "rear_left", "rear_right")
FRONT = (0, 1)
REAR = (2, 3)
JOINT_SUFFIXES = ("hip_abduction", "hip", "knee")
JOINT_NAMES = tuple(f"{leg}_{suffix}" for leg in LEGS for suffix in JOINT_SUFFIXES)


def smootherstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def swing_bump(u: float) -> float:
    """Smooth 0 -> peak -> 0 lift profile over u in [0, 1]."""
    u = float(np.clip(u, 0.0, 1.0))
    return 64.0 * u**3 * (1.0 - u) ** 3


def quat2mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_model(path: Path) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(path))
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise ValueError(f"expected 19/18/12 rollingquad model, got {model.nq}/{model.nv}/{model.nu}")
    model.opt.timestep = 0.002
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    model.opt.impratio = 10
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    return model


def foot_geometry(model: mujoco.MjModel) -> dict:
    """Body-frame foot positions at stand and compact, plus joint indices."""
    joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES], dtype=int)
    qpos_idx = np.asarray(model.jnt_qposadr[joint_ids], dtype=int)
    dof_idx = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
    actuator_ids = np.asarray([model.actuator(f"{name}_servo").id for name in JOINT_NAMES],
                              dtype=int)
    site_ids = np.asarray([model.site(f"{leg}_foot_site").id for leg in LEGS], dtype=int)

    def key_feet(name: str):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key(name).id)
        mujoco.mj_forward(model, data)
        return data.site_xpos[site_ids].copy(), data.qpos[:3].copy()

    stand_feet_world, stand_root = key_feet("stand")
    compact_feet_world, compact_root = key_feet("compact")
    # compact keyframe quaternion is identity, so its body frame == world axes.
    stand_feet_body = stand_feet_world - stand_root
    compact_feet_body = compact_feet_world - compact_root
    return {
        "joint_ids": joint_ids, "qpos_idx": qpos_idx, "dof_idx": dof_idx,
        "actuator_ids": actuator_ids, "site_ids": site_ids,
        "stand_feet_body": stand_feet_body, "compact_feet_body": compact_feet_body,
        "stand_key": model.key("stand").id, "compact_key": model.key("compact").id,
        "stand_ctrl": model.key("stand").ctrl[actuator_ids].copy(),
        "compact_ctrl": model.key("compact").ctrl[actuator_ids].copy(),
        "ctrl_low": model.actuator_ctrlrange[actuator_ids, 0].copy(),
        "ctrl_high": model.actuator_ctrlrange[actuator_ids, 1].copy(),
    }


def leg_ik(model, scratch, root_pose, foot_world, qpos_idx_leg, dof_idx_leg,
           site_id, seed, ctrl_low, ctrl_high, iterations=12):
    """Damped-Newton IK for one independent 3-DoF leg."""
    scratch.qpos[:7] = root_pose
    scratch.qpos[qpos_idx_leg] = seed
    for _ in range(iterations):
        mujoco.mj_forward(model, scratch)
        error = foot_world - scratch.site_xpos[site_id]
        if float(np.linalg.norm(error)) < 2.0e-6:
            break
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jac, None, int(site_id))
        block = jac[:, dof_idx_leg]
        delta = block.T @ np.linalg.solve(block @ block.T + 2.0e-6 * np.eye(3), error)
        scratch.qpos[qpos_idx_leg] += np.clip(delta, -0.12, 0.12)
    return np.clip(scratch.qpos[qpos_idx_leg], ctrl_low, ctrl_high)


def build_choreography(args, geo):
    stand = geo["stand_feet_body"]
    compact = geo["compact_feet_body"]
    front_shift = stand.copy()
    rear_shift = stand.copy()
    for leg in FRONT:
        inward_sign = -np.sign(stand[leg, 1]) or 1.0
        front_shift[leg] += np.asarray([-args.back_shift, inward_sign * args.inward_shift, 0.0])
    for leg in REAR:
        outward_sign = np.sign(stand[leg, 1]) or 1.0
        rear_shift[leg] += np.asarray([args.forward_shift, outward_sign * args.outward_shift, 0.0])
    return {"stand": stand, "compact": compact,
            "front_shift": front_shift, "rear_shift": rear_shift,
            "settle_s": args.settle_s, "front_s": args.front_s,
            "rear_s": args.rear_s, "tuck_s": args.tuck_s, "hold_s": args.hold_s,
            "lift_m": args.lift}


def foot_body_targets(chor, elapsed):
    """Return the (4,3) body-frame foot targets at time ``elapsed``."""
    feet = chor["stand"].copy()
    settle, front, rear, tuck = chor["settle_s"], chor["front_s"], chor["rear_s"], chor["tuck_s"]
    t = elapsed - settle

    def lerp(a, b, u):
        return a + smootherstep(u) * (b - a)

    if t < 0:
        return feet
    if t < front:
        u = t / front
        for leg in FRONT:
            feet[leg] = lerp(chor["stand"][leg], chor["front_shift"][leg], u)
            feet[leg, 2] += chor["lift_m"] * swing_bump(u)
        return feet
    t -= front
    if t < rear:
        u = t / rear
        for leg in FRONT:
            feet[leg] = chor["front_shift"][leg]
        for leg in REAR:
            feet[leg] = lerp(chor["stand"][leg], chor["rear_shift"][leg], u)
            feet[leg, 2] += chor["lift_m"] * swing_bump(u)
        return feet
    t -= rear
    if t < tuck:
        u = t / tuck
        # rear feet finish early, front feet finish LAST (u -> 1.0).
        u_rear = np.clip(u / 0.55, 0.0, 1.0)
        for leg in REAR:
            feet[leg] = lerp(chor["rear_shift"][leg], chor["compact"][leg], u_rear)
        for leg in FRONT:
            feet[leg] = lerp(chor["front_shift"][leg], chor["compact"][leg], u)
        return feet
    return chor["compact"].copy()


def contact_stats(model, data):
    floor = model.geom("floor").id
    minimum, peak, bodies = 0.0, 0.0, set()
    for index in range(data.ncon):
        item = data.contact[index]
        minimum = min(minimum, float(item.dist))
        if floor in item.geom:
            other = int(item.geom[1] if item.geom[0] == floor else item.geom[0])
            bodies.add(model.body(int(model.geom_bodyid[other])).name)
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        peak = max(peak, float(np.linalg.norm(force[:3])))
    return minimum, peak, sorted(bodies)


def font(size):
    for candidate in ("C:/Windows/Fonts/consola.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_frame(renderer, model, data, phase, elapsed, foot_heights):
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
    draw.text((14, 8), "PHYSICS ON | MANUAL STAND->COMPACT | paired-foot choreography",
              font=font(21), fill=(235, 242, 247))
    tilt = np.degrees(np.arccos(np.clip(data.xmat[1, 8], -1, 1)))
    fh = " ".join(f"{h*1000:4.0f}" for h in foot_heights)
    draw.text((14, 34), f"t={elapsed:5.2f}s  {phase:<16} tilt={tilt:.1f}deg", font=font(16), fill=(110, 210, 233))
    draw.text((14, 386), f"root z={data.qpos[2]:.4f}m   foot heights(mm)={fh}",
              font=font(17), fill="white")
    draw.text((14, 414), "full CAD / selective self-collision | gravity ON | free base",
              font=font(14), fill=(174, 187, 201))
    return canvas


def simulate(args, geo, chor, render):
    model = load_model(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, geo["stand_key"])
    data.qpos[2] += 0.0005
    mujoco.mj_forward(model, data)
    scratch = mujoco.MjData(model)
    initial_xy = data.qpos[:2].copy()
    dt = model.opt.timestep
    control_steps = 10  # 50 Hz at 0.002 s
    frame_steps = max(1, round(1.0 / (args.fps * dt)))
    total = (chor["settle_s"] + chor["front_s"] + chor["rear_s"]
             + chor["tuck_s"] + chor["hold_s"])
    count = round(total / dt)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 480)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 320)
    renderer = mujoco.Renderer(model, height=320, width=480) if render else None

    records, states, velocities, controls, cmd_feet, act_feet, frames = [], [], [], [], [], [], []
    floor_bodies = set()
    for index in range(count + 1):
        elapsed = float(data.time)
        if index % control_steps == 0:
            feet_body = foot_body_targets(chor, elapsed)
            R = quat2mat(data.qpos[3:7])
            command = geo["stand_ctrl"].copy()
            scratch.qpos[:] = data.qpos
            for leg in range(4):
                foot_world = data.qpos[:3] + R @ feet_body[leg]
                lo = geo["ctrl_low"][3 * leg:3 * leg + 3]
                hi = geo["ctrl_high"][3 * leg:3 * leg + 3]
                seed = data.qpos[geo["qpos_idx"][3 * leg:3 * leg + 3]]
                solved = leg_ik(model, scratch, data.qpos[:7], foot_world,
                                geo["qpos_idx"][3 * leg:3 * leg + 3],
                                geo["dof_idx"][3 * leg:3 * leg + 3],
                                geo["site_ids"][leg], seed, lo, hi)
                command[3 * leg:3 * leg + 3] = solved
                scratch.qpos[geo["qpos_idx"][3 * leg:3 * leg + 3]] = solved
            data.ctrl[:] = command
        mujoco.mj_forward(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError(f"non-finite simulated state at {elapsed}s")
        foot_heights = data.site_xpos[geo["site_ids"]][:, 2]
        minimum, peak, bodies = contact_stats(model, data)
        floor_bodies.update(bodies)
        tilt = float(np.degrees(np.arccos(np.clip(data.xmat[1, 8], -1, 1))))
        records.append([elapsed, data.qpos[2], tilt, np.linalg.norm(data.qpos[:2] - initial_xy),
                        data.qvel[2], float(np.max(np.abs(data.actuator_force))),
                        minimum, peak, data.ncon])
        states.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        controls.append(data.ctrl.copy())
        cmd_feet.append(foot_body_targets(chor, elapsed).copy())
        act_feet.append(data.site_xpos[geo["site_ids"]].copy())
        if renderer is not None and index % frame_steps == 0:
            t = elapsed - chor["settle_s"]
            phase = ("SETTLE" if t < 0 else "FRONT shift" if t < chor["front_s"]
                     else "REAR shift" if t < chor["front_s"] + chor["rear_s"]
                     else "TUCK (front last)" if t < chor["front_s"] + chor["rear_s"] + chor["tuck_s"]
                     else "HOLD compact")
            frames.append(draw_frame(renderer, model, data, phase, elapsed, foot_heights))
        if index < count:
            mujoco.mj_step(model, data)
    if renderer is not None:
        renderer.close()

    rows = np.asarray(records)
    qpos = np.asarray(states)
    report = {
        "model": str(args.model),
        "choreography": {
            "front": "lift together, backward + inward",
            "rear": "lift together, forward + outward",
            "final_tuck": "front pair finishes LAST",
            "back_shift_m": args.back_shift, "forward_shift_m": args.forward_shift,
            "inward_shift_m": args.inward_shift, "outward_shift_m": args.outward_shift,
            "lift_m": args.lift,
            "settle_s": chor["settle_s"], "front_s": chor["front_s"],
            "rear_s": chor["rear_s"], "tuck_s": chor["tuck_s"], "hold_s": chor["hold_s"],
        },
        "total_seconds": total,
        "control_dt_s": float(dt * control_steps),
        "peak_tilt_deg": float(rows[:, 2].max()),
        "final_tilt_deg": float(rows[-1, 2]),
        "root_z_min_max_m": [float(rows[:, 1].min()), float(rows[:, 1].max())],
        "final_root_height_m": float(data.qpos[2]),
        "peak_abs_torque_nm": float(rows[:, 5].max()),
        "final_xy_drift_m": float(rows[-1, 3]),
        "final_max_compact_error_rad": float(np.abs(data.qpos[geo["qpos_idx"]] - geo["compact_ctrl"]).max()),
        "final_contact_bodies": contact_stats(model, data)[2],
        "contact_bodies": sorted(floor_bodies),
        "finite_states": bool(np.isfinite(qpos).all() and np.isfinite(np.asarray(velocities)).all()),
    }
    stem = "manual_transition"
    np.savez_compressed(args.out_dir / f"{stem}.npz", time=rows[:, 0], qpos=qpos,
                        qvel=np.asarray(velocities), ctrl=np.asarray(controls),
                        commanded_foot_body=np.asarray(cmd_feet),
                        actual_foot_world=np.asarray(act_feet),
                        joint_names=np.asarray(JOINT_NAMES))
    with (args.out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "root_z_m", "tilt_deg", "xy_drift_m", "vz_m_s",
                         "max_torque_nm", "min_contact_dist_m", "peak_contact_force_n", "ncon"])
        writer.writerows(rows[::control_steps])
    if frames:
        gif = args.out_dir / f"{stem}.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=round(1000 / args.fps), loop=0, optimize=False)
        report["animation"] = str(gif)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--back-shift", type=float, default=0.04)
    parser.add_argument("--forward-shift", type=float, default=0.04)
    parser.add_argument("--inward-shift", type=float, default=0.02)
    parser.add_argument("--outward-shift", type=float, default=0.02)
    parser.add_argument("--lift", type=float, default=0.04)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--front-s", type=float, default=0.6)
    parser.add_argument("--rear-s", type=float, default=0.6)
    parser.add_argument("--tuck-s", type=float, default=0.9)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results" / "manual_stand_to_compact")
    args = parser.parse_args()
    for name in ("back_shift", "forward_shift", "inward_shift", "outward_shift", "lift"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.fps <= 0 or args.fps > 50:
        parser.error("fps must be 1..50")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    geo = foot_geometry(load_model(args.model))
    chor = build_choreography(args, geo)
    report = simulate(args, geo, chor, not args.no_render)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RESULTS: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
