"""Collect 0.4 m/s walking-state snapshots for the walk->compact skill.

Runs the deploy-interface walking policy
(rollingquad_2_deploy_robust_dr_policy_stable.json, or any exported policy
JSON with the same metadata block) on the CPU MuJoCo mesh model
rollingquad_abd10.xml at a fixed 0.4 m/s forward command, then records real
reachable states (qpos/qvel/ctrl + the 20-frame deploy observation history)
after a warm-up window.  Training resets from these snapshots.

Run from curl_robot_2d:

    python -m scripts.collect_walking_start_snapshots ^
        --policy ..\\rollingquad_2_deploy_robust_dr_policy_stable.json ^
        --out results\\walk_start_snapshots_0p4

Only numpy + mujoco are required (CPU).  The physics options are pinned to
the runtime profile of the MJX training environment (0.002 s implicitfast,
pyramidal, Newton 20/10, impratio 10, eulerdamp disabled).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.walk_compact_3d import (
    COMMAND_M_S,
    CONTROL_TIMESTEP_S,
    HISTORY_SIZE,
    MESH_XML_REL,
    OBSERVATION_SIZE,
    PHYSICS_TIMESTEP_S,
    POLICY_JOINT_SUFFIXES,
    POLICY_LEGS,
    PROJECT_ROOT,
    SINGLE_OBS_SIZE,
    WALK_COMPACT_CONTRACT,
    xml_fingerprint,
)

LEGS = POLICY_LEGS
JOINTS = tuple(f"{leg}_{joint}" for leg in LEGS
               for joint in POLICY_JOINT_SUFFIXES)
JOINT_NAMES_ROBOT = ("front_right", "rear_left", "front_left", "rear_right")


def _forward(x, layers):
    for weights, bias, activation in layers:
        x = x @ weights + bias
        if activation == "elu":
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        elif activation == "tanh":
            x = np.tanh(x)
        elif activation not in ("linear", ""):
            raise ValueError(f"unsupported activation: {activation}")
    return x


def load_policy(policy_path: Path):
    payload = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    required = ("action_scale", "default_joint_pos", "joint_lower_limits",
                "joint_upper_limits", "observation_history", "in_shape", "layers")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"policy JSON missing {missing}")
    if payload["observation_history"] != HISTORY_SIZE:
        raise ValueError(f"policy history {payload['observation_history']} != {HISTORY_SIZE}")
    if payload["in_shape"][1] != HISTORY_SIZE * SINGLE_OBS_SIZE:
        raise ValueError("policy in_shape does not match 36x20")
    layers = [(np.asarray(layer["weights"][0], dtype=float),
               np.asarray(layer["weights"][1], dtype=float),
               layer["activation"]) for layer in payload["layers"]]
    return payload, layers


class WalkingRunner:
    """CPU deploy-policy runner; control layout mirrors compare_locomotion_energy."""

    def __init__(self, model, payload, layers):
        import mujoco
        self.scale = np.asarray(payload["action_scale"], dtype=float)
        self.pose = np.asarray(payload["default_joint_pos"], dtype=float)
        self.low = np.asarray(payload["joint_lower_limits"], dtype=float)
        self.high = np.asarray(payload["joint_upper_limits"], dtype=float)
        self.layers = layers
        self.hist = np.zeros((HISTORY_SIZE, SINGLE_OBS_SIZE))
        self.hist[:, 5] = -1.0          # stale stationary gravity -z
        self.hist[:, 11] = 1.0          # stale desired-world-z +z
        self.last = np.zeros(12)
        self.qids = np.asarray([model.joint(name).qposadr[0] for name in JOINTS])
        self.aids = np.asarray([model.actuator(f"{name}_servo").id
                                for name in JOINTS])
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                      "base_angular_velocity")
        self.angular_adr = (int(model.sensor_adr[sensor_id])
                            if sensor_id >= 0 else None)
        self.angular_dim = (int(model.sensor_dim[sensor_id])
                            if sensor_id >= 0 else None)

    def reset_history(self):
        self.hist[:] = 0.0
        self.hist[:, 5] = -1.0
        self.hist[:, 11] = 1.0
        self.last[:] = 0.0

    def control(self, data, speed, model):
        rotation = np.zeros(9)
        import mujoco as mj
        mj.mju_quat2Mat(rotation, data.qpos[3:7])
        rotation = rotation.reshape(3, 3)
        if self.angular_adr is not None:
            gyro = np.asarray(data.sensordata[self.angular_adr:
                                              self.angular_adr + self.angular_dim])
        else:
            gyro = rotation.T @ data.qvel[3:6]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        frame = np.concatenate((
            gyro, gravity, [speed, 0.0, 0.0], [0.0, 0.0, 1.0],
            data.qpos[self.qids] - self.pose, self.last))
        self.hist[1:] = self.hist[:-1].copy()
        self.hist[0] = frame
        self.last = np.clip(_forward(self.hist.reshape(-1), self.layers), -1.0, 1.0)
        data.ctrl[self.aids] = np.clip(self.pose + self.scale * self.last,
                                       self.low, self.high)


def _contact_ok(model, data, names):
    """Walking-posture filter: reject self penetration or non-foot ground."""
    for contact in data.contact:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = model.geom_bodyid[[g1, g2]]
        if b1 == 0 or b2 == 0:
            # ground (world body id 0) vs robot; robot name must look like a foot.
            robot_name = names[g2] if b1 == 0 else names[g1]
            if not any(tag in robot_name for tag in ("foot", "shank")):
                return False
        elif contact.dist < 0.0:
            return False
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    default_policy = PROJECT_ROOT.parent / "rollingquad_2_deploy_robust_dr_policy_stable.json"
    p.add_argument("--policy", type=Path, default=default_policy)
    p.add_argument("--xml", type=Path, default=PROJECT_ROOT / MESH_XML_REL)
    p.add_argument("--speed", type=float, default=COMMAND_M_S)
    p.add_argument("--warmup-s", type=float, default=1.5,
                   help="skip the first seconds of every episode")
    p.add_argument("--capture-s", type=float, default=5.0,
                   help="sample the following seconds of every episode")
    p.add_argument("--episodes", type=int, default=1,
                   help="independent episodes (deterministic unless noise added)")
    p.add_argument("--reset-noise", type=float, default=0.0,
                   help="per-episode joint/velocity noise scale (rad, rad/s)")
    p.add_argument("--speed-band", type=float, default=0.08,
                   help="accepted |actual vx - command| around the command")
    p.add_argument("--max-snapshots", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    def _resolve(path):
        path = Path(path)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    args.policy = _resolve(args.policy)
    args.xml = _resolve(args.xml)
    args.out = _resolve(args.out)
    return args


def main(argv=None):
    import mujoco as mj

    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not Path(args.policy).is_file():
        raise ValueError(f"policy not found: {args.policy}")
    xml_path = Path(args.xml)
    if not xml_path.is_file():
        raise ValueError(f"xml not found: {xml_path}")

    payload, layers = load_policy(args.policy)
    model = mj.MjModel.from_xml_path(str(xml_path))
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise ValueError(f"expected 19/18/12 model, got {model.nq}/{model.nv}/{model.nu}")
    # Pin physics to the walk-compact runtime profile (same as the MJX env).
    model.opt.timestep = PHYSICS_TIMESTEP_S
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    model.opt.impratio = 10
    model.opt.cone = mj.mjtCone.mjCONE_PYRAMIDAL
    model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_EULERDAMP
    runner = WalkingRunner(model, payload, layers)
    repeat = round(CONTROL_TIMESTEP_S / model.opt.timestep)
    if not np.isclose(repeat * model.opt.timestep, CONTROL_TIMESTEP_S):
        raise ValueError("physics timestep does not divide the 50 Hz control step")

    stand_key = model.key("stand").id
    stand_qpos = np.asarray(model.key_qpos[stand_key]).copy()
    rng = np.random.default_rng(args.seed)
    geom_names = {i: model.geom(i).name or "" for i in range(model.ngeom)}

    rows = {key: [] for key in ("qpos", "qvel", "ctrl", "hist", "last_action", "time")}
    warmup_steps = int(round(args.warmup_s / CONTROL_TIMESTEP_S))
    capture_steps = int(round(args.capture_s / CONTROL_TIMESTEP_S))
    episodes = max(args.episodes, 1)
    episodes_done = 0
    for episode in range(episodes):
        episodes_done += 1
        data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, data, stand_key)
        data.qpos[runner.qids] = runner.pose
        data.qpos[2] += 0.0005  # deployment-training reset clearance
        if args.reset_noise > 0.0 and episodes > 1:
            data.qpos[runner.qids] += rng.normal(0.0, args.reset_noise, 12)
            data.qvel[6:] += rng.normal(0.0, args.reset_noise, 12)
        runner.reset_history()
        mj.mj_forward(model, data)
        last_ctrl = np.asarray(data.ctrl).copy()
        step = 0
        while step < warmup_steps + capture_steps and \
                len(rows["qpos"]) < args.max_snapshots:
            if step % repeat == 0:
                runner.control(data, args.speed, model)
                last_ctrl = np.asarray(data.ctrl).copy()
                hist_snapshot = np.asarray(runner.hist, dtype=np.float32).reshape(-1)
                last_action = np.asarray(runner.last, dtype=np.float32).copy()
                t_s = float(data.time)
                vx, vy = float(data.qvel[0]), float(data.qvel[1])
                v_ok = abs(vx - args.speed) <= args.speed_band and abs(vy) <= 0.12
                rotation = np.zeros(9)
                mj.mju_quat2Mat(rotation, data.qpos[3:7])
                tilt_ok = float(np.degrees(np.arccos(np.clip(rotation[8], -1, 1)))) <= 20.0
                z = float(data.qpos[2])
                z_ok = 0.145 <= z <= 0.185
                posture_ok = (v_ok and tilt_ok and z_ok
                              and np.isfinite(data.qpos).all()
                              and np.isfinite(data.qvel).all()
                              and _contact_ok(model, data, geom_names))
                if step >= warmup_steps and posture_ok:
                    rows["qpos"].append(np.asarray(data.qpos, dtype=np.float32))
                    rows["qvel"].append(np.asarray(data.qvel, dtype=np.float32))
                    rows["ctrl"].append(last_ctrl)
                    rows["hist"].append(hist_snapshot)
                    rows["last_action"].append(last_action)
                    rows["time"].append(t_s)
            mj.mj_step(model, data)
            step += 1
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f"episode {episode}: nonfinite rollout")
        if len(rows["qpos"]) >= args.max_snapshots:
            break

    count = len(rows["qpos"])
    if count < 100:
        raise RuntimeError(f"only {count} snapshots collected; relax filters or extend capture")
    for key in rows:
        rows[key] = np.asarray(rows[key], dtype=np.float32)
    vx_all = rows["qvel"][:, 0]
    npz_path = args.out / "walk_start_snapshots.npz"
    np.savez_compressed(npz_path, **rows)
    meta = {
        "contract": WALK_COMPACT_CONTRACT,
        "count": count,
        "geometry": "rollingquad_2_abd10",
        "xml_basename": xml_path.name,
        **xml_fingerprint(xml_path),
        "policy_basename": Path(args.policy).name,
        "policy_sha256": hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
        "command_m_s": args.speed,
        "speed_band_m_s": args.speed_band,
        "episodes": episodes_done,
        "warmup_s": args.warmup_s,
        "capture_s": args.capture_s,
        "physics": {"timestep_s": PHYSICS_TIMESTEP_S, "control_dt_s": CONTROL_TIMESTEP_S,
                    "integrator": "implicitfast", "cone": "pyramidal",
                    "iterations": 20, "ls_iterations": 10, "impratio": 10},
        "observation": {"single": SINGLE_OBS_SIZE, "history": HISTORY_SIZE,
                        "size": OBSERVATION_SIZE},
        "action": {"default": [float(x) for x in payload["default_joint_pos"]],
                   "scale": [float(x) for x in payload["action_scale"]],
                   "lower": [float(x) for x in payload["joint_lower_limits"]],
                   "upper": [float(x) for x in payload["joint_upper_limits"]]},
        "observed": {"vx_mean_m_s": float(vx_all.mean()),
                     "vx_min_m_s": float(vx_all.min()),
                     "vx_max_m_s": float(vx_all.max()),
                     "vx_std_m_s": float(vx_all.std()),
                     "root_z_min_m": float(rows["qpos"][:, 2].min()),
                     "root_z_max_m": float(rows["qpos"][:, 2].max())},
        "note": "deploy-interface walking policy states; pose-only compact gate bank",
    }
    meta_path = args.out / "walk_start_snapshots_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"wrote {count} snapshots -> {npz_path}")
    print(f"observed vx mean {meta['observed']['vx_mean_m_s']:.3f} "
          f"min {meta['observed']['vx_min_m_s']:.3f} max {meta['observed']['vx_max_m_s']:.3f}")
    print(f"meta -> {meta_path}")
    return meta


if __name__ == "__main__":
    main()
