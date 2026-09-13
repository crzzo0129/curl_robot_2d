"""Compare rolling and walking mechanical work at a verified steady speed.

The rolling controller is initialized from a mature CEM rolling state because
the exported command-conditioned policy is not a stand-to-roll policy.  CEM
startup work is excluded.  Both modes then run on the same MuJoCo model and
physics settings.  A result is comparable only when the achieved speed passes
the predeclared settling and measurement-window gates.

Run from ``curl_robot_2d``.  Example::

    python -m scripts.compare_neural_energy_at_speed_3d \
      --rolling-policy C:/path/student_rtneural_000000696320.json \
      --out results/neural_energy_v060
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS as GEOMETRY
from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceGeometry,
    advance_oscillator,
    load_cem_reference,
    reference_action,
)
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d,
    disable_rollingquad_self_collision_3d,
    duplicate_planar_action_3d,
    forward_command_to_target_scale_3d,
    model_path_3d,
    reference_startup_scale_3d,
)
from scripts.export_rtneural import _activation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = model_path_3d("rollingquad_2_abd10_no_self_collision")
DEFAULT_CONTROLLER = ROOT / (
    "results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/"
    "01_zero_contact_speed_refine/best_phase_controller.json"
)
DEFAULT_WALKING = ROOT.parent / "rollingquad_2_deploy_robust_dr_policy_stable.json"
ACTIVE = np.asarray((1, 2, 4, 5, 7, 8, 10, 11))
ACTION_SCALES = np.asarray((0.0, 0.8, 1.2) * 4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Policy:
    """Float32 replay of the deployed RTNeural JSON contract."""

    def __init__(self, path: Path, model: mujoco.MjModel):
        self.path = path.resolve()
        self.doc = json.loads(self.path.read_text(encoding="utf-8"))
        if self.doc["in_shape"] != [1, 720] or self.doc["out_shape"] != [1, 12]:
            raise ValueError(f"Unexpected policy shape in {path}")
        if self.doc.get("observation_history") != 20:
            raise ValueError(f"Expected 20 observation frames in {path}")
        self.layers = [
            (layer, [np.asarray(value, dtype=np.float32) for value in layer["weights"]])
            for layer in self.doc["layers"]
        ]
        self.center, self.scale, self.low, self.high = [
            np.asarray(self.doc[key], dtype=np.float64)
            for key in ("default_joint_pos", "action_scale", "joint_lower_limits", "joint_upper_limits")
        ]
        self.kp = np.asarray(self.doc.get("kps", [self.doc["kp"]] * 12), dtype=np.float64)
        self.kd = np.asarray(self.doc.get("kds", [self.doc["kd"]] * 12), dtype=np.float64)
        self.joints = np.asarray([model.joint(name).id for name in CONTROLLER_JOINT_NAMES_3D])
        self.qids = model.jnt_qposadr[self.joints]
        self.vids = model.jnt_dofadr[self.joints]
        self.aids = np.asarray(
            [model.actuator(name + "_servo").id for name in CONTROLLER_JOINT_NAMES_3D]
        )
        self.history = self.cold_history()
        self.last_action = np.zeros(12, dtype=np.float32)

    @staticmethod
    def cold_history() -> np.ndarray:
        history = np.zeros((20, 36), dtype=np.float32)
        history[:, 5] = -1.0
        history[:, 11] = 1.0
        return history

    def set_gains(self, model: mujoco.MjModel) -> None:
        model.actuator_gainprm[self.aids, 0] = self.kp
        model.actuator_biasprm[self.aids, 1] = -self.kp
        model.actuator_biasprm[self.aids, 2] = -self.kd

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, command, action=None) -> np.ndarray:
        torso = model.body("torso").id
        rotation = data.xmat[torso].reshape(3, 3)
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, torso, spatial, 0
        )
        angular_body = rotation.T @ spatial[:3]
        last_action = self.last_action if action is None else action
        frame = np.concatenate(
            (
                angular_body,
                rotation.T @ np.asarray((0.0, 0.0, -1.0)),
                np.asarray(command, dtype=np.float64),
                np.asarray((0.0, 0.0, 1.0)),
                data.qpos[self.qids] - self.center,
                last_action,
            )
        ).astype(np.float32)
        self.history[1:] = self.history[:-1].copy()
        self.history[0] = np.nan_to_num(frame)
        return self.history.reshape(-1)

    def infer(self, observation: np.ndarray) -> np.ndarray:
        x = np.clip(observation, -100.0, 100.0).astype(np.float32)
        for layer, weights in self.layers:
            if layer["type"] == "batchnorm":
                gamma, beta, mean, variance = weights
                x = (x - mean) * (
                    gamma / np.sqrt(variance + np.float32(layer["epsilon"]))
                ) + beta
            elif layer["type"] == "dense":
                kernel, bias = weights
                x = _activation(layer["activation"], x @ kernel + bias)
            else:
                raise ValueError(f"Unsupported layer: {layer['type']}")
        if not np.isfinite(x).all():
            raise RuntimeError(f"Nonfinite policy output from {self.path}")
        return np.clip(x, -1.0, 1.0)

    def step_control(self, model, data, command) -> np.ndarray:
        action = self.infer(self.observe(model, data, command))
        target = np.clip(self.center + self.scale * action, self.low, self.high)
        data.ctrl[self.aids] = target
        self.last_action = action.astype(np.float32)
        return target

    def encode_target(self, target: np.ndarray) -> np.ndarray:
        action = np.zeros(12, dtype=np.float32)
        moving = self.scale != 0
        action[moving] = ((target[moving] - self.center[moving]) / self.scale[moving]).astype(np.float32)
        return np.clip(action, -1.0, 1.0)


class CEMBootstrap:
    """Nominal calibrated straight CEM used only to create mature rolling state."""

    def __init__(self, model, policy: Policy, controller: Path, command: float):
        self.model = model
        self.policy = policy
        self.reference = load_cem_reference(controller, reference_weight=1.0, minimum_residual_gain=0.15)
        self.task = physics_profile_3d(
            "cg20",
            Rolling3DConfig(
                geometry="rollingquad_2_abd10_no_self_collision",
                self_collision_enabled=False,
                reset_joint_noise_rad=0.0,
                reset_velocity_noise=0.0,
                disable_root_damping=True,
            ),
        )
        self.command = command
        self.command_scale = float(forward_command_to_target_scale_3d(np, command))
        self.phase = 0.0
        self.spin = 0.0
        self.planar = np.asarray(
            (GEOMETRY.compact_hip_angle, GEOMETRY.compact_knee_angle) * 2
        )
        self.planar_scale = np.asarray((0.8, 1.2) * 2)
        self.planar_low = np.asarray(
            (GEOMETRY.hip.shell_compatible_range[0], GEOMETRY.knee.shell_compatible_range[0]) * 2
        )
        self.planar_high = np.asarray(
            (GEOMETRY.hip.shell_compatible_range[1], GEOMETRY.knee.shell_compatible_range[1]) * 2
        )
        self.geometry = CEMReferenceGeometry(
            torso_length_m=GEOMETRY.torso_length,
            link_length_m=GEOMETRY.edge_length,
            foot_diameter_m=2 * GEOMETRY.foot_radius,
            upper_link_length_m=GEOMETRY.upper_length,
            lower_link_length_m=GEOMETRY.lower_length,
        )

    def target(self, data) -> np.ndarray:
        self.phase = float(
            advance_oscillator(
                np, self.spin, self.phase, self.model.opt.timestep, self.reference
            )
        )
        planar = reference_action(
            np,
            self.phase,
            self.reference,
            compact_ctrl=self.planar,
            action_scales=self.planar_scale,
            joint_low=self.planar_low,
            joint_high=self.planar_high,
            geometry=self.geometry,
        )
        scale = reference_startup_scale_3d(
            np, data.time, self.task, target_scale=self.command_scale
        )
        effective = np.clip(scale * duplicate_planar_action_3d(np, planar), -1.0, 1.0)
        target = self.policy.center.copy()
        target[ACTIVE] += effective * ACTION_SCALES[ACTIVE]
        return np.clip(target, self.policy.low, self.policy.high)

    def advance_spin(self, data) -> None:
        self.spin += float(data.qvel[4]) * self.model.opt.timestep


def make_model(xml: Path) -> mujoco.MjModel:
    task = physics_profile_3d(
        "cg20",
        Rolling3DConfig(
            geometry="rollingquad_2_abd10_no_self_collision",
            self_collision_enabled=False,
            reset_joint_noise_rad=0.0,
            reset_velocity_noise=0.0,
            disable_root_damping=True,
        ),
    )
    model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
    apply_physics_options_3d(model, task)
    disable_rollingquad_self_collision_3d(model)
    return model


def diagnostic_row(model, data, mode, command, power, policy: Policy) -> dict:
    torso = model.body("torso").id
    rotation = data.xmat[torso].reshape(3, 3)
    row = {
        "time_s": float(data.time),
        "mode": mode,
        "command_m_s": command,
        "x_m": float(data.qpos[0]),
        "y_m": float(data.qpos[1]),
        "height_m": float(data.qpos[2]),
        "vx_m_s": float(data.qvel[0]),
        "positive_power_w": float(np.maximum(power, 0.0).sum()),
        "negative_power_w": float(np.maximum(-power, 0.0).sum()),
        "axis_tilt_deg": float(np.degrees(np.arcsin(np.clip(abs(rotation[2, 1]), 0, 1)))),
        "upright_tilt_deg": float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
        "saturation_fraction": float(
            np.mean(
                np.abs(data.actuator_force)
                >= 0.99 * np.maximum(np.abs(model.actuator_forcerange).max(axis=1), 1e-9)
            )
        ),
    }
    for name, qid, aid in zip(CONTROLLER_JOINT_NAMES_3D, policy.qids, policy.aids):
        row[f"{name}_torque_nm"] = float(data.actuator_force[aid])
        row[f"{name}_speed_rad_s"] = float(data.actuator_velocity[aid])
        row[f"{name}_power_w"] = float(power[aid])
        row[f"{name}_position_rad"] = float(data.qpos[qid])
        row[f"{name}_target_rad"] = float(data.ctrl[aid])
    return row


def simulate(args, mode: str, command: float) -> tuple[list[dict], dict]:
    model = make_model(args.xml)
    data = mujoco.MjData(model)
    policy_path = args.rolling_policy if mode == "roll" else args.walking_policy
    policy = Policy(policy_path, model)
    policy.set_gains(model)
    keyframe = "compact" if mode == "roll" else "stand"
    mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    data.qpos[policy.qids] = policy.center
    data.qvel[:] = 0.0
    data.ctrl[policy.aids] = policy.center
    if mode == "walk":
        data.qpos[2] += 0.0005
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    repeat = round(0.02 / dt)
    if not np.isclose(repeat * dt, 0.02):
        raise ValueError("Physics timestep must divide the 20 ms policy period")
    command_vector = np.asarray((command, 0.0, 0.0))
    bootstrap = CEMBootstrap(model, policy, args.controller, command) if mode == "roll" else None
    bootstrap_steps = round(args.rolling_bootstrap / dt) if bootstrap else 0
    initial_policy_delta = None

    # Populate the exact rolling-policy history while the CEM makes a mature
    # state.  This is the same state/history premise used by snapshot training.
    for k in range(bootstrap_steps):
        if k % repeat == 0:
            policy.observe(model, data, command_vector)
        target = bootstrap.target(data)
        data.ctrl[policy.aids] = target
        policy.last_action = policy.encode_target(target)
        mujoco.mj_step(model, data)
        bootstrap.advance_spin(data)
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all():
            raise RuntimeError("Nonfinite CEM rolling bootstrap")

    policy_start_time = float(data.time)
    rows = []
    steps = round(args.max_policy_duration / dt)
    for k in range(steps):
        if k % repeat == 0:
            old = data.ctrl[policy.aids].copy()
            target = policy.step_control(model, data, command_vector)
            if initial_policy_delta is None:
                initial_policy_delta = float(np.max(np.abs(target - old)))
        mujoco.mj_forward(model, data)
        power = data.actuator_force * data.actuator_velocity
        rows.append(diagnostic_row(model, data, mode, command, power, policy))
        mujoco.mj_step(model, data)
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all():
            raise RuntimeError(f"Nonfinite {mode} state at {data.time:.3f} s")
    return rows, {
        "mode": mode,
        "command_m_s": command,
        "policy_start_time_s": policy_start_time,
        "rolling_bootstrap_s": args.rolling_bootstrap if mode == "roll" else 0.0,
        "initial_policy_target_delta_rad": initial_policy_delta,
        "mass_kg": float(model.body_mass.sum()),
        "gravity_m_s2": float(np.linalg.norm(model.opt.gravity)),
    }


def interval_speed(rows, start, end) -> float:
    selected = [row for row in rows if start - 1e-12 <= row["time_s"] <= end + 1e-12]
    if len(selected) < 2:
        return math.nan
    return (selected[-1]["x_m"] - selected[0]["x_m"]) / (
        selected[-1]["time_s"] - selected[0]["time_s"]
    )


def select_window(rows, meta, args) -> dict:
    target = args.target_speed
    first = meta["policy_start_time_s"]
    last = rows[-1]["time_s"]
    settle = args.settle_window
    settle_block = args.settle_block_duration
    measure = args.measurement_window
    # Search on fixed 20 ms boundaries for the earliest valid endpoint of the
    # settling interval.  This prevents post-hoc choice of a favorable window.
    candidates = np.arange(first + settle, last - measure + 1e-9, 0.02)
    chosen = None
    chosen_blocks = None
    for start in candidates:
        blocks = [
            interval_speed(
                rows,
                start - settle + settle_block * i,
                start - settle + settle_block * (i + 1),
            )
            for i in range(round(settle / settle_block))
        ]
        if (
            all(np.isfinite(blocks))
            and all(abs(speed - target) <= args.settle_block_tolerance for speed in blocks)
            and max(blocks) - min(blocks) <= args.settle_block_range
            and abs(float(np.polyfit(np.arange(len(blocks)), blocks, 1)[0]))
            <= args.settle_slope_limit
        ):
            chosen, chosen_blocks = round(float(start), 6), blocks
            break
    result = {
        "passed": False,
        "failure": "no predeclared steady-speed window found",
        "settle_window_s": settle,
        "settle_block_duration_s": settle_block,
        "measurement_window_s": measure,
        "settle_block_speeds_m_s": chosen_blocks,
    }
    if chosen is None:
        return result
    times = np.asarray([row["time_s"] for row in rows])
    dt = float(np.median(np.diff(times)))
    start_index = int(np.argmin(np.abs(times - chosen)))
    sample_count = round(measure / dt)
    end_index = start_index + sample_count
    if end_index >= len(rows):
        result["failure"] = "measurement window exceeds recorded rollout"
        return result
    selected = rows[start_index:end_index]
    endpoint = rows[end_index]
    duration = sample_count * dt
    dx = sum(row["vx_m_s"] * dt for row in selected)
    # Use position endpoints for speed and distance; integrate power at the
    # physics rate from the state at which force and transmission velocity agree.
    speed = (endpoint["x_m"] - selected[0]["x_m"]) / (
        endpoint["time_s"] - selected[0]["time_s"]
    )
    positive = sum(row["positive_power_w"] for row in selected) * dt
    negative = sum(row["negative_power_w"] for row in selected) * dt
    blocks = [interval_speed(rows, chosen + 2 * i, chosen + 2 * (i + 1))
              for i in range(round(measure / 2))]
    steady_measurement = (
        abs(speed - target) <= args.measurement_mean_tolerance
        and all(abs(value - target) <= args.measurement_block_tolerance for value in blocks)
        and abs(float(np.polyfit(np.arange(len(blocks)), blocks, 1)[0]))
        <= 2 * args.settle_slope_limit
    )
    distance = abs(speed * measure)
    result.update(
        passed=bool(steady_measurement),
        failure=None if steady_measurement else "measurement window did not remain at steady target speed",
        start_time_s=chosen,
        end_time_s=chosen + measure,
        duration_s=duration,
        actual_speed_m_s=speed,
        measurement_2s_block_speeds_m_s=blocks,
        forward_distance_m=distance,
        integrated_vx_distance_m=dx,
        lateral_displacement_m=endpoint["y_m"] - selected[0]["y_m"],
        positive_work_j=positive,
        negative_work_j=negative,
        absolute_work_j=positive + negative,
        positive_power_w=positive / duration,
        absolute_power_w=(positive + negative) / duration,
        positive_j_per_m=positive / distance,
        absolute_j_per_m=(positive + negative) / distance,
        cot_positive=positive / (meta["mass_kg"] * meta["gravity_m_s2"] * distance),
        cot_absolute=(positive + negative) / (meta["mass_kg"] * meta["gravity_m_s2"] * distance),
        max_axis_tilt_deg=max(row["axis_tilt_deg"] for row in selected),
        mean_saturation_fraction=float(np.mean([row["saturation_fraction"] for row in selected])),
    )
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-policy", type=Path, required=True)
    parser.add_argument("--walking-policy", type=Path, default=DEFAULT_WALKING)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-speed", type=float, default=0.6)
    parser.add_argument("--roll-command", type=float, default=0.6)
    parser.add_argument("--walk-command", type=float, default=0.6)
    parser.add_argument("--rolling-bootstrap", type=float, default=5.0)
    parser.add_argument("--max-policy-duration", type=float, default=30.0)
    parser.add_argument("--settle-window", type=float, default=6.0)
    parser.add_argument("--settle-block-duration", type=float, default=2.0)
    parser.add_argument("--measurement-window", type=float, default=10.0)
    parser.add_argument("--settle-block-tolerance", type=float, default=0.04)
    parser.add_argument("--settle-block-range", type=float, default=0.04)
    parser.add_argument("--settle-slope-limit", type=float, default=0.015)
    parser.add_argument("--measurement-mean-tolerance", type=float, default=0.03)
    parser.add_argument("--measurement-block-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    paths = (args.rolling_policy, args.walking_policy, args.xml, args.controller)
    if any(not path.is_file() for path in paths):
        parser.error("Every policy/model/controller input must exist")
    if args.out.exists():
        parser.error("Output directory must be new")
    numeric = [
        args.target_speed, args.roll_command, args.walk_command, args.rolling_bootstrap,
        args.max_policy_duration, args.settle_window, args.settle_block_duration,
        args.measurement_window,
        args.settle_block_tolerance, args.settle_block_range, args.settle_slope_limit,
        args.measurement_mean_tolerance, args.measurement_block_tolerance,
    ]
    if not np.isfinite(numeric).all() or min(
        args.target_speed,
        args.max_policy_duration,
        args.settle_window,
        args.settle_block_duration,
        args.measurement_window,
    ) <= 0:
        parser.error("Numeric settings must be finite and durations/speed positive")
    if (not float(args.settle_window / args.settle_block_duration).is_integer()
            or args.settle_window / args.settle_block_duration < 3
            or not float(args.measurement_window / 2).is_integer()):
        parser.error("Settle window must contain at least three whole blocks; measurement window a multiple of 2 s")
    if args.max_policy_duration < args.settle_window + args.measurement_window:
        parser.error("Policy duration must contain settling plus measurement windows")

    args.out.mkdir(parents=True)
    manifest = {
        "status": "RUNNING",
        "hypothesis": "Rolling uses less positive mechanical work per metre than walking at verified steady 0.6 m/s.",
        "comparison_gate": "Both modes must pass the same predeclared achieved-speed settling and 10 s measurement gates.",
        "energy_scope": "Actuator mechanical work only; not battery/electrical energy.",
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sha256": {path.stem: sha256(path) for path in paths},
        "mujoco_version": mujoco.__version__,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = []
    for mode, command in (("roll", args.roll_command), ("walk", args.walk_command)):
        print(f"[{mode}] command={command:.4f} m/s", flush=True)
        rows, meta = simulate(args, mode, command)
        window = select_window(rows, meta, args)
        result = {**meta, "policy": str((args.rolling_policy if mode == "roll" else args.walking_policy).resolve()),
                  "window": window}
        results.append(result)
        write_rows(args.out / f"{mode}_trace.csv", rows)
        print(json.dumps(result, indent=2), flush=True)
    comparable = all(result["window"]["passed"] for result in results)
    summary = {"comparable_at_target_speed": comparable, "results": results}
    if comparable:
        roll, walk = (result["window"] for result in results)
        summary["comparison"] = {
            "rolling_positive_j_per_m_reduction_fraction": 1.0 - roll["positive_j_per_m"] / walk["positive_j_per_m"],
            "rolling_absolute_j_per_m_reduction_fraction": 1.0 - roll["absolute_j_per_m"] / walk["absolute_j_per_m"],
            "actual_speed_mismatch_fraction": abs(roll["actual_speed_m_s"] - walk["actual_speed_m_s"]) / args.target_speed,
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "COMPLETED"
    manifest["comparable_at_target_speed"] = comparable
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
