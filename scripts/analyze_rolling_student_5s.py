#!/usr/bin/env python3
"""Run and plot a nominal 5 s CPU MuJoCo rollout of the rolling STUDENT.

The simulation reproduces the distillation acceptance contract: compact reset,
50 Hz policy updates, 1 kHz physics, newest-first 20-frame deployment history,
and direct full-amplitude student actions from the first policy update.

The script has separate ``simulate`` and ``plot`` modes so the MuJoCo and
Matplotlib Python environments may be different on a workstation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "FL_abd", "FL_hip", "FL_knee",
    "FR_abd", "FR_hip", "FR_knee",
    "RL_abd", "RL_hip", "RL_knee",
    "RR_abd", "RR_hip", "RR_knee",
)
MODEL_JOINT_NAMES = tuple(
    f"{leg}_{joint}"
    for leg in ("front_left", "front_right", "rear_left", "rear_right")
    for joint in ("hip_abduction", "hip", "knee")
)
CONTROL_DT = 0.020
PHYSICS_DT = 0.001
SUBSTEPS = 20
DURATION_S = 5.0
TORQUE_LIMIT_NM = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("simulate", "plot"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def activation(name: str, value: np.ndarray) -> np.ndarray:
    if name == "elu":
        return np.where(value > 0.0, value, np.expm1(value))
    if name == "relu":
        return np.maximum(value, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-value))
    if name == "tanh":
        return np.tanh(value)
    raise ValueError(f"unsupported activation: {name}")


def load_policy(path: Path) -> tuple[dict, list[tuple[np.ndarray, np.ndarray, str]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    layers = []
    expected_input = int(document["in_shape"][-1])
    for index, layer in enumerate(document["layers"]):
        kernel = np.asarray(layer["weights"][0], dtype=np.float32)
        bias = np.asarray(layer["weights"][1], dtype=np.float32)
        if kernel.shape[0] != expected_input or kernel.shape[1] != bias.size:
            raise ValueError(f"layer {index} has inconsistent shapes")
        layers.append((kernel, bias, layer["activation"]))
        expected_input = bias.size
    if expected_input != 12:
        raise ValueError(f"expected 12 policy outputs, got {expected_input}")
    return document, layers


def infer(layers: list[tuple[np.ndarray, np.ndarray, str]], observation: np.ndarray) -> np.ndarray:
    value = observation.astype(np.float32, copy=False)
    for kernel, bias, name in layers:
        value = activation(name, value @ kernel + bias).astype(np.float32)
    return value


def initial_history() -> np.ndarray:
    frame = np.zeros(36, dtype=np.float32)
    frame[5] = -1.0
    frame[11] = 1.0
    return np.tile(frame, 20)


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def policy_frame(
    data,
    torso_id: int,
    joint_qpos_ids: np.ndarray,
    compact: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(data.xmat[torso_id]).reshape(3, 3)
    angular_world = np.asarray(data.cvel[torso_id, :3])
    angular_body = rotation.T @ angular_world
    projected_gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0))
    frame = np.concatenate(
        (
            angular_body,
            projected_gravity,
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray(data.qpos[joint_qpos_ids]) - compact,
            previous_action,
        )
    )
    if frame.shape != (36,):
        raise RuntimeError(f"deployment frame has wrong shape: {frame.shape}")
    return frame.astype(np.float32)


def configure_training_physics(model, mujoco) -> None:
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    model.opt.timestep = PHYSICS_DT
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_dof = int(model.jnt_dofadr[root_id])
    model.dof_damping[root_dof : root_dof + 6] = 0.0


def state_values(data, torso_id: int, rolling_phase: float) -> dict[str, float]:
    rotation = np.asarray(data.xmat[torso_id]).reshape(3, 3)
    # For pure +y rolling, atan2(R_xz, R_xx) is the signed pitch in [-pi, pi].
    # It remains interpretable through 90 degrees, unlike ZYX asin pitch.
    pitch_wrapped = math.atan2(float(rotation[0, 2]), float(rotation[0, 0]))
    axis_alignment = float(np.clip(abs(rotation[1, 1]), 0.0, 1.0))
    axis_tilt = math.acos(axis_alignment)
    return {
        "pitch_wrapped_rad": pitch_wrapped,
        "phase_rad": rolling_phase,
        "phase_wrapped_rad": float(wrap_angle(rolling_phase)),
        "phase_turns": rolling_phase / (2.0 * math.pi),
        "phase_rate_rad_s": float(data.qvel[4]),
        "axis_tilt_rad": axis_tilt,
        "root_x_m": float(data.qpos[0]),
        "root_y_m": float(data.qpos[1]),
        "root_z_m": float(data.qpos[2]),
        "contact_count": float(data.ncon),
    }


def simulate(policy_path: Path, model_path: Path, out_dir: Path) -> None:
    import mujoco

    out_dir.mkdir(parents=True, exist_ok=True)
    policy, layers = load_policy(policy_path)
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    configure_training_physics(model, mujoco)
    data = mujoco.MjData(model)
    compact_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if min(compact_key, torso_id, floor_geom_id) < 0:
        raise ValueError("model must contain compact keyframe and torso body")

    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_servo")
            for name in MODEL_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(actuator_ids < 0):
        raise ValueError("model actuator names do not match the deployment ABI")
    actuator_joint_ids = np.asarray(model.actuator_trnid[actuator_ids, 0], dtype=np.int32)
    joint_qpos_ids = np.asarray(model.jnt_qposadr[actuator_joint_ids], dtype=np.int32)

    mujoco.mj_resetDataKeyframe(model, data, compact_key)
    compact = np.asarray(policy["default_joint_pos"], dtype=np.float64)
    action_scale = np.asarray(policy["action_scale"], dtype=np.float64)
    joint_low = np.asarray(policy["joint_lower_limits"], dtype=np.float64)
    joint_high = np.asarray(policy["joint_upper_limits"], dtype=np.float64)
    np.testing.assert_allclose(data.qpos[joint_qpos_ids], compact, atol=1e-7)
    data.ctrl[actuator_ids] = compact
    mujoco.mj_forward(model, data)

    history = initial_history()
    previous_action = np.zeros(12, dtype=np.float32)
    rolling_phase = 0.0
    policy_steps = int(round(DURATION_S / CONTROL_DT))
    records: list[dict[str, float]] = []
    self_pair_stats: dict[tuple[str, str], dict[str, float | int]] = {}
    previous_self_pairs: set[tuple[str, str]] = set()

    def self_contact_state() -> tuple[int, float, dict[tuple[str, str], float]]:
        contact_points = 0
        maximum_penetration = 0.0
        active_pairs: dict[tuple[str, str], float] = {}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            if floor_geom_id in (geom_1, geom_2):
                continue
            body_1 = int(model.geom_bodyid[geom_1])
            body_2 = int(model.geom_bodyid[geom_2])
            if body_1 == body_2:
                continue
            name_1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_1) or f"body_{body_1}"
            name_2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_2) or f"body_{body_2}"
            pair = tuple(sorted((name_1, name_2)))
            contact_points += 1
            penetration = max(-float(contact.dist), 0.0)
            maximum_penetration = max(maximum_penetration, penetration)
            active_pairs[pair] = max(active_pairs.get(pair, 0.0), penetration)
        return contact_points, maximum_penetration, active_pairs

    def append_record(action: np.ndarray, target: np.ndarray) -> None:
        nonlocal previous_self_pairs
        self_points, self_penetration, active_pairs = self_contact_state()
        record = {"time_s": float(data.time), **state_values(data, torso_id, rolling_phase)}
        record["self_contact_count"] = float(self_points)
        record["self_penetration_max_m"] = self_penetration
        for index, name in enumerate(JOINT_NAMES):
            record[f"torque_{name}_Nm"] = float(data.actuator_force[actuator_ids[index]])
            record[f"position_{name}_rad"] = float(data.qpos[joint_qpos_ids[index]])
            record[f"target_{name}_rad"] = float(target[index])
            record[f"action_{name}"] = float(action[index])
        records.append(record)
        for pair, pair_penetration in active_pairs.items():
            stats = self_pair_stats.setdefault(
                pair,
                {
                    "active_steps": 0,
                    "onsets": 0,
                    "first_time_s": float(data.time),
                    "last_time_s": float(data.time),
                    "maximum_penetration_m": 0.0,
                },
            )
            stats["active_steps"] = int(stats["active_steps"]) + 1
            stats["last_time_s"] = float(data.time)
            stats["maximum_penetration_m"] = max(
                float(stats["maximum_penetration_m"]), pair_penetration
            )
            if pair not in previous_self_pairs:
                stats["onsets"] = int(stats["onsets"]) + 1
        previous_self_pairs = set(active_pairs)

    append_record(previous_action, compact)
    for _ in range(policy_steps):
        frame = policy_frame(data, torso_id, joint_qpos_ids, compact, previous_action)
        history = np.concatenate((frame, history[:-36])).astype(np.float32)
        action = np.clip(infer(layers, history), -1.0, 1.0)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("policy produced a non-finite action")
        target = np.clip(compact + action_scale * action, joint_low, joint_high)
        data.ctrl[actuator_ids] = target
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
            rolling_phase += PHYSICS_DT * float(data.qvel[4])
            append_record(action, target)
        previous_action = action

    fieldnames = list(records[0])
    csv_path = out_dir / "rolling_student_5s_timeseries.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    arrays = {name: np.asarray([row[name] for row in records]) for name in fieldnames}
    np.savez_compressed(out_dir / "rolling_student_5s_timeseries.npz", **arrays)

    torque = np.column_stack([arrays[f"torque_{name}_Nm"] for name in JOINT_NAMES])
    action = np.column_stack([arrays[f"action_{name}"] for name in JOINT_NAMES])
    target = np.column_stack([arrays[f"target_{name}_rad"] for name in JOINT_NAMES])
    phase_error = wrap_angle(arrays["pitch_wrapped_rad"] - arrays["phase_wrapped_rad"])
    active = arrays["time_s"] > 0.0
    torque_active = torque[active]
    crossing_times = []
    for turn in range(1, int(math.floor(arrays["phase_turns"][-1])) + 1):
        reached = np.flatnonzero(arrays["phase_turns"] >= turn)
        if reached.size:
            index = int(reached[0])
            crossing_times.append(float(arrays["time_s"][index]))
    cycle_periods = np.diff(crossing_times)
    per_joint = []
    for index, name in enumerate(JOINT_NAMES):
        values = torque_active[:, index]
        per_joint.append(
            {
                "joint": name,
                "rms_Nm": float(np.sqrt(np.mean(np.square(values)))),
                "mean_abs_Nm": float(np.mean(np.abs(values))),
                "p99_abs_Nm": float(np.percentile(np.abs(values), 99)),
                "peak_abs_Nm": float(np.max(np.abs(values))),
                "saturation_fraction": float(np.mean(np.abs(values) >= 0.99 * TORQUE_LIMIT_NM)),
            }
        )

    settled = arrays["time_s"] >= 1.0
    self_contact_active = arrays["self_contact_count"][active] > 0.0
    self_collision_enabled = bool(
        np.any(
            np.asarray(model.geom_contype)[np.arange(model.ngeom) != floor_geom_id]
            & np.asarray(model.geom_conaffinity)[np.arange(model.ngeom) != floor_geom_id]
        )
    )
    self_pairs = [
        {
            "body_1": pair[0],
            "body_2": pair[1],
            **stats,
            "active_fraction": int(stats["active_steps"]) / (policy_steps * SUBSTEPS),
        }
        for pair, stats in sorted(
            self_pair_stats.items(),
            key=lambda item: (-int(item[1]["active_steps"]), item[0]),
        )
    ]
    summary = {
        "run": {
            "duration_s": DURATION_S,
            "policy_frequency_Hz": 1.0 / CONTROL_DT,
            "physics_frequency_Hz": 1.0 / PHYSICS_DT,
            "policy_steps": policy_steps,
            "physics_steps": policy_steps * SUBSTEPS,
            "reset": "nominal compact keyframe; no reset noise",
            "action_start": "full direct student output from first policy update; no fade-in",
            "physics_profile": "cg20, implicitfast, elliptic cone, dense Jacobian",
            "self_collision_enabled": self_collision_enabled,
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "model": str(model_path.resolve()),
        },
        "phase": {
            "definition": "signed integral of root qvel[4] at each 1 ms physics step",
            "signed_turns": float(arrays["phase_turns"][-1]),
            "absolute_turns": float(abs(arrays["phase_turns"][-1])),
            "mean_rate_rad_s": float(arrays["phase_rad"][-1] / DURATION_S),
            "mean_rate_rev_s": float(arrays["phase_turns"][-1] / DURATION_S),
            "settled_mean_rate_rad_s": float(np.mean(arrays["phase_rate_rad_s"][settled])),
            "settled_rate_std_rad_s": float(np.std(arrays["phase_rate_rad_s"][settled])),
            "completed_turn_crossing_times_s": crossing_times,
            "completed_cycle_periods_s": [float(value) for value in cycle_periods],
            "completed_cycle_period_mean_s": (
                float(np.mean(cycle_periods)) if cycle_periods.size else None
            ),
            "completed_cycle_period_std_s": (
                float(np.std(cycle_periods)) if cycle_periods.size else None
            ),
        },
        "pitch": {
            "definition": "wrapped torso +y angle atan2(R[0,2], R[0,0])",
            "minimum_deg": float(np.degrees(np.min(arrays["pitch_wrapped_rad"]))),
            "maximum_deg": float(np.degrees(np.max(arrays["pitch_wrapped_rad"]))),
            "phase_tracking_error_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(phase_error))))),
            "phase_tracking_error_p99_abs_deg": float(np.degrees(np.percentile(np.abs(phase_error), 99))),
        },
        "torque": {
            "definition": "MuJoCo actuator_force after each 1 ms step",
            "limit_Nm": TORQUE_LIMIT_NM,
            "overall_rms_Nm": float(np.sqrt(np.mean(np.square(torque_active)))),
            "overall_p99_abs_Nm": float(np.percentile(np.abs(torque_active), 99)),
            "overall_peak_abs_Nm": float(np.max(np.abs(torque_active))),
            "overall_saturation_fraction": float(
                np.mean(np.abs(torque_active) >= 0.99 * TORQUE_LIMIT_NM)
            ),
            "startup_0_to_1s_rms_Nm": float(
                np.sqrt(np.mean(np.square(torque[(arrays["time_s"] > 0.0) & (arrays["time_s"] <= 1.0)])))
            ),
            "steady_1_to_5s_rms_Nm": float(np.sqrt(np.mean(np.square(torque[settled])))),
            "per_joint": per_joint,
        },
        "first_policy_update": {
            "normalized_action": {
                name: float(action[1, index]) for index, name in enumerate(JOINT_NAMES)
            },
            "joint_target_rad": {
                name: float(target[1, index]) for index, name in enumerate(JOINT_NAMES)
            },
        },
        "stability": {
            "root_x_displacement_m": float(arrays["root_x_m"][-1] - arrays["root_x_m"][0]),
            "lateral_drift_m": float(arrays["root_y_m"][-1] - arrays["root_y_m"][0]),
            "root_z_min_m": float(np.min(arrays["root_z_m"])),
            "root_z_max_m": float(np.max(arrays["root_z_m"])),
            "axis_tilt_max_deg": float(np.degrees(np.max(arrays["axis_tilt_rad"]))),
            "action_saturation_fraction": float(np.mean(np.abs(action[active]) >= 0.99)),
        },
        "self_collision": {
            "enabled": self_collision_enabled,
            "compact_initial_contact_points": int(arrays["self_contact_count"][0]),
            "active_step_fraction": float(np.mean(self_contact_active)),
            "maximum_simultaneous_contact_points": int(np.max(arrays["self_contact_count"])),
            "maximum_penetration_m": float(np.max(arrays["self_penetration_max_m"])),
            "pairs": self_pairs,
        },
    }
    (out_dir / "rolling_student_5s_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "rolling_student_5s_torque_by_joint.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(per_joint[0]))
        writer.writeheader()
        writer.writerows(per_joint)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.55, alpha=0.75)
    axis.tick_params(width=0.7, length=3)


def plot(out_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.7,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    values = np.load(out_dir / "rolling_student_5s_timeseries.npz")
    summary = json.loads((out_dir / "rolling_student_5s_summary.json").read_text(encoding="utf-8"))
    time = values["time_s"]
    phase = values["phase_rad"]
    phase_turns = values["phase_turns"]
    pitch = values["pitch_wrapped_rad"]
    phase_error = wrap_angle(pitch - values["phase_wrapped_rad"])
    torque = np.column_stack([values[f"torque_{name}_Nm"] for name in JOINT_NAMES])
    torque_stats = summary["torque"]["per_joint"]
    rms = np.asarray([row["rms_Nm"] for row in torque_stats])
    peak = np.asarray([row["peak_abs_Nm"] for row in torque_stats])

    leg_colors = {"FL": "#0072B2", "FR": "#D55E00", "RL": "#009E73", "RR": "#CC79A7"}
    joint_styles = {"hip": "-", "knee": "--"}
    figure = plt.figure(figsize=(7.2, 8.35), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.05, 1.0, 1.12))

    ax = figure.add_subplot(grid[0, 0])
    ax.plot(time, phase_turns, color="#202A44", linewidth=1.35)
    ax.axhline(0.0, color="#8A94A6", linewidth=0.6)
    ax.set(xlabel="Time (s)", ylabel="Signed rolling phase (turns)", title="a  Continuous rolling progress")
    style_axis(ax)
    ax.text(
        0.03,
        0.94,
        f"{summary['phase']['signed_turns']:.2f} turns / 5 s\n"
        f"{summary['phase']['mean_rate_rev_s']:.2f} rev s$^{{-1}}$",
        transform=ax.transAxes,
        va="top",
        color="#202A44",
    )

    ax = figure.add_subplot(grid[0, 1])
    ax.plot(time, np.degrees(pitch), color="#0072B2", linewidth=0.95, label="Torso pitch (wrapped)")
    ax.plot(
        time,
        np.degrees(values["phase_wrapped_rad"]),
        color="#D55E00",
        linestyle="--",
        linewidth=0.8,
        alpha=0.9,
        label="Integrated phase (wrapped)",
    )
    ax.axhline(0.0, color="#8A94A6", linewidth=0.6)
    ax.set(
        xlabel="Time (s)",
        ylabel="Wrapped angle (deg)",
        ylim=(-190, 190),
        yticks=(-180, -90, 0, 90, 180),
        title="b  Torso pitch and phase locking",
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.03,
        0.06,
        f"pitch–phase RMS {summary['pitch']['phase_tracking_error_rms_deg']:.1f}°\n"
        f"max axis tilt {summary['stability']['axis_tilt_max_deg']:.1f}°",
        transform=ax.transAxes,
        va="bottom",
    )

    ax = figure.add_subplot(grid[1, :])
    extent = (time[0], time[-1], len(JOINT_NAMES) - 0.5, -0.5)
    image = ax.imshow(
        torque.T,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-TORQUE_LIMIT_NM,
        vmax=TORQUE_LIMIT_NM,
        extent=extent,
    )
    ax.set_yticks(np.arange(len(JOINT_NAMES)), JOINT_NAMES)
    ax.set(xlabel="Time (s)", title="c  Actuator torque at 1 kHz")
    colorbar = figure.colorbar(image, ax=ax, location="right", shrink=0.92, pad=0.015)
    colorbar.set_label("Torque (N m)")

    ax = figure.add_subplot(grid[2, 0])
    x = np.arange(len(JOINT_NAMES))
    colors = [leg_colors[name.split("_")[0]] for name in JOINT_NAMES]
    ax.bar(x, rms, width=0.68, color=colors, alpha=0.82, label="RMS")
    ax.scatter(x, peak, s=13, facecolor="white", edgecolor="#202A44", linewidth=0.7, zorder=3, label="Peak |torque|")
    ax.axhline(TORQUE_LIMIT_NM, color="#B22222", linestyle=":", linewidth=1.0, label="3 N m limit")
    ax.set_xticks(x, JOINT_NAMES, rotation=62, ha="right")
    ax.set(ylabel="Torque (N m)", title="d  Joint-level torque load")
    style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = figure.add_subplot(grid[2, 1])
    self_collision_enabled = bool(summary.get("self_collision", {}).get("enabled", False))
    if self_collision_enabled:
        contact_count = values["self_contact_count"]
        penetration_mm = 1000.0 * values["self_penetration_max_m"]
        ax.fill_between(time, 0.0, contact_count, color="#D55E00", alpha=0.30, step="mid")
        ax.plot(time, contact_count, color="#D55E00", linewidth=0.8, label="Contact points")
        ax.set(
            xlabel="Time (s)",
            ylabel="Self-contact points",
            title="e  Self-collision onset and penetration",
        )
        style_axis(ax)
        ax_right = ax.twinx()
        ax_right.plot(time, penetration_mm, color="#0072B2", linewidth=0.8, label="Penetration")
        ax_right.set_ylabel("Maximum penetration (mm)", color="#0072B2")
        ax_right.tick_params(axis="y", colors="#0072B2", width=0.7, length=3)
        ax_right.spines["top"].set_visible(False)
        first_contact = next(
            (pair["first_time_s"] for pair in summary["self_collision"]["pairs"]),
            None,
        )
        if first_contact is not None:
            ax.axvline(first_contact, color="#202A44", linestyle=":", linewidth=1.0)
            ax.text(
                first_contact + 0.06,
                0.95,
                f"first contact {first_contact:.3f} s",
                transform=ax.get_xaxis_transform(),
                va="top",
                color="#202A44",
            )
    else:
        phase_mod = np.mod(phase, 2.0 * math.pi)
        bins = np.linspace(0.0, 2.0 * math.pi, 49)
        centers = 0.5 * (bins[:-1] + bins[1:])
        bin_index = np.clip(np.digitize(phase_mod, bins) - 1, 0, len(centers) - 1)
        for joint_index, name in enumerate(JOINT_NAMES):
            joint = name.split("_")[1]
            if joint == "abd":
                continue
            means = np.full(len(centers), np.nan)
            for index in range(len(centers)):
                mask = bin_index == index
                if np.any(mask):
                    means[index] = np.mean(torque[mask, joint_index])
            leg = name.split("_")[0]
            ax.plot(
                centers / (2.0 * math.pi),
                means,
                color=leg_colors[leg],
                linestyle=joint_styles[joint],
                linewidth=1.0,
                label=f"{leg} {joint}" if joint == "hip" else None,
            )
        ax.axhline(0.0, color="#8A94A6", linewidth=0.6)
        ax.set(xlim=(0, 1), xlabel="Rolling phase (cycle)", ylabel="Mean torque (N m)", title="e  Phase-locked torque pattern")
        style_axis(ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False, ncol=2, title="solid hip; dashed knee")

    figure.suptitle(
        "Rolling STUDENT — 5 s closed-loop simulation "
        f"({'self-collision ON' if self_collision_enabled else 'self-collision OFF'})",
        fontsize=11.0,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        -0.012,
        "Compact reset; 50 Hz policy; 1 kHz MuJoCo cg20; full first action; torque = actuator_force.",
        ha="center",
        fontsize=7.2,
        color="#4B5563",
    )

    stem = out_dir / "rolling_student_5s_analysis"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    print(stem.with_suffix(".png"))


def main() -> None:
    args = parse_args()
    if args.mode == "simulate":
        simulate(args.policy, args.model, args.out)
    else:
        plot(args.out)


if __name__ == "__main__":
    main()
