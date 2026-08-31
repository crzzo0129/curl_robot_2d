"""Signed lateral and same-state imitation diagnostics for rolling policies.

All trace arrays have time and environment as their first two dimensions.
Only active transitions enter summaries; frozen post-termination states must
not bias statistics. Teacher commands are labels queried at student states,
not a separate teacher rollout and never actions applied to the live rollout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PAIR_CHANNELS = ("front_hip", "front_knee", "rear_hip", "rear_knee")


def common_differential_action_3d(xp, effective_action):
    """Return (L+R)/2 and (L-R)/2 for FL,FR,RL,RR hip/knee actions."""

    left = xp.take(effective_action, xp.asarray((0, 1, 4, 5)), axis=-1)
    right = xp.take(effective_action, xp.asarray((2, 3, 6, 7)), axis=-1)
    return 0.5 * (left + right), 0.5 * (left - right)


def lateral_state_features_3d(xp, qpos, qvel, torso_rotation, initial_y):
    """Match environment_3d's world-frame drift and rolling-axis heading."""

    return {
        "y_m": qpos[..., 1] - initial_y,
        "vy_m_s": qvel[..., 1],
        "heading_rad": xp.arctan2(
            -torso_rotation[..., 0, 1], torso_rotation[..., 1, 1]
        ),
    }


def _error_statistics(error, mask):
    selected = np.asarray(error)[mask]
    if not selected.size:
        return {"samples": 0, "rmse": None, "bias_by_channel": None,
                "rmse_by_channel": None}
    return {
        "samples": int(len(selected)),
        "rmse": float(np.sqrt(np.mean(selected ** 2))),
        "bias_by_channel": dict(zip(PAIR_CHANNELS, np.mean(selected, axis=0).tolist())),
        "rmse_by_channel": dict(zip(
            PAIR_CHANNELS, np.sqrt(np.mean(selected ** 2, axis=0)).tolist()
        )),
    }


def _signal_statistics(signal, mask):
    selected = np.asarray(signal)[mask]
    if not selected.size:
        return {"mean_by_channel": None, "rms_by_channel": None}
    return {
        "mean_by_channel": dict(zip(PAIR_CHANNELS, np.mean(selected, axis=0).tolist())),
        "rms_by_channel": dict(zip(
            PAIR_CHANNELS, np.sqrt(np.mean(selected ** 2, axis=0)).tolist()
        )),
    }


def summarize_lateral_trace(trace):
    """Return JSON summary, one row per episode, and active-only time rows."""

    active = np.asarray(trace["active"], dtype=bool)
    if active.ndim != 2 or not np.all(np.any(active, axis=0)):
        raise ValueError("trace must contain active transitions for every episode")
    student = np.asarray(trace["student_action"])
    teacher = np.asarray(trace["teacher_action"])
    if student.shape != active.shape + (8,) or teacher.shape != student.shape:
        raise ValueError("student/teacher actions must have shape (time, env, 8)")
    valid = (
        active & np.asarray(trace["teacher_label_valid"], dtype=bool)
        & np.all(np.isfinite(student), axis=-1)
        & np.all(np.isfinite(teacher), axis=-1)
    )
    student_common, student_diff = common_differential_action_3d(np, student)
    teacher_common, teacher_diff = common_differential_action_3d(np, teacher)
    common_error = student_common - teacher_common
    differential_error = student_diff - teacher_diff

    episode_rows = []
    for env in range(active.shape[1]):
        live = active[:, env]
        labelled = valid[:, env]
        last = np.flatnonzero(live)[-1]
        row = {
            "env_id": env,
            "steps": int(live.sum()),
            "terminal_time_s": float(trace["next_time_s"][last, env]),
            "failed": bool(np.any(trace["failed"][:, env][live])),
            "lateral_failed": bool(np.any(trace["lateral_failed"][:, env][live])),
            "turns": float(trace["turns"][last, env]),
            "final_y_m": float(trace["next_y_m"][last, env]),
            "final_vy_m_s": float(trace["next_vy_m_s"][last, env]),
            "final_heading_rad": float(trace["next_heading_rad"][last, env]),
            "final_heading_deg": float(np.rad2deg(trace["next_heading_rad"][last, env])),
            "max_abs_y_m": float(np.max(np.abs(trace["next_y_m"][:, env][live]))),
            "mean_vy_m_s": float(np.mean(trace["vy_m_s"][:, env][live])),
        }
        for threshold in (0.05, 0.10, 0.15):
            crossings = np.flatnonzero(live & (np.abs(trace["next_y_m"][:, env]) >= threshold))
            row[f"first_abs_y_{round(threshold * 100):02d}cm_s"] = (
                float(trace["next_time_s"][crossings[0], env]) if crossings.size else None
            )
        for name, error in (("common", common_error), ("differential", differential_error)):
            stats = _error_statistics(error[:, env], labelled)
            row[f"{name}_error_rmse"] = stats["rmse"]
            for channel in PAIR_CHANNELS:
                row[f"{name}_error_bias_{channel}"] = (
                    stats["bias_by_channel"][channel] if stats["samples"] else None
                )
        episode_rows.append(row)

    lateral_failure = np.asarray([row["lateral_failed"] for row in episode_rows])
    final_y = np.asarray([row["final_y_m"] for row in episode_rows])
    failed = np.asarray([row["failed"] for row in episode_rows])
    groups = {}
    for name, selected in (
        ("failure_free", ~failed), ("lateral_failed", lateral_failure),
        ("lateral_positive", lateral_failure & (final_y > 0)),
        ("lateral_negative", lateral_failure & (final_y < 0)),
    ):
        groups[name] = {
            "episodes": int(selected.sum()),
            "common_error": _error_statistics(common_error, valid & selected[None, :]),
            "differential_error": _error_statistics(differential_error, valid & selected[None, :]),
        }

    time_rows = []
    for step in range(active.shape[0]):
        live = active[step]
        if not np.any(live):
            continue
        row = {
            "step": step,
            "time_s": float(np.mean(trace["time_s"][step, live])),
            "active_episodes": int(live.sum()),
            "valid_teacher_labels": int(valid[step].sum()),
        }
        for field in ("y_m", "vy_m_s", "heading_rad"):
            values = trace[field][step, live]
            row[f"{field}_mean"] = float(
                np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values)))
                if field == "heading_rad" else np.mean(values)
            )
            row[f"{field}_mean_abs"] = float(np.mean(np.abs(values)))
            row[f"{field}_p05"] = float(np.quantile(values, 0.05))
            row[f"{field}_p95"] = float(np.quantile(values, 0.95))
        for name, error in (("common", common_error), ("differential", differential_error)):
            stats = _error_statistics(error[step], valid[step])
            row[f"{name}_error_rmse"] = stats["rmse"]
            for channel in PAIR_CHANNELS:
                row[f"{name}_error_bias_{channel}"] = (
                    stats["bias_by_channel"][channel] if stats["samples"] else None
                )
        time_rows.append(row)

    summary = {
        "episodes": int(active.shape[1]),
        "active_transitions": int(active.sum()),
        "valid_teacher_labels": int(valid.sum()),
        "lateral_failure_positive_count": int(np.sum(lateral_failure & (final_y > 0))),
        "lateral_failure_negative_count": int(np.sum(lateral_failure & (final_y < 0))),
        "final_y_mean_m": float(np.mean(final_y)),
        "final_y_mean_abs_m": float(np.mean(np.abs(final_y))),
        "common_error": _error_statistics(common_error, valid),
        "differential_error": _error_statistics(differential_error, valid),
        "teacher_differential_action": _signal_statistics(teacher_diff, valid),
        "student_differential_action": _signal_statistics(student_diff, valid),
        "groups": groups,
        "definitions": {
            "axes": "world frame; y is relative to episode initial root y",
            "heading": "atan2(-body_y_axis.x, body_y_axis.y), radians; same as environment",
            "common_action": "(left + right) / 2",
            "differential_action": "(left - right) / 2",
            "action_error": "student minus teacher, normalized action units before joint-limit clipping",
            "pair_channels": list(PAIR_CHANNELS),
            "effective_action_order": ["FL_hip", "FL_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee", "RR_hip", "RR_knee"],
            "time_alignment": "y/vy/heading and both actions share the pre-step student state; next_* are post-student-step",
            "teacher_label": "last-substep effective action from teacher_env.step on a copy of the same student state, matching DAgger targets",
            "termination": "active includes the terminal transition; later frozen states excluded",
            "time_rows": "active-only statistics; active_episodes reports changing survivor count",
        },
    }
    return summary, episode_rows, time_rows


def save_lateral_trace(output_dir, trace):
    """Write raw NPZ plus compact JSON and CSV reports, without plotting."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = {name: np.asarray(value) for name, value in trace.items()}
    summary, episode_rows, time_rows = summarize_lateral_trace(trace)
    for policy in ("student", "teacher"):
        common, differential = common_differential_action_3d(np, trace[f"{policy}_action"])
        trace[f"{policy}_common_action"] = common
        trace[f"{policy}_differential_action"] = differential
    for component in ("common", "differential"):
        trace[f"{component}_error"] = (
            trace[f"student_{component}_action"] - trace[f"teacher_{component}_action"]
        )
    np.savez_compressed(output_dir / "lateral_trace.npz", **trace)
    for name, rows in (("lateral_episodes.csv", episode_rows), ("lateral_timeseries.csv", time_rows)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "lateral_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary
