"""Dependency-light sampling and reporting for rolling takeover experiments.

Candidates are augmented simulator/controller states, NOT stand-to-roll
demonstrations. A successful continuation does not establish reachability
from stand, nor certify a deployment gate.
"""

from dataclasses import dataclass
import math

import numpy as np


PROBE_CASES = ("exact", "state_noise", "phase_noise", "history_noise", "combined")
FAILURES = (
    "nonfinite", "root_low", "root_high", "lateral_drift", "axis_tilt",
    "forbidden_depth", "forbidden_contact",
)


@dataclass(frozen=True)
class HandoffNoise:
    joint_position_rad: float = 0.01
    joint_velocity_rad_s: float = 0.10
    root_linear_velocity_m_s: float = 0.02
    root_angular_velocity_rad_s: float = 0.10
    axis_rotation_rad: float = 0.02
    oscillator_phase_rad: float = 0.20
    previous_action_normalized: float = 0.05

    def validate(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def sampling_steps(window_s, interval_s, control_dt):
    """Always include t=0 and the end; reject ambiguous off-grid timings."""
    for name, value in (("window_s", window_s), ("interval_s", interval_s),
                        ("control_dt", control_dt)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    end, stride = round(window_s / control_dt), round(interval_s / control_dt)
    if stride < 1 or end < 1 or not (
        np.isclose(end * control_dt, window_s, atol=1e-8, rtol=0)
        and np.isclose(stride * control_dt, interval_s, atol=1e-8, rtol=0)
    ):
        raise ValueError("window and sample interval must be exact control-step multiples")
    return tuple(sorted(set(range(0, end + 1, stride)) | {end}))


def perturbation_batch(case, noise, seed, count):
    """Independent uniform offsets; exact replay is never silently disturbed."""
    if case not in PROBE_CASES:
        raise ValueError(f"unknown probe case {case}")
    noise.validate()
    rng = np.random.default_rng(seed)
    state_scale = float(case in ("state_noise", "combined"))
    phase_scale = float(case in ("phase_noise", "combined"))
    history_scale = float(case in ("history_noise", "combined"))
    def draw(width, scale):
        return rng.uniform(-scale, scale, (count, width)).astype(np.float32)
    return {
        "dq": draw(8, state_scale * noise.joint_position_rad),
        "dqd": draw(8, state_scale * noise.joint_velocity_rad_s),
        "dv": np.concatenate((draw(3, state_scale * noise.root_linear_velocity_m_s),
                              draw(3, state_scale * noise.root_angular_velocity_rad_s)), axis=1),
        "daxis": draw(2, state_scale * noise.axis_rotation_rad),
        "dphase": draw(1, phase_scale * noise.oscillator_phase_rad)[:, 0],
        "dhistory": draw(8, history_scale * noise.previous_action_normalized),
    }


def blank_failures():
    return {name: False for name in FAILURES}


def continuation_rows(start, end, *, source_ids, source_success, case, sample_step,
                      dt, horizon_s, minimum_turn_rate, maxima, exact_expected=None):
    """Progress is measured AFTER takeover; lateral coordinates stay global."""
    rows = []
    for i, source_id in enumerate(source_ids):
        elapsed = float(end["time"][i] - start["time"][i])
        translation = float((end["qpos"][i, 0] - start["qpos"][i, 0])
                            / (2 * np.pi * float(start["radius"][i])))
        signed_rotation = float((end["rolling_phase"][i] - start["rolling_phase"][i]) / (2 * np.pi))
        absolute_rotation = float((end["absolute_rotation"][i] - start["absolute_rotation"][i]) / (2 * np.pi))
        conservative = min(absolute_rotation, translation)
        full_horizon = elapsed >= horizon_s - dt * 0.1
        failed = bool(end["failed"][i])
        enough_turns = (conservative >= minimum_turn_rate * horizon_s and signed_rotation > 0)
        row = {
            "source_id": int(source_id), "source_success": bool(source_success[int(source_id)]),
            "sample_step": int(sample_step), "sample_time_s": sample_step * dt,
            "case": case, "trial_index": i,
            "success": bool(full_horizon and not failed and enough_turns),
            "full_horizon": bool(full_horizon), "enough_turns": bool(enough_turns),
            "continued_s": elapsed, "turns_after_handoff": conservative,
            "signed_rotation_turns_after_handoff": signed_rotation,
            "translation_turns_after_handoff": translation,
            "start_y_m": float(start["y"][i]), "end_y_m": float(end["y"][i]),
            "start_vy_m_s": float(start["qvel"][i, 1]),
            "start_heading_rad": float(start["heading"][i]),
            "end_heading_rad": float(end["heading"][i]),
            "start_axis_tilt_rad": float(start["axis_tilt"][i]),
            "start_root_z_m": float(start["qpos"][i, 2]),
            "start_oscillator_phase_rad": float(start["oscillator_phase"][i]),
            "start_rolling_phase_rad": float(start["rolling_phase"][i]),
            "start_contact_penetration_m": float(start["penetration"][i]),
            "max_abs_y_m": float(maxima["y"][i]),
            "max_axis_tilt_rad": float(maxima["axis_tilt"][i]),
            "max_control_sample_torque_nm": float(maxima["torque"][i]),
            "first_command_jump_rad": float(maxima["first_command_jump"][i]),
            **{f"failure_{name}": bool(end[f"failure_{name}"][i]) for name in FAILURES},
        }
        if exact_expected is not None:
            # Full integration snapshot replay must agree with the uninterrupted
            # source, including when the source terminates before the horizon.
            row["exact_replay_qpos_max_error"] = float(np.max(np.abs(
                end["qpos"][i] - exact_expected["qpos"][int(source_id)])))
            row["exact_replay_qvel_max_error"] = float(np.max(np.abs(
                end["qvel"][i] - exact_expected["qvel"][int(source_id)])))
        rows.append(row)
    return rows


def summarize_probes(rows):
    """Keep denominators explicit; never certify a basin from these samples."""
    groups = []
    for step, case in sorted({(r["sample_step"], r["case"]) for r in rows}):
        selected = [r for r in rows if r["sample_step"] == step and r["case"] == case]
        qualified = [r for r in selected if r["source_success"]]
        failure_free = [r["full_horizon"] and not any(r[f"failure_{name}"] for name in FAILURES)
                        for r in selected]
        groups.append({
            "sample_time_s": selected[0]["sample_time_s"], "case": case,
            "trials": len(selected), "source_count": len({r["source_id"] for r in selected}),
            "successes": sum(r["success"] for r in selected),
            "success_rate": float(np.mean([r["success"] for r in selected])),
            "failure_free_rate": float(np.mean(failure_free)),
            "slow_but_failure_free_rate": float(np.mean([
                free and not r["enough_turns"] for free, r in zip(failure_free, selected)])),
            "qualified_source_trials": len(qualified),
            "qualified_source_success_rate": (
                float(np.mean([r["success"] for r in qualified])) if qualified else None),
            "mean_turns_after_handoff": float(np.mean([r["turns_after_handoff"] for r in selected])),
            "minimum_turns_after_handoff": min(r["turns_after_handoff"] for r in selected),
            "max_abs_y_m": max(r["max_abs_y_m"] for r in selected),
            "failure_rates": {name: float(np.mean([r[f"failure_{name}"] for r in selected]))
                              for name in FAILURES},
        })
    return groups
