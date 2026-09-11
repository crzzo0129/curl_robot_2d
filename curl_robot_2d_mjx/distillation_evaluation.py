"""Host-side, per-episode command evaluation reports (no extra rollouts)."""

import csv
import json
import math

import numpy as np


def save_command_evaluation(out, *, forward_commands, yaw_commands,
                            command_changed, steps, warmup_steps, turns,
                            forward_error_sum, yaw_error_sum, failed,
                            failure_flags, episode_length, control_timestep,
                            speed_min, speed_max, minimum_turns, rolling_radius):
    """Retain legacy success and report a separate minimum-speed criterion.

    Group errors are transition-weighted MAE over active steps, including the
    termination transition. Variable-command episodes are reported separately;
    they must not be attributed to their initial command's speed/turn bin.
    """
    forward_commands, yaw_commands = map(np.asarray, (forward_commands, yaw_commands))
    steps, turns = map(np.asarray, (steps, turns))
    command_changed = np.asarray(command_changed, dtype=bool)
    failed = np.asarray(failed, dtype=bool)
    strict_failed = np.asarray(failure_flags["failure_lateral_drift"], dtype=bool) | failed
    forward_error_sum, yaw_error_sum = map(np.asarray, (forward_error_sum, yaw_error_sum))
    horizon_s = episode_length * control_timestep
    # Fixed for all commands, based on the configured lower speed bound, not
    # the minimum command that happened to appear in this random sample.
    minimum_speed_turns = speed_min * horizon_s / (2.0 * math.pi * rolling_radius)
    legacy_success = (~failed) & (turns >= minimum_turns)
    strict_success = (~strict_failed) & (turns >= minimum_turns)
    minimum_speed_success = ((~strict_failed) & (steps == episode_length)
                             & (turns >= minimum_speed_turns))
    rows = []
    for i in range(len(steps)):
        rows.append({
            "episode": i,
            "forward_command_m_s": float(forward_commands[i]),
            "yaw_command_rad_s": float(yaw_commands[i]),
            "command_changed": bool(command_changed[i]),
            "teacher_warmup_steps": int(warmup_steps[i]),
            "student_steps": int(steps[i]),
            "student_duration_s": float(steps[i] * control_timestep),
            "effective_turns": float(turns[i]),
            "forward_mae_m_s": float(forward_error_sum[i] / max(steps[i], 1)),
            "yaw_mae_rad_s": float(yaw_error_sum[i] / max(steps[i], 1)),
            "success": bool(legacy_success[i]),
            "strict_success": bool(strict_success[i]),
            "minimum_speed_success": bool(minimum_speed_success[i]),
            **{name: bool(value[i]) for name, value in failure_flags.items()},
        })

    def summarize(mask):
        count = int(np.sum(mask))
        if not count:
            return {"episodes": 0, "success_rate": None,
                    "strict_success_rate": None, "minimum_speed_success_rate": None,
                    "forward_mae_m_s": None, "yaw_mae_rad_s": None}
        samples = max(int(np.sum(steps[mask])), 1)
        return {
            "episodes": count,
            "success_count": int(np.sum(legacy_success[mask])),
            "success_rate": float(np.mean(legacy_success[mask])),
            "strict_success_rate": float(np.mean(strict_success[mask])),
            "minimum_speed_success_rate": float(np.mean(minimum_speed_success[mask])),
            "failure_free_rate": float(np.mean(~failed[mask])),
            "full_horizon_rate": float(np.mean(steps[mask] == episode_length)),
            "mean_effective_turns": float(np.mean(turns[mask])),
            "mean_student_duration_s": float(np.mean(steps[mask]) * control_timestep),
            "forward_mae_m_s": float(np.sum(forward_error_sum[mask]) / samples),
            "yaw_mae_rad_s": float(np.sum(yaw_error_sum[mask]) / samples),
            "failure_counts": {name: int(np.sum(np.asarray(value)[mask]))
                               for name, value in failure_flags.items()},
        }

    fixed = ~command_changed
    edges = np.linspace(speed_min, speed_max, 4)
    # For a fixed-speed run put all samples in the low bin, leaving others empty.
    speed_index = (np.zeros(len(steps), dtype=int) if speed_min == speed_max else
                   np.searchsorted(edges[1:-1], forward_commands, side="right"))
    speed_masks = {name: fixed & (speed_index == i)
                   for i, name in enumerate(("low", "medium", "high"))}
    turn_masks = {
        "straight": fixed & (np.abs(yaw_commands) <= 1e-3),
        "left_positive_yaw": fixed & (yaw_commands > 1e-3),
        "right_negative_yaw": fixed & (yaw_commands < -1e-3),
    }
    report = {
        "criteria": {
            "legacy_minimum_turns": minimum_turns,
            "minimum_speed_m_s": speed_min,
            "student_horizon_s": horizon_s,
            "rolling_radius_m": rolling_radius,
            "minimum_speed_required_effective_turns": minimum_speed_turns,
            "minimum_speed_success": "full horizon, no strict failure, effective turns >= v_min*T/(2*pi*R)",
            "note": "Supplemental criterion; legacy success unchanged. Effective progress uses world-x translation; no command tracking tolerance is imposed.",
        },
        "aggregation": "MAE weighted by active transitions; rates weighted by episodes; empty groups have null metrics",
        "speed_bin_edges_m_s": edges.tolist(),
        "speed_bin_boundary_rule": "[low, high), final bin includes maximum",
        "overall": summarize(np.ones(len(steps), dtype=bool)),
        "by_speed": {name: summarize(mask) for name, mask in speed_masks.items()},
        "by_turn": {name: summarize(mask) for name, mask in turn_masks.items()},
        "by_speed_and_turn": {f"{speed}/{turn}": summarize(sm & tm)
                              for speed, sm in speed_masks.items()
                              for turn, tm in turn_masks.items()},
        "variable_command_episodes": summarize(command_changed),
        "per_episode": rows,
    }
    with (out / "command_evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (out / "command_evaluation_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("[command evaluation] group / n / success / min-speed success / vx MAE / yaw MAE", flush=True)
    for name, result in {"overall": report["overall"], **report["by_speed"], **report["by_turn"]}.items():
        if result["episodes"]:
            print(f"  {name}: n={result['episodes']} "
                  f"success={result['success_rate']:.1%} "
                  f"min_speed_success={result['minimum_speed_success_rate']:.1%} "
                  f"vx_mae={result['forward_mae_m_s']:.4f} "
                  f"yaw_mae={result['yaw_mae_rad_s']:.4f}", flush=True)
        else:
            print(f"  {name}: n=0 (no samples)", flush=True)
    return report
