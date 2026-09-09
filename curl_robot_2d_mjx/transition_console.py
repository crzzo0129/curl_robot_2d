"""Compact, dependency-free console reports for transition PPO."""

import math


REWARD_GROUPS = (
    ("pose/support", ("reference_tracking", "deploy_pose", "deploy_progress", "upright", "height", "support")),
    ("stand", ("stabilize", "stabilize_pose", "ready", "hold_joint_motion", "hold_body_motion", "hold_foot_slip", "deploy_instability")),
    ("smooth", ("action_rate", "target_rate", "target_acceleration", "action_magnitude", "joint_velocity", "foot_slip")),
    ("contact/end", ("touchdown_speed", "impact", "nonfoot_contact", "termination")),
    ("brake", ("brake_speed", "brake_progress", "brake_capture")),
)


def summarize_raw_evaluation(arrays):
    """Raw env metrics are sums even when named _per_step (no Brax wrapper)."""
    import numpy as np
    lengths = np.asarray(arrays.get("recorded_steps", arrays["terminal_step"]), dtype=float)
    if np.any(lengths <= 0):
        raise ValueError("raw evaluation requires positive recorded episode lengths")
    result = {}
    for name, values in arrays.items():
        if name in ("terminal_step", "recorded_steps"):
            continue
        values = np.asarray(values, dtype=float)
        if name.endswith("_per_step"):
            values = values / lengths
        result["eval/episode_" + name] = float(values.mean())
    result["eval/avg_episode_length"] = float(lengths.mean())
    return result


def format_transition_eval(step, metrics, *, stage, control_dt, source_report):
    def number(key):
        return float(metrics.get(key, math.nan))

    def rate(name):
        return f"{number('eval/episode_' + name):.1%}"

    length = number("eval/avg_episode_length")
    total = number("eval/episode_reward")
    lines = [
        f"\n[eval {int(step):,}] {stage} | success {rate('transition_success')}"
        f" | failure {rate('failed')} | timeout {rate('timeout')}",
        f"  episode {length:.1f} steps / {length * control_dt:.2f}s"
        f" | return {total:+.2f} | return/step {total / length if length > 0 else math.nan:+.4f}",
    ]
    def physical(name):
        key = f"eval/episode_{name}_per_step"
        return number(key) if key in metrics else (
            number(f"eval/episode_{name}") / length if length > 0 else math.nan)

    lines.append(
        f"  mean state: tilt {math.degrees(physical('upright_tilt_rad')):.2f}deg"
        f" | pose error {physical('stand_pose_error_rms_rad'):.3f}rad"
        f" | v {physical('linear_speed_m_s'):.3f}m/s"
        f" | omega {physical('angular_speed_rad_s'):.3f}rad/s")
    lines.append(
        f"  mean contact: feet {physical('foot_contact_count'):.2f}/4"
        f" | nonfoot {physical('nonfoot_contact_count'):.3f}"
        f" | height {physical('root_z_m'):.3f}m")
    cycles = source_report.get("by_cycle", {})
    visited = [str(k) for k, row in cycles.items() if row.get("episodes", 0) > 0]
    lines.append(f"  source coverage: {len(visited)}/{len(cycles)} groups"
                 f" | seen {','.join(visited) or '-'}")
    lines.append("  reward terms (mean episode SUM; zero terms hidden):")
    for group, names in REWARD_GROUPS:
        cells = [f"{name}={number('eval/episode_reward_' + name):+.3f}"
                 for name in names if 'eval/episode_reward_' + name in metrics
                 and (not math.isfinite(number('eval/episode_reward_' + name))
                      or abs(number('eval/episode_reward_' + name)) > 1e-8)]
        for offset in range(0, len(cells), 3):
            label = group if offset == 0 else ""
            lines.append(f"    {label:12s} " + " | ".join(cells[offset:offset + 3]))
    causes = [f"{k.removeprefix('eval/episode_failure_')}={float(v):.1%}"
              for k, v in metrics.items()
              if k.startswith("eval/episode_failure_") and not k.endswith("_std")
              and "_mode_" not in k and float(v) > 0]
    if causes:
        lines.append("  failures: " + ", ".join(causes))
    if "training/learning_rate" in metrics:
        lines.append(f"  PPO: lr {number('training/learning_rate'):.2g}"
                     f" | KL {number('training/kl_mean'):.4f}"
                     f" | policy loss {number('training/policy_loss'):+.4f}"
                     f" | value loss {number('training/v_loss'):.3f}"
                     f" | train {number('training/sps'):.0f} steps/s")
    return "\n".join(lines)
