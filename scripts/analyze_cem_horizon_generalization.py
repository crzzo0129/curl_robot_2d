"""Compare per-point CEM and reference controllers at the CEM time horizon."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco

from scripts.evaluate_fixed_policy_torso_com import (
    DEFAULT_CONTROLLER,
    _controller_settings,
)
from scripts.optimize_phase_controller import (
    _load_controller_parameters,
    rollout_controller,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CEM_DIR = PROJECT_ROOT / "results" / "per_point_cem_torso_com_upper_half"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cem-dir", type=Path, default=DEFAULT_CEM_DIR)
    parser.add_argument("--reference-controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--duration", type=float, default=4.0)
    return parser.parse_args(argv)


def _rollout(model_path: Path, controller_path: Path, duration: float):
    parameters = _load_controller_parameters(controller_path)
    minimum_gap_m, tracking_margin_m = _controller_settings(controller_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return rollout_controller(
        model,
        parameters[:8],
        duration=duration,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        minimum_foot_surface_gap_m=minimum_gap_m,
        foot_gap_tracking_margin_m=tracking_margin_m,
        detailed=False,
    )


def _metrics(prefix: str, rollout, duration: float) -> dict[str, float | bool]:
    summary = rollout.summary
    turns = float(summary["conservative_rolling_turns"])
    return {
        f"{prefix}_score": float(rollout.score),
        f"{prefix}_conservative_rolling_turns": turns,
        f"{prefix}_turn_rate_per_s": turns / duration,
        f"{prefix}_forbidden_contact_total_s": float(
            summary["forbidden_contact_total_s"]
        ),
        f"{prefix}_completed_two_turns": bool(summary["completed_two_turns"]),
    }


def evaluate(args) -> list[dict]:
    cem_dir = args.cem_dir.expanduser().resolve()
    source_rows = json.loads(
        (cem_dir / "summary.json").read_text(encoding="utf-8-sig")
    )
    reference_path = args.reference_controller.expanduser().resolve()
    rows = []
    for index, source in enumerate(source_rows, start=1):
        model_path = Path(source["model_path"])
        cem_path = Path(source["controller_path"])
        fixed_rollout = _rollout(model_path, reference_path, args.duration)
        cem_rollout = _rollout(model_path, cem_path, args.duration)
        row = {
            "torso_com_x_circle_m": float(source["torso_com_x_circle_m"]),
            "torso_com_z_circle_m": float(source["torso_com_z_circle_m"]),
            "torso_com_x_root_m": float(source["torso_com_x_root_m"]),
            "torso_com_z_root_m": float(source["torso_com_z_root_m"]),
            "model_path": str(model_path.resolve()),
            "cem_controller_path": str(cem_path.resolve()),
            "reference_controller_path": str(reference_path),
            **_metrics("fixed_4s", fixed_rollout, args.duration),
            **_metrics("cem_4s", cem_rollout, args.duration),
            "fixed_10s_score": float(source["fixed_policy_score"]),
            "fixed_10s_conservative_rolling_turns": float(
                source["fixed_policy_conservative_rolling_turns"]
            ),
            "cem_10s_score": float(source["score"]),
            "cem_10s_conservative_rolling_turns": float(
                source["conservative_rolling_turns"]
            ),
        }
        row["cem_score_gain_at_4s"] = float(row["cem_4s_score"]) - float(
            row["fixed_4s_score"]
        )
        row["cem_turn_gain_at_4s"] = float(
            row["cem_4s_conservative_rolling_turns"]
        ) - float(row["fixed_4s_conservative_rolling_turns"])
        row["cem_score_gain_at_10s"] = float(row["cem_10s_score"]) - float(
            row["fixed_10s_score"]
        )
        row["cem_turn_gain_at_10s"] = float(
            row["cem_10s_conservative_rolling_turns"]
        ) - float(row["fixed_10s_conservative_rolling_turns"])
        rows.append(row)
        print(
            f"point={index}/{len(source_rows)} "
            f"cem_score_gain_4s={float(row['cem_score_gain_at_4s']):+.3f} "
            f"cem_turn_gain_10s={float(row['cem_turn_gain_at_10s']):+.3f}",
            flush=True,
        )
    return rows


def write_outputs(args, rows: list[dict]) -> tuple[Path, Path, Path]:
    cem_dir = args.cem_dir.expanduser().resolve()
    csv_path = cem_dir / "horizon_comparison.csv"
    json_path = cem_dir / "horizon_comparison.json"
    report_path = cem_dir / "horizon_comparison_zh.md"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    score_better_4s = sum(float(row["cem_score_gain_at_4s"]) > 0.0 for row in rows)
    turns_better_4s = sum(float(row["cem_turn_gain_at_4s"]) > 0.0 for row in rows)
    score_better_10s = sum(float(row["cem_score_gain_at_10s"]) > 0.0 for row in rows)
    turns_better_10s = sum(float(row["cem_turn_gain_at_10s"]) > 0.0 for row in rows)
    short_better_long_worse = sum(
        float(row["cem_score_gain_at_4s"]) > 0.0
        and float(row["cem_turn_gain_at_10s"]) < 0.0
        for row in rows
    )
    lines = [
        "# CEM 优化时长与长时验证对照",
        "",
        "## 结果",
        "",
        f"- 在 4 s 优化窗口，CEM 策略有 {score_better_4s}/{len(rows)} 个点得分高于原策略，{turns_better_4s}/{len(rows)} 个点圈数更高。",
        f"- 在 10 s 最终验证，CEM 策略有 {score_better_10s}/{len(rows)} 个点得分更高，{turns_better_10s}/{len(rows)} 个点圈数更高。",
        f"- 有 {short_better_long_worse}/{len(rows)} 个点在 4 s 得分更高，但 10 s 圈数低于原策略。",
        "",
        "## 解释",
        "",
        "当前逐点 CEM 结果同时受到两项限制：优化窗口只有 4 s，不能直接保证 10 s 持续滚动；两阶段 CEM 从避碰目标切换到持续目标时，也不会持续保留初始参考策略。因而原始 CEM 输出应作为一次短预算搜索结果，而不是每个质心的可靠最优控制器。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return csv_path, json_path, report_path


def main(argv=None):
    args = parse_args(argv)
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    rows = evaluate(args)
    for path in write_outputs(args, rows):
        print(f"output={path.resolve()}")


if __name__ == "__main__":
    main()
