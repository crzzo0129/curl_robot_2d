"""Run one independent warm-started CEM search at each upper-half COM point."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.evaluate_fixed_policy_torso_com import (
    DEFAULT_CONTROLLER,
    DEFAULT_X_CENTER_MM,
    DEFAULT_Z_CENTER_MM,
    _draw_heatmaps,
    is_inside_upper_circle,
    root_coordinates_from_circle,
)
from scripts.optimize_phase_controller import (
    FOOT_GAP_TRACKING_MARGIN_M,
    _load_controller_parameters,
)
from scripts.sweep_torso_com_cem import run_variant, variant_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "per_point_cem_torso_com_upper_half"
DEFAULT_FIXED_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "fixed_policy_torso_com_upper_half"
    / "summary.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-center-mm", type=float, nargs="+", default=list(DEFAULT_X_CENTER_MM))
    parser.add_argument("--z-center-mm", type=float, nargs="+", default=list(DEFAULT_Z_CENTER_MM))
    parser.add_argument("--radius-limit-mm", type=float, default=140.0)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--final-duration", type=float, default=10.0)
    parser.add_argument("--barrier-generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-foot-gap-mm", type=float, default=2.0)
    parser.add_argument(
        "--foot-gap-tracking-margin-mm",
        type=float,
        default=1000.0 * FOOT_GAP_TRACKING_MARGIN_M,
    )
    parser.add_argument("--initial-controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Start CEM from its uniform parameter distribution without a controller.",
    )
    parser.add_argument("--fixed-results", type=Path, default=DEFAULT_FIXED_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore completed per-point result files and run every CEM again.",
    )
    return parser.parse_args(argv)


def upper_half_points(
    x_values_mm: list[float] | tuple[float, ...],
    z_values_mm: list[float] | tuple[float, ...],
    radius_limit_mm: float,
) -> list[tuple[float, float, float, float]]:
    points = []
    radius_limit_m = radius_limit_mm / 1000.0
    for z_mm in z_values_mm:
        for x_mm in x_values_mm:
            x_center_m = x_mm / 1000.0
            z_center_m = z_mm / 1000.0
            if not is_inside_upper_circle(x_center_m, z_center_m, radius_limit_m):
                continue
            x_root_m, z_root_m = root_coordinates_from_circle(x_center_m, z_center_m)
            points.append((x_center_m, z_center_m, x_root_m, z_root_m))
    return points


def _point_key(x_center_m: float, z_center_m: float) -> tuple[float, float]:
    return round(x_center_m, 9), round(z_center_m, 9)


def _load_fixed_lookup(path: Path) -> dict[tuple[float, float], dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        _point_key(
            float(row["torso_com_x_circle_m"]),
            float(row["torso_com_z_circle_m"]),
        ): row
        for row in rows
    }


def _add_context(
    result: dict,
    *,
    x_center_m: float,
    z_center_m: float,
    x_root_m: float,
    z_root_m: float,
    fixed_row: dict | None,
    args,
) -> dict:
    enriched = {
        "torso_com_x_circle_m": x_center_m,
        "torso_com_z_circle_m": z_center_m,
        "torso_com_x_root_m": x_root_m,
        "torso_com_z_root_m": z_root_m,
        **result,
        "cem_generations": args.generations,
        "cem_population": args.population,
        "cem_seed": args.seed,
        "cem_initialization": (
            "cold_start_uniform" if args.cold_start else "warm_start_controller"
        ),
    }
    if fixed_row is not None:
        fixed_turns = float(fixed_row["conservative_rolling_turns"])
        fixed_score = float(fixed_row["score"])
        fixed_contact = float(fixed_row["forbidden_contact_total_s"])
        enriched.update(
            {
                "fixed_policy_conservative_rolling_turns": fixed_turns,
                "fixed_policy_score": fixed_score,
                "fixed_policy_forbidden_contact_total_s": fixed_contact,
                "cem_turn_gain_over_fixed": float(
                    enriched["conservative_rolling_turns"]
                )
                - fixed_turns,
                "cem_score_gain_over_fixed": float(enriched["score"]) - fixed_score,
            }
        )
    return enriched


def _write_tables(output_dir: Path, results: list[dict]) -> tuple[Path, Path]:
    grid_rows = sorted(
        results,
        key=lambda row: (
            float(row["torso_com_z_circle_m"]),
            float(row["torso_com_x_circle_m"]),
        ),
    )
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)
    json_path.write_text(
        json.dumps(grid_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def run_sweep(args) -> list[dict]:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_parameters = (
        None
        if args.cold_start
        else _load_controller_parameters(args.initial_controller)
    )
    fixed_lookup = _load_fixed_lookup(args.fixed_results.expanduser().resolve())
    points = upper_half_points(
        args.x_center_mm,
        args.z_center_mm,
        args.radius_limit_mm,
    )
    results = []
    for index, (x_center_m, z_center_m, x_root_m, z_root_m) in enumerate(
        points, start=1
    ):
        variant = variant_name(x_root_m, z_root_m)
        result_path = args.output_dir / variant / "result.json"
        if result_path.exists() and not args.restart:
            enriched = json.loads(result_path.read_text(encoding="utf-8-sig"))
            status = "resume"
        else:
            print(
                f"point={index}/{len(points)} circle=({1000*x_center_m:+.1f},"
                f"{1000*z_center_m:+.1f})mm root=({1000*x_root_m:+.1f},"
                f"{1000*z_root_m:+.1f})mm",
                flush=True,
            )
            result = run_variant(args, x_root_m, z_root_m, initial_parameters)
            enriched = _add_context(
                result,
                x_center_m=x_center_m,
                z_center_m=z_center_m,
                x_root_m=x_root_m,
                z_root_m=z_root_m,
                fixed_row=fixed_lookup.get(_point_key(x_center_m, z_center_m)),
                args=args,
            )
            result_path.write_text(
                json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            status = "complete"
        results.append(enriched)
        _write_tables(args.output_dir, results)
        print(
            f"  {status} turns={float(enriched['conservative_rolling_turns']):.3f} "
            f"contact={float(enriched['forbidden_contact_total_s']):.3f}s "
            f"score={float(enriched['score']):.3f}",
            flush=True,
        )
    return results


def _write_report(path: Path, results: list[dict], args) -> None:
    best_turns = max(results, key=lambda row: float(row["conservative_rolling_turns"]))
    best_score = max(results, key=lambda row: float(row["score"]))
    low_contact = [
        row
        for row in results
        if float(row["forbidden_contact_total_s"]) <= 0.1
    ]
    best_low_contact = max(
        low_contact,
        key=lambda row: float(row["conservative_rolling_turns"]),
    )
    original = min(
        results,
        key=lambda row: abs(float(row["torso_com_x_root_m"]) - 0.025)
        + abs(float(row["torso_com_z_root_m"]) - 0.015),
    )
    ranked = sorted(
        results,
        key=lambda row: (
            float(row["conservative_rolling_turns"]),
            -float(row["forbidden_contact_total_s"]),
        ),
        reverse=True,
    )
    gains = [
        float(row["cem_turn_gain_over_fixed"])
        for row in results
        if "cem_turn_gain_over_fixed" in row
    ]
    lines = [
        "# 上半圆 Torso 质心逐点 CEM 报告",
        "",
        "## 实验定义",
        "",
        "- 每个质心点独立运行一次 CEM，不继承其他点的优化结果。",
        (
            f"- 初始化：{'不加载控制器，第一代从完整参数范围均匀采样' if args.cold_start else '从同一个已有控制器附近开始'}；"
            f"随机种子 `{args.seed}`。"
        ),
        f"- CEM 预算：{args.generations} 代，每代 {args.population} 个候选，优化 rollout {args.duration:g} s。",
        f"- 最终评估时长：{args.final_duration:g} s；有效点数：{len(results)}。",
        "",
        "## 核心结果",
        "",
        (
            f"原始质心 root=({1000*float(original['torso_com_x_root_m']):+.1f}, "
            f"{1000*float(original['torso_com_z_root_m']):+.1f}) mm："
            f"{float(original['conservative_rolling_turns']):.2f} 圈，"
            f"得分 {float(original['score']):.2f}。"
        ),
        (
            f"圈数最高点 root=({1000*float(best_turns['torso_com_x_root_m']):+.1f}, "
            f"{1000*float(best_turns['torso_com_z_root_m']):+.1f}) mm："
            f"{float(best_turns['conservative_rolling_turns']):.2f} 圈，"
            f"得分 {float(best_turns['score']):.2f}。"
        ),
        (
            f"综合得分最高点 root=({1000*float(best_score['torso_com_x_root_m']):+.1f}, "
            f"{1000*float(best_score['torso_com_z_root_m']):+.1f}) mm："
            f"{float(best_score['conservative_rolling_turns']):.2f} 圈，"
            f"得分 {float(best_score['score']):.2f}，"
            f"非法接触 {float(best_score['forbidden_contact_total_s']):.3f} s。"
        ),
    ]
    if args.cold_start:
        lines.extend(
            [
                (
                    f"在非法接触不超过 0.1 s 的点中，最高仅为 "
                    f"{float(best_low_contact['conservative_rolling_turns']):.2f} 圈；"
                    "本次短预算没有找到持续滚动且接触干净的 cold-start 策略。"
                ),
                "",
            ]
        )
    if gains and not args.cold_start:
        lines.extend(
            [
                (
                    f"与固定策略相比，{sum(gain > 0.05 for gain in gains)}/{len(gains)} "
                    f"个点的圈数提高超过 0.05 圈。"
                ),
                "",
            ]
        )
    if args.cold_start:
        lines.extend(
            [
                "## 圈数排名（前 10）",
                "",
                "| 排名 | root x (mm) | root z (mm) | CEM 圈数 | 位移 (m) | 非法接触 (s) | 得分 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "## 圈数排名（前 10）",
                "",
                "| 排名 | root x (mm) | root z (mm) | 固定策略圈数 | CEM 圈数 | 提升 | 非法接触 (s) | 得分 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for index, row in enumerate(ranked[:10], start=1):
        if args.cold_start:
            lines.append(
                f"| {index} | {1000*float(row['torso_com_x_root_m']):.1f} | "
                f"{1000*float(row['torso_com_z_root_m']):.2f} | "
                f"{float(row['conservative_rolling_turns']):.2f} | "
                f"{float(row['root_x_displacement_m']):.2f} | "
                f"{float(row['forbidden_contact_total_s']):.3f} | "
                f"{float(row['score']):.1f} |"
            )
        else:
            fixed_turns = row.get("fixed_policy_conservative_rolling_turns")
            gain = row.get("cem_turn_gain_over_fixed")
            lines.append(
                f"| {index} | {1000*float(row['torso_com_x_root_m']):.1f} | "
                f"{1000*float(row['torso_com_z_root_m']):.2f} | "
                f"{float(fixed_turns):.2f} | {float(row['conservative_rolling_turns']):.2f} | "
                f"{float(gain):+.2f} | {float(row['forbidden_contact_total_s']):.3f} | "
                f"{float(row['score']):.1f} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "每个点只有一次 CEM 随机搜索。本结果适合筛选候选区域，不足以区分接近候选点之间的小差异；最终候选应增加预算并使用多个随机种子复验。",
            "",
            "当前 CEM 使用 4 s 优化窗口，而本表使用 10 s 最终验证，因此原始 CEM 候选可能出现短时优化、长时失效。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def _rolling_selection_key(row: dict) -> tuple[bool, float, float, float]:
    return (
        bool(row["completed_two_turns"]),
        float(row["conservative_rolling_turns"]),
        -float(row["forbidden_contact_total_s"]),
        float(row["score"]),
    )


def build_best_available_results(
    cem_results: list[dict], fixed_lookup: dict[tuple[float, float], dict]
) -> list[dict]:
    selected_rows = []
    protected_keys = {
        "variant",
        "torso_com_x_circle_m",
        "torso_com_z_circle_m",
        "torso_com_x_root_m",
        "torso_com_z_root_m",
        "torso_com_x_m",
        "torso_com_z_m",
        "model_path",
    }
    for cem_row in cem_results:
        fixed_row = fixed_lookup[
            _point_key(
                float(cem_row["torso_com_x_circle_m"]),
                float(cem_row["torso_com_z_circle_m"]),
            )
        ]
        use_cem = _rolling_selection_key(cem_row) > _rolling_selection_key(fixed_row)
        selected = dict(cem_row)
        selected["raw_cem_controller_path"] = cem_row["controller_path"]
        selected["raw_cem_conservative_rolling_turns"] = cem_row[
            "conservative_rolling_turns"
        ]
        selected["raw_cem_score"] = cem_row["score"]
        selected["raw_cem_forbidden_contact_total_s"] = cem_row[
            "forbidden_contact_total_s"
        ]
        if use_cem:
            selected["selected_strategy_source"] = "per_point_cem"
            selected["selected_controller_path"] = cem_row["controller_path"]
        else:
            for key, value in fixed_row.items():
                if key in selected and key not in protected_keys:
                    selected[key] = value
            selected["selected_strategy_source"] = "fixed_reference_fallback"
            selected["selected_controller_path"] = str(DEFAULT_CONTROLLER.resolve())
        selected_rows.append(selected)
    return selected_rows


def _write_best_available_report(path: Path, results: list[dict], args) -> None:
    best_turns = max(results, key=lambda row: float(row["conservative_rolling_turns"]))
    best_score = max(results, key=lambda row: float(row["score"]))
    original = min(
        results,
        key=lambda row: abs(float(row["torso_com_x_root_m"]) - 0.025)
        + abs(float(row["torso_com_z_root_m"]) - 0.015),
    )
    cem_selected = sum(
        row["selected_strategy_source"] == "per_point_cem" for row in results
    )
    ranked = sorted(results, key=_rolling_selection_key, reverse=True)
    lines = [
        "# 逐点 CEM 后滚动策略择优报告",
        "",
        "## 选择规则",
        "",
        "每个质心都已经独立运行一次 CEM。最终在该点的 CEM 候选与原固定策略之间，依次按照完成两圈、保守圈数、较少非法接触和综合得分选择可用策略。原始 CEM 输出未被覆盖。",
        "",
        "## 核心结果",
        "",
        f"- 44 个点中，有 {cem_selected} 个点选择 CEM 新策略，其余使用原策略回退。",
        (
            f"- 原始质心 root=({1000*float(original['torso_com_x_root_m']):+.1f}, "
            f"{1000*float(original['torso_com_z_root_m']):+.1f}) mm："
            f"{float(original['conservative_rolling_turns']):.2f} 圈。"
        ),
        (
            f"- 圈数最高点 root=({1000*float(best_turns['torso_com_x_root_m']):+.1f}, "
            f"{1000*float(best_turns['torso_com_z_root_m']):+.1f}) mm："
            f"{float(best_turns['conservative_rolling_turns']):.2f} 圈，"
            f"来源 `{best_turns['selected_strategy_source']}`。"
        ),
        (
            f"- 综合得分最高点 root=({1000*float(best_score['torso_com_x_root_m']):+.1f}, "
            f"{1000*float(best_score['torso_com_z_root_m']):+.1f}) mm："
            f"得分 {float(best_score['score']):.2f}，"
            f"{float(best_score['conservative_rolling_turns']):.2f} 圈。"
        ),
        "",
        "## 圈数排名（前 10）",
        "",
        "| 排名 | root x (mm) | root z (mm) | 固定策略 | 原始 CEM | 最终选择 | 来源 | 非法接触 (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for index, row in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {index} | {1000*float(row['torso_com_x_root_m']):.1f} | "
            f"{1000*float(row['torso_com_z_root_m']):.2f} | "
            f"{float(row['fixed_policy_conservative_rolling_turns']):.2f} | "
            f"{float(row['raw_cem_conservative_rolling_turns']):.2f} | "
            f"{float(row['conservative_rolling_turns']):.2f} | "
            f"{row['selected_strategy_source']} | "
            f"{float(row['forbidden_contact_total_s']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"CEM 预算仍为单种子、{args.generations} 代、每代 {args.population} 个候选。择优表用于避免采用明显低于起点的策略，不等价于证明参考策略已经全局最优。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_final_outputs(args, results) -> tuple[Path, ...]:
    csv_path, json_path = _write_tables(args.output_dir, results)
    plot_path = args.output_dir / "per_point_cem_heatmaps.png"
    report_path = args.output_dir / "report_zh.md"
    _draw_heatmaps(
        plot_path,
        results,
        list(args.x_center_mm),
        list(args.z_center_mm),
        heading=(
            "Cold-start per-point CEM: torso COM design potential"
            if args.cold_start
            else "Independent per-point CEM: torso COM design potential"
        ),
        subtitle=(
            "One CEM search from a uniform parameter distribution per valid point. "
            "Circle-center coordinates (mm)."
            if args.cold_start
            else "One warm-started CEM search per valid point. Coordinates are "
            "relative to the rolling-circle center (mm)."
        ),
    )
    _write_report(report_path, results, args)
    ranked_path = args.output_dir / "summary_ranked.csv"
    ranked = sorted(
        results,
        key=lambda row: float(row["conservative_rolling_turns"]),
        reverse=True,
    )
    with ranked_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)
    if args.cold_start:
        return csv_path, json_path, ranked_path, plot_path, report_path
    fixed_lookup = _load_fixed_lookup(args.fixed_results.expanduser().resolve())
    selected = build_best_available_results(results, fixed_lookup)
    selected_csv = args.output_dir / "summary_best_available.csv"
    selected_json = args.output_dir / "summary_best_available.json"
    selected_plot = args.output_dir / "best_available_heatmaps.png"
    selected_report = args.output_dir / "report_best_available_zh.md"
    selected_grid = sorted(
        selected,
        key=lambda row: (
            float(row["torso_com_z_circle_m"]),
            float(row["torso_com_x_circle_m"]),
        ),
    )
    with selected_csv.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(selected_grid[0].keys()))
        writer.writeheader()
        writer.writerows(selected_grid)
    selected_json.write_text(
        json.dumps(selected_grid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _draw_heatmaps(
        selected_plot,
        selected_grid,
        list(args.x_center_mm),
        list(args.z_center_mm),
        heading="Best available policy after one per-point CEM search",
        subtitle=(
            "Rolling-first selection between the per-point CEM candidate and "
            "the fixed reference. Circle-center coordinates (mm)."
        ),
    )
    _write_best_available_report(selected_report, selected_grid, args)
    return (
        csv_path,
        json_path,
        ranked_path,
        plot_path,
        report_path,
        selected_csv,
        selected_json,
        selected_plot,
        selected_report,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.radius_limit_mm <= 0.0:
        raise SystemExit("--radius-limit-mm must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.minimum_foot_gap_mm < 0.0:
        raise SystemExit("--minimum-foot-gap-mm cannot be negative")
    results = run_sweep(args)
    if not results:
        raise SystemExit("No valid COM points")
    for output_path in write_final_outputs(args, results):
        print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
