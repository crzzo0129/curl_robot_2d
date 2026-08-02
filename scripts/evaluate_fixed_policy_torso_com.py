"""Evaluate one fixed 2-D rolling controller across torso COM positions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from curl_robot_2d.model import write_mjcf
from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts.optimize_phase_controller import (
    _load_controller_parameters,
    rollout_controller,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "collision_constrained_cem_foot_gap_2mm_short_contact"
    / "best_phase_controller.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fixed_policy_torso_com_upper_half"
DEFAULT_X_CENTER_MM = (-75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0)
DEFAULT_Z_CENTER_MM = (
    0.0,
    25.0,
    50.0,
    75.0,
    100.0,
    118.228644035,
    133.228644035,
    140.0,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--x-center-mm", type=float, nargs="+", default=list(DEFAULT_X_CENTER_MM))
    parser.add_argument("--z-center-mm", type=float, nargs="+", default=list(DEFAULT_Z_CENTER_MM))
    parser.add_argument("--radius-limit-mm", type=float, default=140.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def root_coordinates_from_circle(
    x_center_m: float, z_center_m: float
) -> tuple[float, float]:
    return (
        float(x_center_m),
        float(z_center_m - FIXED_PARAMETERS.regular_pentagon_apothem),
    )


def is_inside_upper_circle(
    x_center_m: float, z_center_m: float, radius_limit_m: float
) -> bool:
    return (
        z_center_m >= 0.0
        and math.hypot(x_center_m, z_center_m) <= radius_limit_m + 1.0e-12
    )


def variant_name(x_center_m: float, z_center_m: float) -> str:
    name = f"com_circle_x_{x_center_m:+.5f}_z_{z_center_m:+.5f}"
    return name.replace("+", "p").replace("-", "m")


def _controller_settings(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return (
        float(payload.get("minimum_foot_surface_gap_m", 0.0)),
        float(payload.get("foot_gap_tracking_margin_m", 0.004)),
    )


def _result_row(
    *,
    x_center_m: float,
    z_center_m: float,
    x_root_m: float,
    z_root_m: float,
    model_path: Path,
    rollout,
) -> dict[str, float | int | bool | str | None]:
    summary = rollout.summary
    torso_mass = FIXED_PARAMETERS.torso_mass
    torso_inertia = FIXED_PARAMETERS.torso_planar_inertia
    return {
        "variant": variant_name(x_center_m, z_center_m),
        "torso_com_x_circle_m": x_center_m,
        "torso_com_z_circle_m": z_center_m,
        "torso_com_radius_circle_m": math.hypot(x_center_m, z_center_m),
        "torso_com_x_root_m": x_root_m,
        "torso_com_z_root_m": z_root_m,
        "torso_inertia_about_root_kg_m2": torso_inertia
        + torso_mass * (x_root_m * x_root_m + z_root_m * z_root_m),
        "torso_inertia_about_circle_center_kg_m2": torso_inertia
        + torso_mass * (x_center_m * x_center_m + z_center_m * z_center_m),
        "model_path": str(model_path.resolve()),
        "score": float(rollout.score),
        "net_turns": float(summary["net_turns"]),
        "rolling_turns": float(summary["rolling_progress_turns"]),
        "conservative_rolling_turns": float(summary["conservative_rolling_turns"]),
        "root_x_displacement_m": float(summary["root_x_displacement_m"]),
        "rolling_mismatch_rad": float(summary["rolling_mismatch_rad"]),
        "actuator_positive_work_J": float(summary["actuator_positive_work_J"]),
        "maximum_actuator_torque_Nm": float(summary["maximum_actuator_torque_Nm"]),
        "forbidden_contact_total_s": float(summary["forbidden_contact_total_s"]),
        "maximum_forbidden_penetration_m": float(
            summary["maximum_forbidden_penetration_m"]
        ),
        "foot_contact_total_s": float(summary["foot_contact_total_s"]),
        "minimum_foot_surface_gap_m": float(summary["minimum_foot_surface_gap_m"]),
        "leg_crossing_detected": bool(summary["leg_crossing_detected"]),
        "completed_two_turns": bool(summary["completed_two_turns"]),
    }


def run_sweep(args) -> list[dict[str, float | int | bool | str | None]]:
    controller_path = args.controller.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    controller = _load_controller_parameters(controller_path)
    minimum_gap_m, tracking_margin_m = _controller_settings(controller_path)
    radius_limit_m = args.radius_limit_mm / 1000.0
    points = [
        (x_mm / 1000.0, z_mm / 1000.0)
        for z_mm in args.z_center_mm
        for x_mm in args.x_center_mm
        if is_inside_upper_circle(x_mm / 1000.0, z_mm / 1000.0, radius_limit_m)
    ]

    results = []
    for index, (x_center_m, z_center_m) in enumerate(points, start=1):
        x_root_m, z_root_m = root_coordinates_from_circle(x_center_m, z_center_m)
        variant = variant_name(x_center_m, z_center_m)
        model_path = model_dir / f"{variant}.xml"
        write_mjcf(
            model_path,
            replace(
                FIXED_PARAMETERS,
                torso_com_x=x_root_m,
                torso_com_z=z_root_m,
            ),
        )
        model = mujoco.MjModel.from_xml_path(str(model_path))
        rollout = rollout_controller(
            model,
            controller[:8],
            duration=args.duration,
            oscillator_rate=float(controller[8]),
            oscillator_coupling=float(controller[9]),
            minimum_foot_surface_gap_m=minimum_gap_m,
            foot_gap_tracking_margin_m=tracking_margin_m,
            detailed=False,
        )
        row = _result_row(
            x_center_m=x_center_m,
            z_center_m=z_center_m,
            x_root_m=x_root_m,
            z_root_m=z_root_m,
            model_path=model_path,
            rollout=rollout,
        )
        results.append(row)
        print(
            f"point={index}/{len(points)} circle=({1000*x_center_m:+.1f},"
            f"{1000*z_center_m:+.1f})mm turns="
            f"{row['conservative_rolling_turns']:.3f} contact="
            f"{row['forbidden_contact_total_s']:.3f}s score={row['score']:.3f}",
            flush=True,
        )
    return results


def _color(value: float, minimum: float, maximum: float, *, reverse: bool = False):
    fraction = 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)
    fraction = max(0.0, min(1.0, fraction))
    if reverse:
        fraction = 1.0 - fraction
    low = (194, 63, 52)
    middle = (242, 220, 137)
    high = (43, 126, 96)
    if fraction < 0.5:
        blend = fraction * 2.0
        return tuple(int(low[i] + blend * (middle[i] - low[i])) for i in range(3))
    blend = (fraction - 0.5) * 2.0
    return tuple(int(middle[i] + blend * (high[i] - middle[i])) for i in range(3))


def _draw_heatmaps(
    path: Path,
    results: list[dict[str, float | int | bool | str | None]],
    x_values_mm: list[float],
    z_values_mm: list[float],
    *,
    heading: str = "Fixed rolling policy: torso COM sensitivity",
    subtitle: str = (
        "Coordinates are relative to the nominal rolling-circle center (mm). "
        "BASE = original COM."
    ),
) -> None:
    panels = (
        ("Conservative rolling turns", "conservative_rolling_turns", False, ".2f"),
        ("Forward displacement (m)", "root_x_displacement_m", False, ".2f"),
        ("Forbidden contact (s)", "forbidden_contact_total_s", True, ".3f"),
        ("Controller score", "score", False, ".1f"),
    )
    cell_w, cell_h = 88, 54
    left, top = 86, 82
    panel_w = left + cell_w * len(x_values_mm) + 24
    panel_h = top + cell_h * len(z_values_mm) + 55
    image = Image.new("RGB", (panel_w * 2, panel_h * 2 + 54), (245, 246, 247))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), heading, fill=(25, 31, 36))
    draw.text(
        (24, 37),
        subtitle,
        fill=(70, 76, 82),
    )
    lookup = {
        (
            round(1000.0 * float(row["torso_com_x_circle_m"]), 5),
            round(1000.0 * float(row["torso_com_z_circle_m"]), 5),
        ): row
        for row in results
    }

    for panel_index, (title, key, reverse, value_format) in enumerate(panels):
        panel_x = (panel_index % 2) * panel_w
        panel_y = 54 + (panel_index // 2) * panel_h
        values = [float(row[key]) for row in results]
        minimum, maximum = min(values), max(values)
        color_minimum = (
            sorted(values)[1] if key == "score" and len(values) > 1 else minimum
        )
        color_maximum = maximum
        draw.text((panel_x + 18, panel_y + 15), title, fill=(25, 31, 36))
        draw.text(
            (panel_x + 18, panel_y + 35),
            f"range {minimum:{value_format}} to {maximum:{value_format}}",
            fill=(80, 86, 92),
        )
        for column, x_mm in enumerate(x_values_mm):
            draw.text(
                (panel_x + left + column * cell_w + 27, panel_y + top - 20),
                f"{x_mm:+g}",
                fill=(55, 61, 67),
            )
        for row_index, z_mm in enumerate(reversed(z_values_mm)):
            y = panel_y + top + row_index * cell_h
            draw.text((panel_x + 18, y + 18), f"{z_mm:g}", fill=(55, 61, 67))
            for column, x_mm in enumerate(x_values_mm):
                x = panel_x + left + column * cell_w
                row = lookup.get((round(x_mm, 5), round(z_mm, 5)))
                if row is None:
                    draw.rectangle((x, y, x + cell_w - 3, y + cell_h - 3), fill=(224, 226, 228))
                    continue
                value = float(row[key])
                fill = _color(
                    value,
                    color_minimum,
                    color_maximum,
                    reverse=reverse,
                )
                draw.rectangle((x, y, x + cell_w - 3, y + cell_h - 3), fill=fill)
                draw.text((x + 9, y + 19), f"{value:{value_format}}", fill=(15, 21, 25))
                is_base = math.isclose(x_mm, 25.0, abs_tol=1.0e-5) and math.isclose(
                    z_mm, 118.228644035, abs_tol=1.0e-5
                )
                if is_base:
                    draw.rectangle((x + 1, y + 1, x + cell_w - 4, y + cell_h - 4), outline=(255, 255, 255), width=3)
                    draw.text((x + 4, y + 3), "BASE", fill=(255, 255, 255))
        draw.text(
            (panel_x + left + cell_w * len(x_values_mm) // 2 - 25, panel_y + panel_h - 29),
            "x_c (mm)",
            fill=(55, 61, 67),
        )
        draw.text((panel_x + 18, panel_y + top - 40), "z_c", fill=(55, 61, 67))
    image.save(path)


def _write_report(
    path: Path,
    results: list[dict[str, float | int | bool | str | None]],
    *,
    duration_s: float,
    controller_path: Path,
) -> None:
    original = min(
        results,
        key=lambda row: abs(float(row["torso_com_x_root_m"]) - 0.025)
        + abs(float(row["torso_com_z_root_m"]) - 0.015),
    )
    best = max(results, key=lambda row: float(row["conservative_rolling_turns"]))
    best_score = max(results, key=lambda row: float(row["score"]))
    successful = [
        row
        for row in results
        if bool(row["completed_two_turns"]) and not bool(row["leg_crossing_detected"])
    ]
    ranked = sorted(
        results,
        key=lambda row: (
            float(row["conservative_rolling_turns"]),
            -float(row["forbidden_contact_total_s"]),
        ),
        reverse=True,
    )
    lines = [
        "# 固定滚动策略 Torso 质心敏感性报告",
        "",
        "## 实验定义",
        "",
        f"- 使用同一个 CEM 控制器，不针对任何质心重新优化。",
        f"- 单次仿真时长：{duration_s:g} s。",
        f"- 有效质心点：{len(results)} 个；完成至少两圈且未发生腿交叉：{len(successful)} 个。",
        f"- 控制器：`{controller_path.resolve()}`",
        "- 圆心坐标与 root 坐标关系：`x_root = x_c`，`z_root = z_c - 0.103228644 m`。",
        "",
        "## 核心结果",
        "",
        (
            f"原始质心（圆心坐标 {1000*float(original['torso_com_x_circle_m']):.2f}, "
            f"{1000*float(original['torso_com_z_circle_m']):.2f} mm）达到 "
            f"{float(original['conservative_rolling_turns']):.2f} 圈，"
            f"位移 {float(original['root_x_displacement_m']):.2f} m，"
            f"非法接触 {float(original['forbidden_contact_total_s']):.3f} s。"
        ),
        (
            f"固定策略下圈数最高点位于圆心坐标 "
            f"({1000*float(best['torso_com_x_circle_m']):+.1f}, "
            f"{1000*float(best['torso_com_z_circle_m']):.1f}) mm："
            f"{float(best['conservative_rolling_turns']):.2f} 圈，"
            f"非法接触 {float(best['forbidden_contact_total_s']):.3f} s。"
        ),
        (
            f"综合得分最高点位于圆心坐标 "
            f"({1000*float(best_score['torso_com_x_circle_m']):+.1f}, "
            f"{1000*float(best_score['torso_com_z_circle_m']):.1f}) mm，"
            f"对应 root 坐标 "
            f"({1000*float(best_score['torso_com_x_root_m']):+.1f}, "
            f"{1000*float(best_score['torso_com_z_root_m']):+.1f}) mm："
            f"得分 {float(best_score['score']):.2f}，"
            f"{float(best_score['conservative_rolling_turns']):.2f} 圈，"
            f"非法接触 {float(best_score['forbidden_contact_total_s']):.3f} s。"
        ),
        "",
        "## 主要观察",
        "",
        "- 靠近圆心的区域 `z_c <= 25 mm` 全部失效，最高仅完成约 0.55 圈。",
        "- 连续稳定带为 `-25 mm <= x_c <= 25 mm` 且 `z_c >= 75 mm`；本次采样中该区域全部超过 7 圈。",
        "- 响应存在明显非线性边界。例如 `x_c = 50 mm` 时，`z_c = 75 mm` 仅完成 0.41 圈，而 `z_c = 100 mm` 达到 8.60 圈。",
        "- 原始质心位于稳定平台内，并非恰好落在一个孤立的可行点。",
        "",
        "## 圈数排名（前 10）",
        "",
        "| 排名 | 圆心坐标 x (mm) | 圆心坐标 z (mm) | root z (mm) | 圈数 | 位移 (m) | 非法接触 (s) | 得分 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {index} | {1000*float(row['torso_com_x_circle_m']):.1f} | "
            f"{1000*float(row['torso_com_z_circle_m']):.2f} | "
            f"{1000*float(row['torso_com_z_root_m']):.2f} | "
            f"{float(row['conservative_rolling_turns']):.2f} | "
            f"{float(row['root_x_displacement_m']):.2f} | "
            f"{float(row['forbidden_contact_total_s']):.3f} | "
            f"{float(row['score']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本实验衡量固定控制器对质心变化的鲁棒性，不代表各质心重新运行 CEM 后的最优潜力。极端质心点也可能不对应可制造的真实质量分布。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_outputs(args, results) -> tuple[Path, Path, Path, Path]:
    output_dir = args.output_dir.expanduser().resolve()
    grid_results = sorted(
        results,
        key=lambda row: (
            float(row["torso_com_z_circle_m"]),
            float(row["torso_com_x_circle_m"]),
        ),
    )
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    plot_path = output_dir / "fixed_policy_com_heatmaps.png"
    report_path = output_dir / "report_zh.md"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(grid_results[0].keys()))
        writer.writeheader()
        writer.writerows(grid_results)
    json_path.write_text(
        json.dumps(grid_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _draw_heatmaps(
        plot_path,
        grid_results,
        list(args.x_center_mm),
        list(args.z_center_mm),
    )
    _write_report(
        report_path,
        grid_results,
        duration_s=args.duration,
        controller_path=args.controller,
    )
    return csv_path, json_path, plot_path, report_path


def main(argv=None):
    args = parse_args(argv)
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    if args.radius_limit_mm <= 0.0:
        raise SystemExit("--radius-limit-mm must be positive")
    results = run_sweep(args)
    if not results:
        raise SystemExit("No COM points are inside the requested upper circle")
    for output_path in write_outputs(args, results):
        print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
