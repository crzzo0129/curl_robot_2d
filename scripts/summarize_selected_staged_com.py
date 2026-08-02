"""Summarize selected torso-COM points run through the staged CEM pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "staged_cem_selected_com"
STAGE_LABELS = ("No self-collision", "Collision", "2 mm foot gap")
POINTS = (
    (
        "x=-50, z=15",
        -50.0,
        15.0,
        DEFAULT_OUTPUT_DIR / "x_m50_z_p15" / "summary.json",
    ),
    (
        "x=0, z=15",
        0.0,
        15.0,
        PROJECT_ROOT / "results" / "staged_cem_com_x0_z15" / "summary.json",
    ),
    (
        "x=+50, z=15",
        50.0,
        15.0,
        DEFAULT_OUTPUT_DIR / "x_p50_z_p15" / "summary.json",
    ),
    (
        "x=0, z=-53.23",
        0.0,
        -53.228644035,
        DEFAULT_OUTPUT_DIR / "x_p0_z_m53" / "summary.json",
    ),
)
COLORS = (
    (213, 94, 58),
    (35, 119, 147),
    (27, 135, 94),
    (221, 151, 44),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _font(size: int, *, bold: bool = False):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else (
        "arial.ttf",
        "DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_rows() -> list[dict]:
    rows = []
    for point_label, x_mm, z_mm, path in POINTS:
        stages = json.loads(path.read_text(encoding="utf-8-sig"))
        if len(stages) != len(STAGE_LABELS):
            raise ValueError(f"Expected three stages in {path}")
        for stage_index, (stage_label, stage) in enumerate(
            zip(STAGE_LABELS, stages), start=1
        ):
            rows.append(
                {
                    "point": point_label,
                    "torso_com_x_root_mm": x_mm,
                    "torso_com_z_root_mm": z_mm,
                    "stage_index": stage_index,
                    "stage": stage_label,
                    "conservative_rolling_turns": float(
                        stage["conservative_rolling_turns"]
                    ),
                    "score": float(stage["score"]),
                    "foot_contact_total_s": float(
                        stage["foot_contact_total_s"]
                    ),
                    "maximum_foot_overlap_mm": 1000.0
                    * max(-float(stage["minimum_foot_surface_gap_m"]), 0.0),
                    "forbidden_contact_total_s": float(
                        stage["forbidden_contact_total_s"]
                    ),
                    "maximum_forbidden_penetration_mm": 1000.0
                    * float(stage["maximum_forbidden_penetration_m"]),
                    "actuator_positive_work_J": float(
                        stage["actuator_positive_work_J"]
                    ),
                    "airborne_fraction": float(stage["airborne_fraction"]),
                    "maximum_actuator_torque_Nm": float(
                        stage["maximum_actuator_torque_Nm"]
                    ),
                    "leg_crossing_detected": bool(
                        stage["leg_crossing_detected"]
                    ),
                    "source_summary": str(path.resolve()),
                }
            )
    return rows


def _panel(draw, bounds, title, y_max, y_label):
    left, top, right, bottom = bounds
    draw.rectangle(bounds, fill=(249, 250, 251), outline=(205, 211, 217), width=2)
    draw.text((left + 16, top + 12), title, fill=(28, 34, 40), font=_font(21, bold=True))
    plot = (left + 68, top + 58, right - 24, bottom - 55)
    px0, py0, px1, py1 = plot
    for tick in range(6):
        value = y_max * tick / 5.0
        y = py1 - (py1 - py0) * tick / 5.0
        draw.line((px0, y, px1, y), fill=(224, 228, 232), width=1)
        draw.text((left + 8, y - 8), f"{value:.2f}", fill=(84, 91, 98), font=_font(14))
    draw.text((left + 10, bottom - 28), y_label, fill=(84, 91, 98), font=_font(14))
    return plot


def draw_plot(path: Path, rows: list[dict]) -> None:
    image = Image.new("RGB", (1500, 940), (240, 243, 245))
    draw = ImageDraw.Draw(image)
    draw.text((44, 25), "Staged CEM torso-COM comparison", fill=(22, 28, 34), font=_font(32, bold=True))
    draw.text(
        (44, 66),
        "Root coordinates (mm); identical three-stage pipeline at every point",
        fill=(79, 87, 94),
        font=_font(18),
    )

    by_point = {
        label: [row for row in rows if row["point"] == label]
        for label, *_ in POINTS
    }
    line_plot = _panel(
        draw,
        (40, 105, 965, 485),
        "10 s conservative rolling across stages",
        12.0,
        "turns",
    )
    x0, y0, x1, y1 = line_plot
    stage_x = [x0 + (x1 - x0) * index / 2.0 for index in range(3)]
    for index, label in enumerate(STAGE_LABELS):
        draw.text((stage_x[index] - 62, y1 + 14), label, fill=(65, 72, 78), font=_font(14))
    for point_index, (label, *_rest) in enumerate(POINTS):
        values = [row["conservative_rolling_turns"] for row in by_point[label]]
        coords = [
            (stage_x[index], y1 - (y1 - y0) * value / 12.0)
            for index, value in enumerate(values)
        ]
        draw.line(coords, fill=COLORS[point_index], width=4)
        for x, y in coords:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COLORS[point_index])

    legend_x, legend_y = 1000, 125
    draw.text((legend_x, legend_y), "COM point", fill=(28, 34, 40), font=_font(20, bold=True))
    for index, (label, *_rest) in enumerate(POINTS):
        y = legend_y + 42 + 48 * index
        draw.line((legend_x, y + 9, legend_x + 38, y + 9), fill=COLORS[index], width=5)
        draw.text((legend_x + 52, y), label, fill=(45, 52, 58), font=_font(18))

    final_rows = [row for row in rows if row["stage_index"] == 3]
    contact_plot = _panel(
        draw,
        (40, 515, 725, 910),
        "Final-stage contact time",
        0.32,
        "seconds",
    )
    x0, y0, x1, y1 = contact_plot
    group_w = (x1 - x0) / len(final_rows)
    bar_w = group_w * 0.26
    for index, row in enumerate(final_rows):
        center = x0 + group_w * (index + 0.5)
        values = (
            row["foot_contact_total_s"],
            row["forbidden_contact_total_s"],
        )
        for offset, value, color in (
            (-bar_w, values[0], (37, 132, 112)),
            (0.0, values[1], (205, 82, 75)),
        ):
            top = y1 - (y1 - y0) * value / 0.32
            draw.rectangle(
                (center + offset, top, center + offset + bar_w, y1),
                fill=color,
            )
        draw.text((center - 50, y1 + 13), row["point"], fill=(65, 72, 78), font=_font(13))
    draw.rectangle((470, 540, 486, 556), fill=(37, 132, 112))
    draw.text((493, 537), "foot", fill=(65, 72, 78), font=_font(14))
    draw.rectangle((558, 540, 574, 556), fill=(205, 82, 75))
    draw.text((581, 537), "other", fill=(65, 72, 78), font=_font(14))

    overlap_plot = _panel(
        draw,
        (760, 515, 1460, 910),
        "Final-stage maximum foot overlap",
        0.70,
        "millimetres",
    )
    x0, y0, x1, y1 = overlap_plot
    group_w = (x1 - x0) / len(final_rows)
    for index, (row, color) in enumerate(zip(final_rows, COLORS)):
        center = x0 + group_w * (index + 0.5)
        value = row["maximum_foot_overlap_mm"]
        top = y1 - (y1 - y0) * value / 0.70
        draw.rectangle((center - 25, top, center + 25, y1), fill=color)
        draw.text((center - 50, y1 + 13), row["point"], fill=(65, 72, 78), font=_font(13))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_report(path: Path, rows: list[dict]) -> None:
    by_point = {
        label: [row for row in rows if row["point"] == label]
        for label, *_ in POINTS
    }
    final_rows = [row for row in rows if row["stage_index"] == 3]
    final_by_point = {row["point"]: row for row in final_rows}
    baseline = final_by_point["x=0, z=15"]
    positive_x = final_by_point["x=+50, z=15"]
    negative_x = final_by_point["x=-50, z=15"]
    low_z = final_by_point["x=0, z=-53.23"]

    lines = [
        "# 少量质心点三阶段 CEM 对比",
        "",
        "## 实验设计",
        "",
        "所有坐标均相对 root，单位为 mm。每个点使用同一三阶段 pipeline：无自碰撞冷启动、恢复碰撞约束、增加 2 mm 足端间隙目标。每点仅运行一个固定 seed 序列。",
        "",
        "## 三阶段圈数",
        "",
        "| 质心点 | 无自碰撞 | 碰撞约束 | 2 mm 足端限制 |",
        "|---|---:|---:|---:|",
    ]
    for label, *_ in POINTS:
        values = by_point[label]
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{float(row['conservative_rolling_turns']):.3f}"
                for row in values
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 最终阶段",
            "",
            "| 质心点 | 圈数 | 得分 | 足端接触 | 最大足端重叠 | 其他非法接触 | 正执行器功 | 腾空比例 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {row['point']} | {row['conservative_rolling_turns']:.3f} | "
            f"{row['score']:.3f} | {row['foot_contact_total_s']:.3f} s | "
            f"{row['maximum_foot_overlap_mm']:.3f} mm | "
            f"{row['forbidden_contact_total_s']:.3f} s | "
            f"{row['actuator_positive_work_J']:.2f} J | "
            f"{100.0*row['airborne_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 观察",
            "",
            (
                f"1. 固定 z=15 mm 时，最终圈数随 x 从 -50、0 到 +50 mm 依次为 "
                f"{negative_x['conservative_rolling_turns']:.3f}、"
                f"{baseline['conservative_rolling_turns']:.3f}、"
                f"{positive_x['conservative_rolling_turns']:.3f}。正 x 点最快，但相对 x=0 仅增加 "
                f"{positive_x['conservative_rolling_turns']-baseline['conservative_rolling_turns']:.3f} 圈。"
            ),
            (
                f"2. +50 mm 点足端接触最短（{positive_x['foot_contact_total_s']:.3f} s），"
                f"但最大瞬时足端重叠为 {positive_x['maximum_foot_overlap_mm']:.3f} mm，"
                f"腾空比例为 {100.0*positive_x['airborne_fraction']:.2f}%，因此不是无条件更优。"
            ),
            (
                f"3. 把 z 从 +15 mm 降到 -53.23 mm 后，圈数只从 "
                f"{baseline['conservative_rolling_turns']:.3f} 降到 {low_z['conservative_rolling_turns']:.3f}，"
                f"正执行器功从 {baseline['actuator_positive_work_J']:.2f} J 降到 "
                f"{low_z['actuator_positive_work_J']:.2f} J；但足端接触增至 "
                f"{low_z['foot_contact_total_s']:.3f} s，其他非法接触增至 "
                f"{low_z['forbidden_contact_total_s']:.3f} s。"
            ),
            (
                f"4. 综合当前评分，x=0、z=15 mm 得分最高（{baseline['score']:.3f}），"
                f"与 +50 mm 点（{positive_x['score']:.3f}）非常接近。"
            ),
            "",
            "## 结论边界",
            "",
            "这些结果说明三阶段 pipeline 能在不同质心下重新找到策略，并揭示速度、能耗和接触之间的取舍。每点只有一个 seed，横向不对称既可能包含真实的滚动方向效应，也可能包含 CEM 进入不同局部最优的影响；目前不应把 0.2 圈量级的小差异解释为确定性优劣。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    csv_path = output_dir / "comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "comparison_summary.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_path = output_dir / "comparison.png"
    draw_plot(plot_path, rows)
    report_path = output_dir / "comparison_report_zh.md"
    write_report(report_path, rows)
    for path in (csv_path, json_path, plot_path, report_path):
        print(f"output={path}")


if __name__ == "__main__":
    main()
