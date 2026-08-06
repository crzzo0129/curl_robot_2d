"""Summarize the 3x3 staged-CEM torso COM design grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "staged_cem_com_grid"
CONTACT_TIME_LIMIT_S = 0.1
X_VALUES = (-50.0, 0.0, 50.0)
Z_VALUES = (-53.228644035, -20.0, 15.0)
POINTS = (
    (-50.0, 15.0, PROJECT_ROOT / "results" / "staged_cem_selected_com" / "x_m50_z_p15" / "summary.json"),
    (0.0, 15.0, PROJECT_ROOT / "results" / "staged_cem_com_x0_z15" / "summary.json"),
    (50.0, 15.0, PROJECT_ROOT / "results" / "staged_cem_selected_com" / "x_p50_z_p15" / "summary.json"),
    (-50.0, -20.0, OUTPUT_DIR / "x_m50_z_m20" / "summary.json"),
    (0.0, -20.0, OUTPUT_DIR / "x_p0_z_m20" / "summary.json"),
    (50.0, -20.0, OUTPUT_DIR / "x_p50_z_m20" / "summary.json"),
    (-50.0, -53.228644035, OUTPUT_DIR / "x_m50_z_m53" / "summary.json"),
    (0.0, -53.228644035, PROJECT_ROOT / "results" / "staged_cem_selected_com" / "x_p0_z_m53" / "summary.json"),
    (50.0, -53.228644035, OUTPUT_DIR / "x_p50_z_m53" / "summary.json"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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


def _point_label(x_mm: float, z_mm: float) -> str:
    return f"({x_mm:+.0f}, {z_mm:+.2f})"


def load_rows() -> list[dict]:
    rows = []
    for x_mm, z_mm, source_path in POINTS:
        stages = json.loads(source_path.read_text(encoding="utf-8-sig"))
        if len(stages) != 3:
            raise ValueError(f"Expected three stages in {source_path}")
        final = stages[2]
        turns = float(final["conservative_rolling_turns"])
        positive_work = float(final["actuator_positive_work_J"])
        foot_contact = float(final["foot_contact_total_s"])
        forbidden_contact = float(final["forbidden_contact_total_s"])
        rows.append(
            {
                "point": _point_label(x_mm, z_mm),
                "torso_com_x_root_mm": x_mm,
                "torso_com_z_root_mm": z_mm,
                "stage1_turns": float(stages[0]["conservative_rolling_turns"]),
                "stage2_turns": float(stages[1]["conservative_rolling_turns"]),
                "final_turns": turns,
                "score": float(final["score"]),
                "actuator_positive_work_J": positive_work,
                "turns_per_positive_J": turns / positive_work,
                "foot_contact_total_s": foot_contact,
                "forbidden_contact_total_s": forbidden_contact,
                "maximum_foot_overlap_mm": 1000.0
                * max(-float(final["minimum_foot_surface_gap_m"]), 0.0),
                "maximum_forbidden_penetration_mm": 1000.0
                * float(final["maximum_forbidden_penetration_m"]),
                "airborne_fraction": float(final["airborne_fraction"]),
                "maximum_actuator_torque_Nm": float(
                    final["maximum_actuator_torque_Nm"]
                ),
                "leg_crossing_detected": bool(final["leg_crossing_detected"]),
                "contact_feasible": (
                    foot_contact <= CONTACT_TIME_LIMIT_S
                    and forbidden_contact <= CONTACT_TIME_LIMIT_S
                    and not bool(final["leg_crossing_detected"])
                ),
                "source_summary": str(source_path.resolve()),
            }
        )
    _mark_pareto(rows, feasible_only=False, key="energy_turns_pareto")
    _mark_pareto(rows, feasible_only=True, key="feasible_energy_turns_pareto")
    return rows


def _mark_pareto(rows: list[dict], *, feasible_only: bool, key: str) -> None:
    candidates = [row for row in rows if row["contact_feasible"]] if feasible_only else rows
    for row in rows:
        if feasible_only and not row["contact_feasible"]:
            row[key] = False
            continue
        row[key] = not any(
            other is not row
            and other["actuator_positive_work_J"] <= row["actuator_positive_work_J"]
            and other["final_turns"] >= row["final_turns"]
            and (
                other["actuator_positive_work_J"] < row["actuator_positive_work_J"]
                or other["final_turns"] > row["final_turns"]
            )
            for other in candidates
        )


def _mix(first, second, amount: float):
    amount = min(max(amount, 0.0), 1.0)
    return tuple(
        round(a + (b - a) * amount) for a, b in zip(first, second)
    )


def _heat_color(value: float, low: float, high: float, *, lower_is_better: bool):
    amount = 0.5 if high <= low else (value - low) / (high - low)
    if lower_is_better:
        amount = 1.0 - amount
    bad = (204, 78, 68)
    middle = (239, 190, 77)
    good = (42, 137, 101)
    return _mix(bad, middle, amount * 2.0) if amount < 0.5 else _mix(middle, good, (amount - 0.5) * 2.0)


def _draw_heatmap(draw, bounds, rows, *, key, title, unit, digits, lower_is_better=False):
    left, top, right, bottom = bounds
    draw.rectangle(bounds, fill=(249, 250, 251), outline=(198, 205, 211), width=2)
    draw.text((left + 18, top + 14), title, fill=(27, 33, 39), font=_font(21, bold=True))
    lookup = {
        (float(row["torso_com_x_root_mm"]), float(row["torso_com_z_root_mm"])): row
        for row in rows
    }
    values = [float(row[key]) for row in rows]
    low, high = min(values), max(values)
    grid_left, grid_top = left + 94, top + 64
    grid_right, grid_bottom = right - 22, bottom - 52
    cell_w = (grid_right - grid_left) / 3.0
    cell_h = (grid_bottom - grid_top) / 3.0
    for column, x_mm in enumerate(X_VALUES):
        x = grid_left + cell_w * (column + 0.5)
        draw.text((x - 22, grid_bottom + 13), f"{x_mm:+.0f}", fill=(65, 72, 79), font=_font(14))
    for row_index, z_mm in enumerate(reversed(Z_VALUES)):
        y = grid_top + cell_h * (row_index + 0.5)
        draw.text((left + 12, y - 8), f"{z_mm:+.1f}", fill=(65, 72, 79), font=_font(14))
        for column, x_mm in enumerate(X_VALUES):
            row = lookup[(x_mm, z_mm)]
            value = float(row[key])
            x0 = grid_left + cell_w * column
            y0 = grid_top + cell_h * row_index
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            color = _heat_color(value, low, high, lower_is_better=lower_is_better)
            draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=color)
            text = f"{value:.{digits}f}{unit}"
            bbox = draw.textbbox((0, 0), text, font=_font(18, bold=True))
            draw.text(
                ((x0 + x1 - (bbox[2] - bbox[0])) / 2, (y0 + y1 - (bbox[3] - bbox[1])) / 2 - 2),
                text,
                fill=(20, 25, 29),
                font=_font(18, bold=True),
            )
    draw.text((grid_left + (grid_right - grid_left) / 2 - 62, bottom - 25), "root x (mm)", fill=(82, 89, 96), font=_font(14))
    draw.text((left + 8, top + 43), "root z", fill=(82, 89, 96), font=_font(13))


def _draw_scatter(draw, bounds, rows):
    left, top, right, bottom = bounds
    draw.rectangle(bounds, fill=(249, 250, 251), outline=(198, 205, 211), width=2)
    draw.text((left + 18, top + 14), "Turns-energy Pareto", fill=(27, 33, 39), font=_font(21, bold=True))
    x0, y0, x1, y1 = left + 78, top + 62, right - 24, bottom - 58
    works = [float(row["actuator_positive_work_J"]) for row in rows]
    turns = [float(row["final_turns"]) for row in rows]
    x_min, x_max = min(works) - 2.0, max(works) + 2.0
    y_min, y_max = min(turns) - 0.2, max(turns) + 0.2
    for tick in range(5):
        work = x_min + (x_max - x_min) * tick / 4.0
        x = x0 + (x1 - x0) * tick / 4.0
        draw.line((x, y0, x, y1), fill=(224, 228, 232), width=1)
        draw.text((x - 15, y1 + 10), f"{work:.0f}", fill=(75, 82, 89), font=_font(13))
        turns_tick = y_min + (y_max - y_min) * tick / 4.0
        y = y1 - (y1 - y0) * tick / 4.0
        draw.line((x0, y, x1, y), fill=(224, 228, 232), width=1)
        draw.text((left + 18, y - 7), f"{turns_tick:.1f}", fill=(75, 82, 89), font=_font(13))
    for row in rows:
        x = x0 + (x1 - x0) * (float(row["actuator_positive_work_J"]) - x_min) / (x_max - x_min)
        y = y1 - (y1 - y0) * (float(row["final_turns"]) - y_min) / (y_max - y_min)
        if row["contact_feasible"]:
            fill = (40, 126, 162)
            outline = (226, 157, 48) if row["feasible_energy_turns_pareto"] else (26, 77, 96)
            width = 4 if row["feasible_energy_turns_pareto"] else 2
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=fill, outline=outline, width=width)
        else:
            draw.line((x - 7, y - 7, x + 7, y + 7), fill=(197, 67, 61), width=4)
            draw.line((x - 7, y + 7, x + 7, y - 7), fill=(197, 67, 61), width=4)
        draw.text((x + 10, y - 16), row["point"], fill=(50, 57, 63), font=_font(12))
    draw.text((x0 + (x1 - x0) / 2 - 75, bottom - 27), "positive actuator work (J)", fill=(75, 82, 89), font=_font(14))
    draw.text((left + 11, top + 42), "turns", fill=(75, 82, 89), font=_font(13))
    draw.ellipse((right - 247, top + 22, right - 235, top + 34), fill=(40, 126, 162))
    draw.text((right - 229, top + 18), "contact-feasible", fill=(65, 72, 79), font=_font(12))
    draw.line((right - 122, top + 22, right - 110, top + 34), fill=(197, 67, 61), width=3)
    draw.line((right - 122, top + 34, right - 110, top + 22), fill=(197, 67, 61), width=3)
    draw.text((right - 104, top + 18), "contact-fail", fill=(65, 72, 79), font=_font(12))


def draw_plot(path: Path, rows: list[dict]) -> None:
    image = Image.new("RGB", (1800, 1120), (239, 242, 244))
    draw = ImageDraw.Draw(image)
    draw.text((44, 24), "3x3 staged-CEM torso COM design grid", fill=(20, 26, 31), font=_font(32, bold=True))
    draw.text((44, 65), "Final 10 s stage; root coordinates in millimetres", fill=(76, 84, 91), font=_font(18))
    panels = (
        ((35, 105, 590, 565), "final_turns", "Rolling turns", "", 3, False),
        ((622, 105, 1177, 565), "actuator_positive_work_J", "Positive actuator work", " J", 1, True),
        ((1209, 105, 1764, 565), "turns_per_positive_J", "Energy efficiency", " turn/J", 3, False),
        ((35, 602, 590, 1062), "score", "CEM score", "", 1, False),
        ((622, 602, 1177, 1062), "forbidden_contact_total_s", "Other forbidden contact", " s", 3, True),
    )
    for bounds, key, title, unit, digits, lower_is_better in panels:
        _draw_heatmap(
            draw,
            bounds,
            rows,
            key=key,
            title=title,
            unit=unit,
            digits=digits,
            lower_is_better=lower_is_better,
        )
    _draw_scatter(draw, (1209, 602, 1764, 1062), rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_report(path: Path, rows: list[dict]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row["torso_com_z_root_mm"]), float(row["torso_com_x_root_mm"])),
    )
    fastest = max(rows, key=lambda row: float(row["final_turns"]))
    most_efficient = max(rows, key=lambda row: float(row["turns_per_positive_J"]))
    feasible = [row for row in rows if row["contact_feasible"]]
    feasible_efficient = max(feasible, key=lambda row: float(row["turns_per_positive_J"]))
    best_score = max(rows, key=lambda row: float(row["score"]))
    pareto = [row for row in rows if row["feasible_energy_turns_pareto"]]
    lines = [
        "# 3x3 质心三阶段 CEM 对比",
        "",
        "## 实验定义",
        "",
        "- root x：-50、0、+50 mm；root z：-53.23、-20、+15 mm。",
        "- 每点使用相同三阶段 pipeline 和固定 seed 序列。",
        "- 能量使用 10 s 正执行器功；效率定义为保守圈数除以正执行器功。",
        "- 接触可行定义：足端接触不超过 0.1 s、其他非法接触不超过 0.1 s，且无腿交叉。",
        "",
        "## 最终阶段数据",
        "",
        "| root x | root z | 圈数 | 正功 | 圈/J | 得分 | 足端接触 | 其他非法接触 | 最大足端重叠 | 可行 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['torso_com_x_root_mm']:+.0f} | {row['torso_com_z_root_mm']:+.2f} | "
            f"{row['final_turns']:.3f} | {row['actuator_positive_work_J']:.2f} J | "
            f"{row['turns_per_positive_J']:.3f} | {row['score']:.2f} | "
            f"{row['foot_contact_total_s']:.3f} s | {row['forbidden_contact_total_s']:.3f} s | "
            f"{row['maximum_foot_overlap_mm']:.3f} mm | {'是' if row['contact_feasible'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 主要结果",
            "",
            (
                f"- 最高圈数：{fastest['point']}，{fastest['final_turns']:.3f} 圈，"
                f"正功 {fastest['actuator_positive_work_J']:.2f} J。"
            ),
            (
                f"- 最高原始能量效率：{most_efficient['point']}，"
                f"{most_efficient['turns_per_positive_J']:.3f} 圈/J；但接触不满足阈值。"
            ),
            (
                f"- 最高接触可行能量效率：{feasible_efficient['point']}，"
                f"{feasible_efficient['final_turns']:.3f} 圈，"
                f"{feasible_efficient['actuator_positive_work_J']:.2f} J，"
                f"{feasible_efficient['turns_per_positive_J']:.3f} 圈/J。"
            ),
            (
                f"- 最高 CEM 得分：{best_score['point']}，{best_score['score']:.2f}。"
            ),
            "- 接触可行的圈数-能量 Pareto 点："
            + "、".join(row["point"] for row in sorted(pareto, key=lambda item: item["actuator_positive_work_J"]))
            + "。",
            "",
            "## 趋势",
            "",
            "1. z=+15 和 z=-20 时，正 x 点圈数最高；z=-53.23 时该优势消失，并出现较多非法接触。",
            "2. 中心列 x=0 对 z 的圈数不敏感，但质心降低会减少正功：36.30、30.62、26.97 J。最低点虽然效率最高，但接触不合格。",
            "3. (+50,-20) 与 (+50,+15) 圈数几乎相同，但前者正功低约 11%，腾空更少，综合得分更高。",
            "4. (0,-20) 只比 (0,+15) 少约 0.15 圈，却减少约 16% 正功，并保持接触可行，是当前最清洁的节能候选。",
            "5. 负 x 整体较慢；降低 z 能恢复部分圈数，但中间高度的非法接触超过 0.1 s。",
            "",
            "## 结论边界",
            "",
            "每个点只有一个 seed，因此该网格适合筛选候选区域和识别明显失败区域，不足以证明相邻点之间 0.1–0.2 圈的差异具有统计稳定性。下一步应只对 Pareto 候选增加 seed，而不是继续扩大整个网格。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    csv_path = output_dir / "grid_final_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "grid_final_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_path = output_dir / "grid_turns_energy_pareto.png"
    draw_plot(plot_path, rows)
    report_path = output_dir / "grid_report_zh.md"
    write_report(report_path, rows)
    for output_path in (csv_path, json_path, plot_path, report_path):
        print(f"output={output_path}")


if __name__ == "__main__":
    main()
