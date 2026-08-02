"""Run the historical three-stage CEM curriculum at one torso COM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.model import write_mjcf
from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts.optimize_phase_controller import (
    FOOT_GAP_TRACKING_MARGIN_M,
    _load_controller_parameters,
    optimize_controller,
    rollout_controller,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "staged_cem_com_x0_z15"


@dataclass(frozen=True)
class StageConfig:
    name: str
    description: str
    model_kind: str
    generations: int
    population: int
    elite_count: int
    duration_s: float
    barrier_generations: int
    seed: int
    minimum_foot_gap_m: float
    enforce_leg_crossing_constraint: bool


STAGES = (
    StageConfig(
        name="01_no_self_collision_cold_start",
        description="No robot self-collision; uniform cold start",
        model_kind="no_self_collision",
        generations=12,
        population=64,
        elite_count=10,
        duration_s=8.0,
        barrier_generations=3,
        seed=11,
        minimum_foot_gap_m=0.0,
        enforce_leg_crossing_constraint=False,
    ),
    StageConfig(
        name="02_collision_constrained",
        description="Current self-collision constraints",
        model_kind="collision",
        generations=14,
        population=48,
        elite_count=8,
        duration_s=6.0,
        barrier_generations=3,
        seed=23,
        minimum_foot_gap_m=0.0,
        enforce_leg_crossing_constraint=True,
    ),
    StageConfig(
        name="03_foot_gap_2mm",
        description="Self-collision plus 2 mm minimum foot gap",
        model_kind="collision",
        generations=10,
        population=48,
        elite_count=8,
        duration_s=10.0,
        barrier_generations=0,
        seed=0,
        minimum_foot_gap_m=0.002,
        enforce_leg_crossing_constraint=True,
    ),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso-com-x-mm", type=float, default=0.0)
    parser.add_argument("--torso-com-z-mm", type=float, default=15.0)
    parser.add_argument("--final-duration", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Run all stages again instead of resuming completed stages.",
    )
    return parser.parse_args(argv)


def _run_stage(
    config: StageConfig,
    *,
    model_path: Path,
    output_dir: Path,
    initial_parameters: np.ndarray | None,
    final_duration_s: float,
    workers: int,
) -> tuple[np.ndarray, dict]:
    parameters, history, _ = optimize_controller(
        model_path=model_path,
        generations=config.generations,
        population=config.population,
        elite_count=config.elite_count,
        duration=config.duration_s,
        seed=config.seed,
        barrier_generations=config.barrier_generations,
        initial_parameters=initial_parameters,
        workers=workers,
        minimum_foot_surface_gap_m=config.minimum_foot_gap_m,
        foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
        enforce_leg_crossing_constraint=(
            config.enforce_leg_crossing_constraint
        ),
    )
    baseline_model = mujoco.MjModel.from_xml_path(str(model_path))
    baseline = rollout_controller(
        baseline_model,
        np.zeros(8),
        duration=final_duration_s,
        enforce_leg_crossing_constraint=(
            config.enforce_leg_crossing_constraint
        ),
        detailed=True,
    )
    controlled_model = mujoco.MjModel.from_xml_path(str(model_path))
    controlled = rollout_controller(
        controlled_model,
        parameters[:8],
        duration=final_duration_s,
        oscillator_rate=float(parameters[8]),
        oscillator_coupling=float(parameters[9]),
        minimum_foot_surface_gap_m=config.minimum_foot_gap_m,
        foot_gap_tracking_margin_m=FOOT_GAP_TRACKING_MARGIN_M,
        enforce_leg_crossing_constraint=(
            config.enforce_leg_crossing_constraint
        ),
        detailed=True,
    )
    outputs = write_outputs(
        output_dir,
        parameters,
        history,
        baseline,
        controlled,
        config.minimum_foot_gap_m,
        FOOT_GAP_TRACKING_MARGIN_M,
    )
    summary = {
        "stage": config.name,
        "description": config.description,
        "model_kind": config.model_kind,
        "model_path": str(model_path.resolve()),
        "controller_path": str(outputs[0].resolve()),
        "initialization": (
            "cold_start_uniform"
            if initial_parameters is None
            else "previous_stage_controller"
        ),
        "generations": config.generations,
        "population": config.population,
        "elite_count": config.elite_count,
        "duration_s": config.duration_s,
        "final_duration_s": final_duration_s,
        "barrier_generations": config.barrier_generations,
        "seed": config.seed,
        "minimum_foot_gap_m": config.minimum_foot_gap_m,
        "leg_crossing_constraint_enabled": (
            config.enforce_leg_crossing_constraint
        ),
        **controlled.summary,
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return parameters, summary


def _write_summary(output_dir: Path, results: list[dict], args) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 三阶段 CEM 质心对照",
        "",
        (
            f"Torso 质心（root 坐标）：({args.torso_com_x_mm:+.1f}, "
            f"{args.torso_com_z_mm:+.1f}) mm。"
        ),
        "",
        "| 阶段 | 初始化 | 自碰撞 | 间隙目标 | 10 s 圈数 | 足端接触 | 最大足端重叠 | 其他非法接触 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['stage']} | {result['initialization']} | "
            f"{'开' if result['model_kind'] == 'collision' else '关'} | "
            f"{1000.0*float(result['minimum_foot_gap_m']):.1f} mm | "
            f"{float(result['conservative_rolling_turns']):.3f} | "
            f"{float(result['foot_contact_total_s']):.3f} s | "
            f"{1000.0*max(-float(result['minimum_foot_surface_gap_m']), 0.0):.3f} mm | "
            f"{float(result['forbidden_contact_total_s']):.3f} s |"
        )
    lines.extend(
        [
            "",
            "## 阶段定义",
            "",
            "1. 第一阶段从完整参数范围均匀冷启动，关闭机器人自碰撞和腿交叉失败检查。",
            "2. 第二阶段加载第一阶段控制器，恢复当前物理自碰撞和腿交叉约束。",
            "3. 第三阶段加载第二阶段控制器，并增加 2 mm 足端表面间隙与 4 mm 跟踪余量。",
            "",
            "## CEM 配置",
            "",
            "| 阶段 | 代数 x 候选 | 精英 | 越障代数 | rollout | seed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for config in STAGES:
        lines.append(
            f"| {config.name} | {config.generations} x {config.population} | "
            f"{config.elite_count} | {config.barrier_generations} | "
            f"{config.duration_s:g} s | {config.seed} |"
        )
    if len(results) == len(STAGES):
        collision_stage = results[1]
        gap_stage = results[2]
        contact_before = float(collision_stage["foot_contact_total_s"])
        contact_reduction = (
            100.0
            * (1.0 - float(gap_stage["foot_contact_total_s"]) / contact_before)
            if contact_before > 0.0
            else 0.0
        )
        overlap_before = max(
            -float(collision_stage["minimum_foot_surface_gap_m"]), 0.0
        )
        overlap_after = max(
            -float(gap_stage["minimum_foot_surface_gap_m"]), 0.0
        )
        overlap_reduction = (
            100.0 * (1.0 - overlap_after / overlap_before)
            if overlap_before > 0.0
            else 0.0
        )
        lines.extend(
            [
                "",
                "## 结论",
                "",
                (
                    "三阶段路线成功。最终阶段 10 s 达到 "
                    f"{float(gap_stage['conservative_rolling_turns']):.3f} 圈，"
                    f"得分 {float(gap_stage['score']):.3f}，没有腿交叉。"
                ),
                (
                    "加入足端间隙目标后，足端接触时间相对第二阶段减少 "
                    f"{contact_reduction:.1f}%，最大足端重叠减少 "
                    f"{overlap_reduction:.1f}%。"
                ),
                (
                    "2 mm 是控制目标而不是硬约束；动态回放中仍有 "
                    f"{float(gap_stage['foot_contact_total_s']):.3f} s 足端接触，"
                    f"最大瞬时重叠 {1000.0*overlap_after:.3f} mm。"
                ),
                "",
            ]
        )
    (output_dir / "report_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.final_duration <= 0.0:
        raise SystemExit("--final-duration must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    model_dir = output_dir / "models"
    parameters = replace(
        FIXED_PARAMETERS,
        torso_com_x=args.torso_com_x_mm / 1000.0,
        torso_com_z=args.torso_com_z_mm / 1000.0,
    )
    no_collision_model = write_mjcf(
        model_dir / "no_self_collision.xml",
        parameters,
        enable_self_collision=False,
    )
    collision_model = write_mjcf(
        model_dir / "collision.xml",
        parameters,
        enable_self_collision=True,
    )
    models = {
        "no_self_collision": no_collision_model,
        "collision": collision_model,
    }

    previous_parameters: np.ndarray | None = None
    results: list[dict] = []
    for index, config in enumerate(STAGES, start=1):
        stage_dir = output_dir / config.name
        result_path = stage_dir / "result.json"
        controller_path = stage_dir / "best_phase_controller.json"
        if result_path.exists() and controller_path.exists() and not args.restart:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            previous_parameters = _load_controller_parameters(controller_path)
            status = "resume"
        else:
            print(
                f"stage={index}/{len(STAGES)} name={config.name} "
                f"initialization={'cold' if previous_parameters is None else 'previous'}",
                flush=True,
            )
            previous_parameters, result = _run_stage(
                config,
                model_path=models[config.model_kind],
                output_dir=stage_dir,
                initial_parameters=previous_parameters,
                final_duration_s=args.final_duration,
                workers=args.workers,
            )
            status = "complete"
        results.append(result)
        _write_summary(output_dir, results, args)
        print(
            f"  {status} turns={float(result['conservative_rolling_turns']):.3f} "
            f"contact={float(result['forbidden_contact_total_s']):.3f}s "
            f"score={float(result['score']):.3f}",
            flush=True,
        )

    print(f"output={output_dir / 'report_zh.md'}")
    print(f"output={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
