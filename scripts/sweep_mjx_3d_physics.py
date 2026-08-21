"""Sweep 3-D MJX reference physics timestep and contact cone settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from curl_robot_2d_mjx.config_3d import GEOMETRY_NAMES_3D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "mjx_3d_physics_sweep"
DEFAULT_TIMESTEPS_MS = (1.0, 2.0, 4.0)
DEFAULT_CONES = ("elliptic", "pyramidal")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _case_name(timestep_ms: float, cone: str) -> str:
    timestep = f"{timestep_ms:g}".replace(".", "p")
    return f"dt{timestep}ms_{cone}"


def _action_repeat(control_timestep_ms: float, timestep_ms: float) -> int:
    ratio = control_timestep_ms / timestep_ms
    repeat = round(ratio)
    if repeat < 1 or not math.isclose(ratio, repeat, abs_tol=1e-9):
        raise ValueError(
            f"physics timestep {timestep_ms:g} ms does not divide the "
            f"{control_timestep_ms:g} ms control timestep"
        )
    return repeat


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--controller", type=Path)
    parser.add_argument(
        "--geometry", choices=GEOMETRY_NAMES_3D, default="pupper_open60"
    )
    parser.add_argument(
        "--timesteps-ms",
        type=_positive_float,
        nargs="+",
        default=list(DEFAULT_TIMESTEPS_MS),
    )
    parser.add_argument(
        "--cones",
        choices=("elliptic", "pyramidal"),
        nargs="+",
        default=list(DEFAULT_CONES),
    )
    parser.add_argument(
        "--control-timestep-ms", type=_positive_float, default=20.0
    )
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Use at least two equal chunks so the last excludes XLA compile.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=10000)
    parser.add_argument(
        "--reset-joint-noise-rad", type=_nonnegative_float, default=0.005
    )
    parser.add_argument(
        "--reset-velocity-noise", type=_nonnegative_float, default=0.005
    )
    parser.add_argument(
        "--reset-pair-differential-scale", type=float, default=None
    )
    parser.add_argument(
        "--reset-axis-tilt-noise-rad",
        type=_nonnegative_float,
        default=0.0,
    )
    parser.add_argument(
        "--reference-ramp-start-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--reference-ramp-duration-s", type=_positive_float, default=0.25
    )
    parser.add_argument("--memory-fraction", type=_positive_float, default=0.50)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed cases and resume partial evaluator chunks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.episode_length < 1 or args.batch_size < 1 or args.chunk_size < 1:
        parser.error("episode length and batch sizes must be positive")
    if args.reset_pair_differential_scale is not None and not (
        0.0 <= args.reset_pair_differential_scale <= 1.0
    ):
        parser.error("--reset-pair-differential-scale must be in [0, 1]")
    try:
        for timestep_ms in args.timesteps_ms:
            _action_repeat(args.control_timestep_ms, timestep_ms)
    except ValueError as error:
        parser.error(str(error))
    return args


def _evaluation_command(args, *, timestep_ms, cone, case_dir) -> list[str]:
    repeat = _action_repeat(args.control_timestep_ms, timestep_ms)
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_mjx_3d_policy",
        "--evaluation-mode",
        "reference",
        "--geometry",
        args.geometry,
        "--physics-profile",
        "cg20",
        "--physics-timestep-ms",
        str(timestep_ms),
        "--action-repeat",
        str(repeat),
        "--cone",
        cone,
        "--episode-length",
        str(args.episode_length),
        "--batch-size",
        str(args.batch_size),
        "--chunk-size",
        str(args.chunk_size),
        "--seed",
        str(args.seed),
        "--environment-seed",
        str(args.environment_seed),
        "--reset-joint-noise-rad",
        str(args.reset_joint_noise_rad),
        "--reset-velocity-noise",
        str(args.reset_velocity_noise),
        "--reset-axis-tilt-noise-rad",
        str(args.reset_axis_tilt_noise_rad),
        "--reference-ramp-start-scale",
        str(args.reference_ramp_start_scale),
        "--reference-ramp-duration-s",
        str(args.reference_ramp_duration_s),
        "--explicit-phase-observation",
        "--mujoco-gl",
        args.mujoco_gl,
        "--memory-fraction",
        str(args.memory_fraction),
        "--out",
        str(case_dir),
    ]
    if args.controller is not None:
        command.extend(("--controller", str(args.controller)))
    if args.reset_pair_differential_scale is not None:
        command.extend(
            (
                "--reset-pair-differential-scale",
                str(args.reset_pair_differential_scale),
            )
        )
    if args.resume:
        command.append("--resume")
    return command


def _load_case(case_dir: Path) -> dict:
    path = case_dir / "deterministic_eval.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, summary: dict) -> dict[str, object]:
    task = summary["task"]
    turns = summary["conservative_turns"]
    failures = summary["failure_rates"]
    chunk_wall_times = summary.get(
        "chunk_wall_times_s", [summary["chunk_wall_time_s"]]
    )
    return {
        "case": name,
        "physics_timestep_ms": 1000.0 * task["physics_timestep"],
        "action_repeat": task["action_repeat"],
        "control_timestep_ms": 1000.0
        * task["physics_timestep"]
        * task["action_repeat"],
        "cone": task["cone_name"],
        "turns_median": turns["median"],
        "turns_p05": turns["p05"],
        "turns_p95": turns["p95"],
        "failure_rate": summary["failure_rate"],
        "timeout_rate": summary["timeout_rate"],
        "lateral_drift_rate": failures["lateral_drift"],
        "forbidden_depth_rate": failures["forbidden_depth"],
        "forbidden_contact_rate": failures["forbidden_contact"],
        "nonfinite_rate": failures["nonfinite"],
        "chunk_wall_time_s": summary["chunk_wall_time_s"],
        "steady_chunk_wall_time_s": chunk_wall_times[-1],
    }


def _write_summary(args, rows: list[dict[str, object]]) -> None:
    baseline = next(
        (
            row
            for row in rows
            if row["physics_timestep_ms"] == min(args.timesteps_ms)
            and row["cone"] == "elliptic"
        ),
        rows[0],
    )
    for row in rows:
        row["wall_speedup_vs_baseline"] = baseline[
            "steady_chunk_wall_time_s"
        ] / max(row["steady_chunk_wall_time_s"], 1e-12)
        row["turns_delta_vs_baseline"] = (
            row["turns_median"] - baseline["turns_median"]
        )
        row["failure_delta_vs_baseline"] = (
            row["failure_rate"] - baseline["failure_rate"]
        )

    payload = {
        "physics_profile": "cg20",
        "baseline_case": baseline["case"],
        "control_timestep_ms": args.control_timestep_ms,
        "batch_size": args.batch_size,
        "episode_length": args.episode_length,
        "seed": args.seed,
        "environment_seed": args.environment_seed,
        "cases": rows,
    }
    (args.out / "sweep_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[physics sweep summary]", flush=True)
    for row in rows:
        print(
            f"  {row['case']}: turns={row['turns_median']:.3f} "
            f"failed={row['failure_rate']:.2%} "
            f"lateral={row['lateral_drift_rate']:.2%} "
            f"speedup={row['wall_speedup_vs_baseline']:.2f}x "
            f"turn_delta={row['turns_delta_vs_baseline']:+.3f}",
            flush=True,
        )
    print(f"  output={args.out.resolve()}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestep_ms in args.timesteps_ms:
        for cone in args.cones:
            name = _case_name(timestep_ms, cone)
            case_dir = args.out / name
            summary_path = case_dir / "deterministic_eval.json"
            command = _evaluation_command(
                args,
                timestep_ms=timestep_ms,
                cone=cone,
                case_dir=case_dir,
            )
            print(f"[physics sweep] {name}", flush=True)
            print("  " + " ".join(command), flush=True)
            if args.dry_run:
                continue
            if summary_path.exists() and args.resume:
                print("  loaded completed case", flush=True)
            else:
                if case_dir.exists() and not args.resume:
                    raise SystemExit(
                        f"case output already exists; use --resume: {case_dir}"
                    )
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            rows.append(_row(name, _load_case(case_dir)))

    if not args.dry_run:
        _write_summary(args, rows)


if __name__ == "__main__":
    main()
