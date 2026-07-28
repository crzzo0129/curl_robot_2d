"""Run a staged, single-GPU PPO sweep before committing to a long run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_mjx_ppo import PRESETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    learning_rate: float
    entropy_cost: float
    discounting: float
    reward_termination: float

    def training_args(self) -> list[str]:
        return [
            "--learning-rate",
            str(self.learning_rate),
            "--entropy-cost",
            str(self.entropy_cost),
            "--discounting",
            str(self.discounting),
            "--reward-termination",
            str(self.reward_termination),
        ]


# The smoke result had 100% early failures and KL=0.111.  These candidates
# test longer credit assignment, stronger failure cost, and gentler updates
# while retaining the current settings as a control.
DEFAULT_CANDIDATES = (
    Candidate("baseline", 3e-4, 1e-2, 0.990, 5.0),
    Candidate("terminal10", 3e-4, 1e-2, 0.990, 10.0),
    Candidate("terminal20", 3e-4, 1e-2, 0.990, 20.0),
    Candidate("horizon995", 3e-4, 1e-2, 0.995, 20.0),
    Candidate("horizon997", 3e-4, 1e-2, 0.997, 20.0),
    Candidate("stable_lr", 1e-4, 1e-2, 0.995, 20.0),
    Candidate("very_stable_lr", 3e-5, 1e-2, 0.995, 20.0),
    Candidate("stable_low_entropy", 1e-4, 3e-3, 0.995, 20.0),
)


DEFAULT_BUDGETS = {
    "4090": {
        "screen_steps": 524_288,
        "confirm_steps": 4_194_304,
        "final_steps": PRESETS["4090"]["steps"],
    },
    "h200": {
        "screen_steps": 2_097_152,
        "confirm_steps": 16_777_216,
        "final_steps": PRESETS["h200"]["steps"],
    },
}


def _recent_mean(rows, key: str, *, count: int = 3, default=0.0):
    values = [
        float(row[key])
        for row in rows[-count:]
        if key in row and math.isfinite(float(row[key]))
    ]
    return sum(values) / len(values) if values else float(default)


def score_training(output_dir: Path, episode_length: int) -> dict:
    """Score physical behavior without comparing differently weighted reward."""

    history_path = output_dir / "metrics_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"no evaluation metrics in {history_path}")

    avg_length = _recent_mean(
        history, "eval/avg_episode_length", default=0.0
    )
    failed_rate = _recent_mean(
        history, "eval/episode_failed", default=1.0
    )
    timeout_rate = _recent_mean(
        history, "eval/episode_timeout", default=0.0
    )
    nonfinite_rate = _recent_mean(
        history, "eval/episode_failure_nonfinite", default=0.0
    )
    root_low_rate = _recent_mean(
        history, "eval/episode_failure_root_low", default=0.0
    )
    foot_gap_rate = _recent_mean(
        history, "eval/episode_failure_foot_gap", default=0.0
    )
    roll_per_step = _recent_mean(
        history, "eval/avg_roll_progress_rad", default=-math.inf
    )
    penetration_m = _recent_mean(
        history, "eval/avg_forbidden_penetration_m", default=math.inf
    )

    survival_fraction = min(max(avg_length / episode_length, 0.0), 1.0)
    net_turns = roll_per_step * avg_length / (2.0 * math.pi)
    progress_quality = min(max(net_turns / 3.0, -1.0), 1.0)
    safety_quality = 1.0 - min(max(penetration_m / 0.001, 0.0), 1.0)
    selection_score = (
        0.50 * survival_fraction
        + 0.35 * progress_quality
        + 0.10 * (1.0 - min(max(failed_rate, 0.0), 1.0))
        + 0.05 * safety_quality
    )
    rejected = (
        nonfinite_rate > 0.0
        or not math.isfinite(roll_per_step)
        or not math.isfinite(penetration_m)
        or not math.isfinite(selection_score)
    )
    if rejected:
        selection_score = -1_000_000.0

    return {
        "selection_score": selection_score,
        "rejected": rejected,
        "avg_episode_length": avg_length,
        "survival_fraction": survival_fraction,
        "estimated_net_turns": net_turns,
        "avg_roll_progress_rad": roll_per_step,
        "failed_rate": failed_rate,
        "timeout_rate": timeout_rate,
        "failure_root_low_rate": root_low_rate,
        "failure_foot_gap_rate": foot_gap_rate,
        "failure_nonfinite_rate": nonfinite_rate,
        "avg_forbidden_penetration_m": penetration_m,
    }


def _completed_output(base: Path) -> Path | None:
    candidates = [base, *sorted(base.parent.glob(f"{base.name}_retry*"))]
    for candidate in reversed(candidates):
        if (
            (candidate / "training_summary.json").is_file()
            and (candidate / "metrics_history.json").is_file()
        ):
            return candidate
    return None


def _next_output(base: Path) -> Path:
    if not base.exists():
        return base
    retry = 1
    while (candidate := base.with_name(f"{base.name}_retry{retry}")).exists():
        retry += 1
    return candidate


def _training_command(
    *,
    candidate: Candidate,
    preset: str,
    physics_profile: str,
    steps: int,
    num_evals: int,
    seed: int,
    episode_length: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.train_mjx_ppo",
        "--preset",
        preset,
        "--physics-profile",
        physics_profile,
        "--steps",
        str(steps),
        "--num-evals",
        str(num_evals),
        "--episode-length",
        str(episode_length),
        "--seed",
        str(seed),
        "--mujoco-gl",
        "disable",
        "--out",
        str(output_dir),
        *candidate.training_args(),
    ]


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] command: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def _run_candidate(
    *,
    candidate: Candidate,
    stage: str,
    output_root: Path,
    preset: str,
    physics_profile: str,
    steps: int,
    num_evals: int,
    seed: int,
    episode_length: int,
    resume: bool,
    dry_run: bool,
) -> dict | None:
    base = output_root / stage / candidate.name
    completed = _completed_output(base) if resume else None
    output_dir = completed or _next_output(base)
    command = _training_command(
        candidate=candidate,
        preset=preset,
        physics_profile=physics_profile,
        steps=steps,
        num_evals=num_evals,
        seed=seed,
        episode_length=episode_length,
        output_dir=output_dir,
    )
    if dry_run:
        print(f"[sweep] dry-run: {' '.join(command)}")
        return {
            "candidate": asdict(candidate),
            "stage": stage,
            "output_dir": str(output_dir),
            "command": command,
        }
    if completed is None:
        return_code = _run_command(
            command, output_root / "logs" / f"{stage}_{candidate.name}.log"
        )
        if return_code != 0:
            print(
                f"[sweep] candidate={candidate.name} failed "
                f"with exit code {return_code}",
                flush=True,
            )
            return None
    else:
        print(f"[sweep] reusing completed run: {completed}", flush=True)

    try:
        metrics = score_training(output_dir, episode_length)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(
            f"[sweep] candidate={candidate.name} has invalid metrics: "
            f"{error}",
            flush=True,
        )
        return None
    result = {
        "candidate": asdict(candidate),
        "stage": stage,
        "steps": steps,
        "seed": seed,
        "output_dir": str(output_dir),
        **metrics,
    }
    print(
        f"[sweep] candidate={candidate.name} "
        f"score={metrics['selection_score']:.4f} "
        f"length={metrics['avg_episode_length']:.1f} "
        f"turns={metrics['estimated_net_turns']:.3f}",
        flush=True,
    )
    return result


def _write_results(path: Path, rows: list[dict]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    if not rows:
        return
    flat_rows = [
        {
            "name": row["candidate"]["name"],
            **{
                key: value
                for key, value in row.items()
                if key not in ("candidate", "command")
            },
            **{
                f"parameter_{key}": value
                for key, value in row["candidate"].items()
                if key != "name"
            },
        }
        for row in rows
    ]
    with path.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def _rank(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda result: result["selection_score"],
        reverse=True,
    )


def final_quality_gate(
    result: dict,
    *,
    minimum_survival_fraction: float,
    minimum_net_turns: float,
) -> dict:
    """Require evidence of both survival and rolling before a long run."""

    reasons = []
    if result.get("rejected", True):
        reasons.append("candidate metrics were rejected")
    if result.get("survival_fraction", 0.0) < minimum_survival_fraction:
        reasons.append(
            "survival_fraction "
            f"{result.get('survival_fraction', 0.0):.3f} "
            f"< {minimum_survival_fraction:.3f}"
        )
    if result.get("estimated_net_turns", -math.inf) < minimum_net_turns:
        reasons.append(
            "estimated_net_turns "
            f"{result.get('estimated_net_turns', -math.inf):.3f} "
            f"< {minimum_net_turns:.3f}"
        )
    return {
        "passed": not reasons,
        "minimum_survival_fraction": minimum_survival_fraction,
        "minimum_net_turns": minimum_net_turns,
        "reasons": reasons,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Screen PPO settings, confirm the top candidates with a new "
            "seed, then launch one long run from scratch."
        )
    )
    parser.add_argument("--hardware", choices=tuple(DEFAULT_BUDGETS), required=True)
    parser.add_argument("--physics-profile", default="cg12")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--screen-steps", type=int)
    parser.add_argument("--confirm-steps", type=int)
    parser.add_argument("--final-steps", type=int)
    parser.add_argument("--screen-evals", type=int, default=6)
    parser.add_argument("--confirm-evals", type=int, default=8)
    parser.add_argument("--final-evals", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--min-final-survival-fraction", type=float, default=0.20
    )
    parser.add_argument("--min-final-turns", type=float, default=0.25)
    parser.add_argument(
        "--out", type=Path, default=Path("results") / "mjx_ppo_sweep"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument(
        "--force-final",
        action="store_true",
        help="Run the long stage even when no confirmed candidate passes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.top_k > len(DEFAULT_CANDIDATES):
        parser.error(f"--top-k must be between 1 and {len(DEFAULT_CANDIDATES)}")
    for name in ("episode_length", "screen_evals", "confirm_evals", "final_evals"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.min_final_survival_fraction <= 1.0:
        parser.error("--min-final-survival-fraction must be in [0, 1]")
    if args.min_final_turns < 0.0:
        parser.error("--min-final-turns must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    budgets = DEFAULT_BUDGETS[args.hardware]
    screen_steps = args.screen_steps or budgets["screen_steps"]
    confirm_steps = args.confirm_steps or budgets["confirm_steps"]
    final_steps = args.final_steps or budgets["final_steps"]
    output_root = args.out.expanduser().resolve()
    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not (args.resume or args.dry_run)
    ):
        raise SystemExit(
            f"Output directory is not empty: {output_root}. "
            "Use --resume or choose a new --out path."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_config = {
        "hardware": args.hardware,
        "physics_profile": args.physics_profile,
        "episode_length": args.episode_length,
        "screen_steps": screen_steps,
        "confirm_steps": confirm_steps,
        "final_steps": final_steps,
        "screen_seed": args.seed,
        "confirm_seed": args.seed + 1,
        "final_seed": args.seed + 2,
        "top_k": args.top_k,
        "minimum_final_survival_fraction": (
            args.min_final_survival_fraction
        ),
        "minimum_final_net_turns": args.min_final_turns,
        "candidates": [asdict(candidate) for candidate in DEFAULT_CANDIDATES],
    }
    (output_root / "sweep_config.json").write_text(
        json.dumps(sweep_config, indent=2) + "\n", encoding="utf-8"
    )

    screen_results = []
    for candidate in DEFAULT_CANDIDATES:
        result = _run_candidate(
            candidate=candidate,
            stage="screen",
            output_root=output_root,
            preset=args.hardware,
            physics_profile=args.physics_profile,
            steps=screen_steps,
            num_evals=args.screen_evals,
            seed=args.seed,
            episode_length=args.episode_length,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if result is not None:
            screen_results.append(result)
    if args.dry_run:
        return
    ranked_screen = _rank(screen_results)
    _write_results(output_root / "leaderboard_screen", ranked_screen)
    if not ranked_screen:
        raise SystemExit("All screening runs failed.")

    candidate_by_name = {
        candidate.name: candidate for candidate in DEFAULT_CANDIDATES
    }
    eligible_screen = [
        result for result in ranked_screen if not result["rejected"]
    ]
    if not eligible_screen:
        raise SystemExit("All screening runs were rejected.")
    confirmation_queue = [
        candidate_by_name[result["candidate"]["name"]]
        for result in eligible_screen
    ]
    confirm_results = []
    for candidate in confirmation_queue:
        result = _run_candidate(
            candidate=candidate,
            stage="confirm",
            output_root=output_root,
            preset=args.hardware,
            physics_profile=args.physics_profile,
            steps=confirm_steps,
            num_evals=args.confirm_evals,
            seed=args.seed + 1,
            episode_length=args.episode_length,
            resume=args.resume,
            dry_run=False,
        )
        if result is not None and not result["rejected"]:
            screen_score = next(
                item["selection_score"]
                for item in ranked_screen
                if item["candidate"]["name"] == candidate.name
            )
            result["screen_score"] = screen_score
            result["combined_score"] = (
                0.25 * screen_score + 0.75 * result["selection_score"]
            )
            result["quality_gate"] = final_quality_gate(
                result,
                minimum_survival_fraction=(
                    args.min_final_survival_fraction
                ),
                minimum_net_turns=args.min_final_turns,
            )
            confirm_results.append(result)
            if (
                len(confirm_results) >= args.top_k
                and any(
                    item["quality_gate"]["passed"]
                    for item in confirm_results
                )
            ):
                break
    ranked_confirm = sorted(
        confirm_results,
        key=lambda result: result["combined_score"],
        reverse=True,
    )
    _write_results(output_root / "leaderboard_confirm", ranked_confirm)
    if not ranked_confirm:
        raise SystemExit("All confirmation runs failed.")

    passing_confirm = [
        result
        for result in ranked_confirm
        if result["quality_gate"]["passed"]
    ]
    gate_summary = {
        "passed": bool(passing_confirm),
        "forced": bool(args.force_final and not passing_confirm),
        "minimum_survival_fraction": args.min_final_survival_fraction,
        "minimum_net_turns": args.min_final_turns,
        "confirmed_candidates": [
            {
                "name": result["candidate"]["name"],
                "combined_score": result["combined_score"],
                "survival_fraction": result["survival_fraction"],
                "estimated_net_turns": result["estimated_net_turns"],
                "quality_gate": result["quality_gate"],
            }
            for result in ranked_confirm
        ],
    }
    (output_root / "quality_gate.json").write_text(
        json.dumps(gate_summary, indent=2) + "\n", encoding="utf-8"
    )
    if not passing_confirm and not args.force_final:
        print(
            "[sweep] no confirmed candidate passed the final quality gate; "
            "the long run will not start",
            flush=True,
        )
        return

    winner_result = (
        passing_confirm[0] if passing_confirm else ranked_confirm[0]
    )
    winner = candidate_by_name[winner_result["candidate"]["name"]]
    selection = {
        "selected_candidate": asdict(winner),
        "screen_result": next(
            result
            for result in ranked_screen
            if result["candidate"]["name"] == winner.name
        ),
        "confirm_result": winner_result,
        "quality_gate": winner_result["quality_gate"],
        "quality_gate_forced": gate_summary["forced"],
    }
    (output_root / "selected_candidate.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[sweep] selected={winner.name} "
        f"combined_score={winner_result['combined_score']:.4f}",
        flush=True,
    )
    if args.skip_final:
        return

    final_result = _run_candidate(
        candidate=winner,
        stage="final",
        output_root=output_root,
        preset=args.hardware,
        physics_profile=args.physics_profile,
        steps=final_steps,
        num_evals=args.final_evals,
        seed=args.seed + 2,
        episode_length=args.episode_length,
        resume=args.resume,
        dry_run=False,
    )
    if final_result is None:
        raise SystemExit("The selected final run failed.")
    final_result["selected_from"] = winner.name
    final_result["completed_at_unix_s"] = time.time()
    (output_root / "final_result.json").write_text(
        json.dumps(final_result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
