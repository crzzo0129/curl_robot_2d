"""Train conservative nominal collision avoidance on the 180 mm 3-D geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import train_mjx_3d_residual_ppo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "staged_cem_real_geometry_180_d50_foot60"
    / "03_foot_gap_2mm"
    / "best_phase_controller.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(train_mjx_3d_residual_ppo.PRESETS),
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--recipe",
        choices=("real_geometry_contact_v1", "real_geometry_contact_v2"),
        default="real_geometry_contact_v2",
    )
    parser.add_argument("--controller", type=Path, default=DEFAULT_REAL_CONTROLLER)
    parser.add_argument("--restore-params", type=Path)
    parser.add_argument("--residual-gain", type=float)
    parser.add_argument("--residual-pair-differential-scale", type=float)
    parser.add_argument("--initial-policy-std", type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "glfw", "osmesa", "disable"),
        default="disable",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args(argv)


def training_argv(args: argparse.Namespace) -> list[str]:
    output = args.out or (
        Path("results")
        / f"mjx_3d_{args.recipe}_{args.preset}_seed{args.seed}"
    )
    values = [
        "--preset",
        args.preset,
        "--geometry",
        "real",
        "--recipe",
        args.recipe,
        "--physics-profile",
        "cg20",
        "--curriculum",
        "none",
        "--controller",
        str(args.controller),
        "--minimum-foot-gap-mm",
        "2",
        "--foot-gap-tracking-margin-mm",
        "12",
        "--reference-ramp-start-scale",
        "0.25",
        "--reference-ramp-duration-s",
        "0.25",
        "--reset-joint-noise-rad",
        "0.005",
        "--reset-velocity-noise",
        "0.005",
        "--reset-pair-differential-scale",
        "0.0",
        "--reset-axis-tilt-noise-rad",
        "0.0",
        "--seed",
        str(args.seed),
        "--mujoco-gl",
        args.mujoco_gl,
        "--memory-fraction",
        str(args.memory_fraction),
        "--out",
        str(output),
    ]
    if args.steps is not None:
        values.extend(("--steps", str(args.steps)))
    if args.restore_params is not None:
        values.extend(("--restore-params", str(args.restore_params)))
    if args.residual_gain is not None:
        values.extend(("--minimum-residual-gain", str(args.residual_gain)))
    if args.residual_pair_differential_scale is not None:
        values.extend(
            (
                "--residual-pair-differential-scale",
                str(args.residual_pair_differential_scale),
            )
        )
    if args.initial_policy_std is not None:
        values.extend(("--initial-policy-std", str(args.initial_policy_std)))
    if args.allow_existing_output:
        values.append("--allow-existing-output")
    return values


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.controller.exists():
        raise SystemExit(f"Real-geometry CEM controller not found: {args.controller}")
    train_mjx_3d_residual_ppo.main(training_argv(args))


if __name__ == "__main__":
    main()
