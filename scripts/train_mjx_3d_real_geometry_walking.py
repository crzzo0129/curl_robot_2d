"""Run an overnight PPO walk training job on the 180 mm real geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import train_mjx_3d_walking_ppo


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(train_mjx_3d_walking_ppo.PRESETS_WALKING_3D),
        default="h200",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--memory-fraction", type=float, default=0.82)
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
        / f"mjx_3d_real_geometry_walking_{args.preset}_seed{args.seed}"
    )
    values = [
        "--preset",
        args.preset,
        "--geometry",
        "real",
        "--recipe",
        "direct_v1",
        "--physics-profile",
        "cg12",
        "--seed",
        str(args.seed),
        "--memory-fraction",
        str(args.memory_fraction),
        "--mujoco-gl",
        args.mujoco_gl,
        "--out",
        str(output),
        "--save-ppo-checkpoints",
        "--skip-evaluation",
    ]
    if args.steps is not None:
        values.extend(("--steps", str(args.steps)))
    if args.allow_existing_output:
        values.append("--allow-existing-output")
    return values


def main(argv=None) -> None:
    args = parse_args(argv)
    train_mjx_3d_walking_ppo.main(training_argv(args))


if __name__ == "__main__":
    main()
