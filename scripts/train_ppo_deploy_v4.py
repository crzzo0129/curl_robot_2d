#!/usr/bin/env python3
"""Train deploy walking V4 across 0.10-0.60 m/s with deploy DR."""

from __future__ import annotations

import sys

from scripts import train_ppo_deploy_v2 as v2


V4_DEFAULT_ARGUMENTS = (
    "--domain-randomization",
    "--observation-noise",
    "0.30",
    "--train-speed-range",
    "0.10",
    "0.60",
    "--command-deadzone",
    "0.10",
    "--stand-probability",
    "0.35",
    "--straight-yaw-weight",
    "0.50",
    "--straight-yaw-scale-rad-s",
    "0.08",
    "--straight-lateral-weight",
    "0.30",
    "--straight-lateral-scale-m-s",
    "0.05",
    "--heading-drift-weight",
    "0.80",
    "--heading-drift-scale-rad",
    "0.12",
    "--lateral-drift-weight",
    "0.60",
    "--lateral-drift-scale-m",
    "0.10",
    "--speed-error-weight",
    "3.0",
    "--forward-progress-weight",
    "1.0",
    "--fore-aft-pose-weight",
    "0.40",
    "--fore-aft-pose-scale-rad",
    "0.12",
    "--fore-aft-leg-length-weight",
    "1.50",
    "--fore-aft-leg-length-scale-m",
    "0.015",
    "--front-hip-response-weight",
    "75",
    "--action-rate-weight",
    "0.04",
    "--action-rate-scale-rad",
    "0.05",
    "--stand-action-rate-weight",
    "1.0",
    "--stand-action-rate-scale-rad",
    "0.02",
    "--stand-joint-velocity-weight",
    "0.75",
    "--stand-joint-velocity-scale-rad-s",
    "0.15",
    "--stand-body-angular-weight",
    "0.75",
    "--stand-body-angular-scale-rad-s",
    "0.08",
    "--stand-body-linear-weight",
    "0.75",
    "--stand-body-linear-scale-m-s",
    "0.03",
    "--stand-action-weight",
    "0.12",
    "--stand-action-scale-rad",
    "0.12",
    "--stand-height-weight",
    "0.75",
    "--stand-height-scale-m",
    "0.01",
    "--mirror-weight",
    "0.08",
    "--scale-aware-mirror",
    "--normalized-stability-rewards",
    "--smooth-stability-rewards",
    "--reward-clip-min",
    "-10",
    "--selection-speeds",
    "0.10",
    "0.30",
    "0.60",
    "--selection-duration-s",
    "10",
    "--selection-warmup-s",
    "2",
    "--selection-deploy-perturbations",
    "--v4-selection-metrics",
)


def _has_option(arguments, option):
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def parse_args(argv=None):
    user_arguments = list(sys.argv[1:] if argv is None else argv)
    args = v2.parse_args([*V4_DEFAULT_ARGUMENTS, *user_arguments])
    if not _has_option(user_arguments, "--out"):
        args.out = (
            v2.PROJECT_ROOT
            / "results"
            / f"deploy_walk_v4_{args.command_stage}_{args.preset}_seed"
            f"{args.seed}"
        )
    return args


def main(argv=None):
    v2._run(parse_args(argv))


if __name__ == "__main__":
    main()
