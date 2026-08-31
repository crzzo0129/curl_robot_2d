"""Inspect real ROLL cycle coverage and actual speeds without JAX or training."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from curl_robot_2d_mjx.config_transition_3d import transition_curriculum_config_3d
from curl_robot_2d_mjx.transition_snapshot_cli_3d import (
    add_cycle_selection_arguments, apply_cycle_selection_arguments,
)


def inspect_bank(path, task, *, require_coverage=True):
    import mujoco
    from curl_robot_2d_mjx.environment_3d import model_path_3d
    from curl_robot_2d_mjx.transition_initialization_3d import load_roll_snapshots_3d
    model = mujoco.MjModel.from_xml_path(str(model_path_3d(task.geometry)))
    _, report = load_roll_snapshots_3d(path, model, task, return_report=True,
                                      require_coverage=require_coverage)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--stage", choices=("brake_early", "brake_later", "brake_full"),
                        default="brake_early")
    parser.add_argument("--snapshot-tail-fraction", type=float, default=1.0)
    add_cycle_selection_arguments(parser)
    args = parser.parse_args(argv)
    task = apply_cycle_selection_arguments(replace(transition_curriculum_config_3d(args.stage),
               snapshot_tail_fraction=args.snapshot_tail_fraction), args)
    report = inspect_bank(args.bank, task, require_coverage=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["coverage_complete"]:
        raise SystemExit("incomplete cycle/phase coverage; recollect before training")


if __name__ == "__main__":
    main()
