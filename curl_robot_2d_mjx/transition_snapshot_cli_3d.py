"""Shared non-JAX CLI settings for real-cycle snapshot selection."""

from dataclasses import replace

from curl_robot_2d_mjx.config_transition_3d import validate_transition_config_3d


def add_cycle_selection_arguments(parser):
    parser.add_argument("--snapshot-min-turns", type=int,
                        help="override lower bound, inclusive, >=1 completed net turn")
    parser.add_argument("--snapshot-max-turns", type=int,
                        help="override upper bound, exclusive; brake_full defaults to all later turns")
    parser.add_argument("--snapshot-roll-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--snapshot-phase-bins", type=int, default=8)


def apply_cycle_selection_arguments(task, args):
    bounds = {name: getattr(args, name) for name in ("snapshot_min_turns", "snapshot_max_turns")
              if getattr(args, name) is not None}
    if bounds and not task.curriculum_stage.startswith("brake_"):
        raise ValueError("snapshot turn overrides are only meaningful for BRAKE stages")
    task = replace(task, **bounds, snapshot_roll_direction=args.snapshot_roll_direction,
                   snapshot_phase_bins=args.snapshot_phase_bins)
    validate_transition_config_3d(task)
    return task
