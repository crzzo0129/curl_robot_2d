"""Run the complete curl_2d rolling-to-walking transition baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco

from curl_robot_2d.roll_to_walk import (
    DEFAULT_CONTROLLER_PATH,
    RollToWalkConfig,
    load_roll_controller,
    simulate_roll_to_walk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "roll_to_walk"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulate ROLL -> BRAKE -> DEPLOY -> WALK for curl_2d."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER_PATH)
    parser.add_argument("--roll-duration", type=float, default=1.4)
    parser.add_argument("--brake-duration", type=float, default=1.2)
    parser.add_argument("--deploy-duration", type=float, default=1.6)
    parser.add_argument("--walk-duration", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    config = RollToWalkConfig(
        roll_duration_s=args.roll_duration,
        brake_duration_s=args.brake_duration,
        deploy_duration_s=args.deploy_duration,
        walk_duration_s=args.walk_duration,
    )
    result = simulate_roll_to_walk(
        model,
        config,
        load_roll_controller(args.controller),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {**result.summary, "mode_history": result.mode_history},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.no_csv:
        rollout_path = args.output_dir / "rollout.csv"
        with rollout_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(result.columns)
            writer.writerows(result.rows.tolist())
        print(f"rollout={rollout_path}")
    print(f"summary={summary_path}")
    print(json.dumps({**result.summary, "mode_history": result.mode_history}, indent=2))


if __name__ == "__main__":
    main()
