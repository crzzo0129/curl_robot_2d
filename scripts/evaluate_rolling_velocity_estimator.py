"""Evaluate exported estimator on saved test episodes or independent rollouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.rolling_velocity_estimator import RollingVelocityEstimator, prepare_dataset
from scripts.train_rolling_velocity_estimator import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all-episodes", action="store_true",
                        help="evaluate all episodes of a NEW independent dataset, or explicitly run diagnostics")
    args = parser.parse_args(argv)
    model = RollingVelocityEstimator(args.model)
    data = prepare_dataset(args.data, model.config)
    with np.load(args.model, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"]))
    fingerprint = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.all_episodes:
        rows = np.ones(len(data["x"]), bool)
    else:
        if fingerprint != provenance["dataset_sha256"]:
            parser.error("saved test split requires original data; use --all-episodes for a NEW independent dataset")
        rows = np.isin(data["episode"], provenance["split_episodes"]["test"])
    if not rows.any():
        parser.error("no evaluation windows")
    report, predicted = evaluate(model, data, rows, model.y_mean)
    report["dataset_sha256"] = fingerprint
    report["scope"] = "all_episodes" if args.all_episodes else "saved_test_episodes"
    report["episodes"] = np.unique(data["episode"][rows]).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with np.load(args.data, allow_pickle=False) as archive:
        times = archive["time_s"][data["source_row"][rows]]
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "time_s", "speed_true_m_s", "speed_est_m_s", "speed_valid",
                         "turn_true_rad_s", "turn_est_rad_s", "turn_valid"])
        for episode, t, true, pred, mask in zip(data["episode"][rows], times,
                                                data["y"][rows], predicted, data["mask"][rows]):
            writer.writerow([episode, t, true[0], pred[0], int(mask[0]), true[1], pred[1], int(mask[1])])
    print(json.dumps(report["estimator"], indent=2))


if __name__ == "__main__":
    main()
