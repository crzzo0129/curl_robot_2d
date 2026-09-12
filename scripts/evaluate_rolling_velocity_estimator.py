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


def write_preview(out, episode_ids, times, target, prediction, mask, commands=None):
    """Diagnostic plot of one complete held-out episode; no label smoothing here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    episode = np.unique(episode_ids)[0]
    selected = episode_ids == episode
    t = times[selected]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, layout="constrained")
    for column, axis in enumerate(axes):
        truth = np.where(mask[selected, column], target[selected, column], np.nan)
        estimate = np.where(mask[selected, column], prediction[selected, column], np.nan)
        axis.plot(t, truth, color="#1d3557", lw=1.3, label="Measured target")
        axis.plot(t, estimate, color="#e76f51", lw=1., alpha=.85, label="Estimator")
        if commands is not None:
            axis.step(t, commands[selected, 0 if column == 0 else 2], where="post",
                      color="#777777", lw=1., ls="--", label="CEM drive command")
        axis.set_ylabel("Forward speed (m/s)" if column == 0 else "Trajectory turn rate (rad/s)")
        axis.grid(alpha=.18)
        axis.legend(loc="upper right", ncols=3, fontsize=8)
    axes[0].set_title(f"Independent CEM estimator evaluation — episode {episode}")
    axes[1].set_xlabel("Time (s)")
    fig.savefig(out / "prediction_preview.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="save first evaluated episode preview; requires matplotlib")
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
        has_commands = all(k in data for k in ("command", "command_segment", "command_age_s"))
        command_fields = (["speed_command_m_s", "turn_command_rad_s", "command_segment", "command_age_s"]
                          if has_commands else [])
        command_values = (np.column_stack((data["command"][rows, 0], data["command"][rows, 2],
                                           data["command_segment"][rows], data["command_age_s"][rows]))
                          if has_commands else None)
        writer.writerow(["episode", "time_s", "speed_true_m_s", "speed_est_m_s", "speed_valid",
                         "turn_true_rad_s", "turn_est_rad_s", "turn_valid", *command_fields])
        for i, (episode, t, true, pred, mask) in enumerate(zip(data["episode"][rows], times,
                                                data["y"][rows], predicted, data["mask"][rows])):
            extra = command_values[i].tolist() if has_commands else []
            writer.writerow([episode, t, true[0], pred[0], int(mask[0]), true[1], pred[1], int(mask[1]), *extra])
    if has_commands:
        ep, segment, age = data["episode"][rows], data["command_segment"][rows], data["command_age_s"][rows]
        truth, valid = data["y"][rows], data["mask"][rows]
        with (args.out / "command_response.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["episode", "segment", "command_speed_m_s", "command_turn_rad_s",
                             "mean_speed_after_0p5s", "mean_turn_after_0p5s", "speed_samples", "turn_samples"])
            for episode in np.unique(ep):
                for part in np.unique(segment[ep == episode]):
                    selected = (ep == episode) & (segment == part)
                    index = np.flatnonzero(selected)[0]
                    settled = selected & (age >= .5)
                    means, counts = [], []
                    for column in range(2):
                        values = truth[settled & valid[:, column], column]
                        means.append(float(values.mean()) if values.size else None)
                        counts.append(len(values))
                    writer.writerow([episode, part, *command_values[index, :2], *means, *counts])
    if args.plot:
        write_preview(args.out, data["episode"][rows], times, data["y"][rows], predicted,
                      data["mask"][rows], data["command"][rows] if has_commands else None)
    print(json.dumps(report["estimator"], indent=2))


if __name__ == "__main__":
    main()
