"""Measure sustained turning separately from per-frame trajectory fluctuations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(data_path, predictions_path, out, settle_s=2., plot=False):
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in
                ("frames", "episode", "time_s", "position_world", "body_y_world", "metadata_json")}
    metadata = json.loads(str(data["metadata_json"]))
    dt = metadata["control_dt"]
    predictions = np.genfromtxt(predictions_path, delimiter=",", names=True)
    rows = []
    for episode in np.unique(predictions["episode"]):
        p = predictions[predictions["episode"] == episode]
        raw_rows = np.flatnonzero(data["episode"] == episode)
        indexes = np.rint(p["time_s"] / dt).astype(int)
        raw_rows = raw_rows[indexes]
        if not np.allclose(data["time_s"][raw_rows], p["time_s"], atol=1e-7):
            raise ValueError("predictions and raw observation timestamps disagree")
        frames = data["frames"][raw_rows]
        gyro = -np.sum(frames[:, :3] * frames[:, 3:6], axis=1)
        for segment in np.unique(p["command_segment"]):
            part = p["command_segment"] == segment
            valid = part & (p["turn_valid"] > 0)
            tail = valid & (p["command_age_s"] >= settle_s)
            if not tail.any():
                continue
            truth, estimate = p["turn_true_rad_s"][tail], p["turn_est_rad_s"][tail]
            mean_truth, mean_est = float(truth.mean()), float(estimate.mean())
            rows.append(dict(episode=int(episode), segment=int(segment),
                speed_command_m_s=float(p["speed_command_m_s"][part][0]),
                turn_command_rad_s=float(p["turn_command_rad_s"][part][0]),
                mean_true_rad_s=mean_truth, mean_est_rad_s=mean_est,
                mean_gyro_rad_s=float(gyro[tail].mean()),
                mean_error_rad_s=mean_est-mean_truth,
                frame_rmse_rad_s=float(np.sqrt(np.mean((estimate-truth)**2))),
                true_fluctuation_rms_rad_s=float(truth.std()),
                full_valid_fraction=float(p["turn_valid"][part].mean()),
                full_valid_net_turn_deg=float(np.rad2deg(p["turn_true_rad_s"][valid].sum()*dt)),
                full_valid_net_error_deg=float(np.rad2deg(
                    (p["turn_est_rad_s"][valid]-p["turn_true_rad_s"][valid]).sum()*dt)),
                tail_samples=int(tail.sum())))
    # Exclude the startup segment. Each subsequent episode/segment has one vote.
    selected = [row for row in rows if row["segment"] > 0]

    def metrics(parts):
        if not parts:
            return dict(segments=0)
        true = np.array([r["mean_true_rad_s"] for r in parts])
        pred = np.array([r["mean_est_rad_s"] for r in parts])
        gyro = np.array([r["mean_gyro_rad_s"] for r in parts])
        strong = np.abs(true) >= .02
        full = [r["full_valid_net_error_deg"] for r in parts if r["full_valid_fraction"] == 1.]
        return dict(segments=len(parts), mean_true_rad_s=float(true.mean()), mean_est_rad_s=float(pred.mean()),
            segment_mean_rmse_rad_s=float(np.sqrt(np.mean((pred-true)**2))),
            segment_mean_mae_rad_s=float(np.mean(np.abs(pred-true))),
            zero_baseline_segment_rmse_rad_s=float(np.sqrt(np.mean(true**2))),
            gyro_baseline_segment_rmse_rad_s=float(np.sqrt(np.mean((gyro-true)**2))),
            actual_turn_segments=int(strong.sum()),
            actual_turn_sign_accuracy=float(np.mean(np.sign(pred[strong]) == np.sign(true[strong]))) if strong.any() else None,
            fully_valid_segments=len(full),
            full_segment_net_angle_rmse_deg=float(np.sqrt(np.mean(np.square(full)))) if full else None)

    report = dict(dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        control_dt=dt, settle_s=settle_s,
        protocol="Exclude startup segment; segment means exclude first settle_s; "
                 "net angles use complete segments only when all labels are valid. "
                 "No training-target changes or additional smoothing.",
        all_segments=metrics(selected),
        by_command_sign={name: metrics([r for r in selected if np.sign(r["turn_command_rad_s"]) == sign])
                         for name, sign in (("straight", 0), ("left", 1), ("right", -1))})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "turn_segments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "turn_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        episode = np.unique(predictions["episode"])[0]
        p = predictions[predictions["episode"] == episode]
        raw = data["episode"] == episode
        xy = data["position_world"][raw, :2]
        axis = data["body_y_world"][raw][0]
        h = np.array([axis[1], -axis[0]]); h /= np.linalg.norm(h)
        xy = (xy-xy[0]) @ np.column_stack((h, [-h[1], h[0]]))
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        axes[0, 0].plot(xy[:, 0], xy[:, 1], color="#1d3557")
        axes[0, 0].set_aspect("equal", adjustable="datalim")
        axes[0, 0].set(xlabel="Initial forward direction (m)", ylabel="Lateral displacement (m)",
                       title=f"Independent episode {int(episode)}: actual ground path")
        for ax, true_key, est_key, valid_key, ylabel in (
            (axes[0, 1], "speed_true_m_s", "speed_est_m_s", "speed_valid", "Forward speed (m/s)"),
            (axes[1, 0], "turn_true_rad_s", "turn_est_rad_s", "turn_valid", "Trajectory turn rate (rad/s)")):
            ax.plot(p["time_s"], np.where(p[valid_key], p[true_key], np.nan), lw=1., color="#1d3557", label="Actual target")
            ax.plot(p["time_s"], np.where(p[valid_key], p[est_key], np.nan), lw=.8, alpha=.8, color="#e76f51", label="Estimator")
            ax.set(xlabel="Time (s)", ylabel=ylabel)
            ax.legend(fontsize=8)
        axes[1, 0].set_title("Existing causal labels; no extra plot smoothing")
        parts = [r for r in rows if r["episode"] == episode and r["segment"] > 0]
        x = np.array([r["segment"] for r in parts])
        for key, label, color in (("mean_true_rad_s", "Actual mean", "#1d3557"),
                                  ("mean_est_rad_s", "Estimated mean", "#e76f51")):
            axes[1, 1].plot(x, [r[key] for r in parts], "o-", label=label, color=color)
        axes[1, 1].plot(x, [r["turn_command_rad_s"] for r in parts], "--", label="Drive command", color="#777777")
        axes[1, 1].set(xlabel="Command segment", ylabel="Mean trajectory turn rate (rad/s)",
                       title=f"Sustained turning: exclude first {settle_s:g} s of each segment")
        axes[1, 1].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.savefig(out / "calibrated_estimator_preview.png", dpi=160)
        plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--settle-s", type=float, default=2.)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.settle_s) or args.settle_s < 0:
        parser.error("settle-s must be finite and nonnegative")
    analyze(args.data, args.predictions, args.out, args.settle_s, args.plot)


if __name__ == "__main__":
    main()
