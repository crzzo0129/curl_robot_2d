"""Reproduce historical steering offsets with the high-speed CEM reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts import evaluate_3d_symmetric_cem_reference as evaluator
from curl_robot_2d_mjx.rolling_velocity_estimator import EstimatorConfig, MotionLabeler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    controller = Path("results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/"
                      "01_zero_contact_speed_refine/best_phase_controller.json")
    xml = Path("assets/rollingquad_description_2/mjcf/rollingquad_abd10.xml")
    provenance = dict(
        mujoco_version=mujoco.__version__,
        files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (controller, xml)},
        protocol="Historical evaluator, cg20, compact start, 10 s, target scale 1, "
                 "front abduction -10 deg, rear +10 deg, gain .30, differential scale .25. "
                 "One deterministic rollout per amplitude; self collisions enabled.",
        trajectory_metrics="Existing MotionLabeler with dt=.02 s and EMA tau=.06 s. "
                           "Means exclude t<2 s; no additional smoothing. "
                           "Net course change integrates valid labels over that window.",
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    summaries, paths = [], []
    for raw in (0., .25, -.25, .5, -.5, .75, -.75):
        name = f"raw_{raw:+.2f}".replace("+", "p").replace("-", "n").replace(".", "p")
        motion_path = args.out / f"{name}_motion.npz"
        options = evaluator.parse_args([
            "--xml", str(xml), "--controller", str(controller),
            "--geometry", "rollingquad_2", "--physics-profile", "cg20",
            "--duration", "10", "--control-dt", ".02", "--target-scale", "1",
            "--front-abduction-deg", "-10", "--rear-abduction-deg", "10",
            "--residual-gain", ".30", "--differential-scale", ".25",
            "--differential-residual", str(raw), str(raw), str(raw), str(-raw),
            "--motion-series-out", str(motion_path),
        ])
        print(f"Running raw differential {raw:+.2f}", flush=True)
        result = evaluator.run_smoke(options)
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with np.load(motion_path) as archive:
            motion = {key: archive[key] for key in archive.files}
        dt = result["control_dt_s"]
        assert np.allclose(np.diff(motion["time_s"]), dt)
        labeler = MotionLabeler(EstimatorConfig(control_dt=dt))
        labels, valid = zip(*(labeler.update(v, y) for v, y in
                             zip(motion["velocity_world"], motion["body_y_world"])))
        labels, valid = np.asarray(labels), np.asarray(valid)
        tail = motion["time_s"] >= 2.
        turn_rows = tail & valid[:, 1]
        body_y = motion["body_y_world"]
        heading = np.unwrap(np.arctan2(-body_y[:, 0], body_y[:, 1]))
        selected = np.flatnonzero(tail)
        axis_tail_rate = (heading[selected[-1]] - heading[selected[0]]) / (
            motion["time_s"][selected[-1]] - motion["time_s"][selected[0]])
        row = dict(
            raw_amplitude=raw, normalized_offset=raw * .30 * .25,
            axis_rate_full_rad_s=result["rolling_axis_heading_rate_rad_s"],
            axis_turn_full_deg=float(np.rad2deg(result["rolling_axis_heading_change_rad"])),
            axis_rate_after2s_rad_s=float(axis_tail_rate),
            trajectory_rate_after2s_rad_s=float(labels[turn_rows, 1].mean()),
            trajectory_turn_after2s_deg=float(np.rad2deg(labels[turn_rows, 1].sum() * dt)),
            trajectory_valid_fraction_after2s=float(valid[tail, 1].mean()),
            forward_speed_after2s_m_s=float(labels[tail & valid[:, 0], 0].mean()),
            self_contact_fraction=result["self_contact_fraction"],
            axis_elevation_rms_deg=float(np.rad2deg(result["rolling_axis_elevation_rms_rad"])),
            nonfinite=result["nonfinite"],
        )
        summaries.append(row)
        paths.append((raw, motion, heading))
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for raw, motion, heading in sorted(paths, key=lambda item: item[0]):
            label = f"raw {raw:+.2f}"
            xy = motion["position_world"][:, :2] - motion["position_world"][0, :2]
            axes[0].plot(xy[:, 0], xy[:, 1], label=label)
            axes[1].plot(motion["time_s"], np.rad2deg(heading-heading[0]), label=label)
        axes[0].set(xlabel="World x displacement (m)", ylabel="World y displacement (m)",
                    title="Actual root ground paths (equal spatial scale)")
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[1].set(xlabel="Time (s)", ylabel="Rolling-axis heading change (deg)",
                    title="Heading change; high-speed CEM, gain 0.30")
        for ax in axes:
            ax.grid(alpha=.2)
        axes[1].legend(ncols=2, fontsize=8)
        fig.savefig(args.out / "steering_paths.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
