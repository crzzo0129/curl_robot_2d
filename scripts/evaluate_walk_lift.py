"""Compare walking lift settings using actual collision-mesh ground clearance."""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.walking_cem import LEGS, MODEL_PATH, SineWalkingController, rollout
from scripts.walk_cem import DEFAULT_POLICY, load_policy, save_json


def evaluate_lift(parameters, metadata, duration):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    controller = SineWalkingController(model, parameters, hold_s=metadata["hold_s"],
        ramp_s=metadata["ramp_s"], heading_feedback=metadata.get("heading_feedback", False))
    foot_geometry = []
    for leg in LEGS:
        geom = model.geom(f"{leg}_foot_proxy").id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        foot_geometry.append((geom, model.mesh_vert[start:start + count].copy()))
    peaks = {}
    def measure(m, d):
        if d.time < controller.hold_s + controller.ramp_s + 1:
            return
        cycle = int((d.time - controller.hold_s) * parameters.frequency_hz)
        heights = [float(np.min(vertices @ d.geom_xmat[g].reshape(3, 3)[2, :]
                                + d.geom_xpos[g, 2])) for g, vertices in foot_geometry]
        peaks[cycle] = np.maximum(peaks.get(cycle, np.full(4, -np.inf)), heights)
    summary, trajectory = rollout(model, controller, duration_s=duration, record=True, callback=measure)
    # Discard the first/last potentially partial cycles.
    cycles = sorted(peaks)
    if len(cycles) > 2:
        clearance = np.median([peaks[c] for c in cycles[1:-1]], axis=0)
        summary["median_cycle_peak_foot_clearance_m"] = dict(zip(LEGS, clearance.tolist()))
    return summary, trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--lifts", type=float, nargs="+", default=[0.014193407167509952, 0.022, 0.028, 0.035])
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/walk_cem_lift"))
    args = parser.parse_args()
    parameters, metadata = load_policy(args.policy)
    results = []
    for lift in args.lifts:
        p = replace(parameters, lift_m=lift)
        summary, trajectory = evaluate_lift(p, metadata, args.duration)
        entry = dict(parameters=asdict(p), **summary)
        results.append(entry)
        name = f"lift_{lift:.6f}"
        save_json(args.out / f"{name}.json", entry)
        np.savez_compressed(args.out / f"{name}.npz", **trajectory)
        print(json.dumps(entry), flush=True)
    save_json(args.out / "comparison.json", results)


if __name__ == "__main__":
    main()
