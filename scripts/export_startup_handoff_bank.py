"""Export intact, successful teacher handoff samples to a small portable JSON."""

import argparse
import json
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.autonomous_startup_3d import CONTRACT
from scripts.analyze_3d_roll_handoff import compare_configs, read_trials


def export_bank(probes, time_s):
    result, candidates = None, []
    for directory in map(Path, probes):
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if not summary["teacher_tested"] or (directory / "invalid_replay.json").exists():
            raise ValueError("a completed, real-teacher, exact-replay-verified probe is required")
        if result is None:
            result = {"contract": CONTRACT, **{key: summary[key] for key in (
                "teacher_sha256", "model_sha256", "teacher_config_payload")},
                "interpretation": "empirical candidates only; NOT certified gates or stand demonstrations",
                "probe_runtime": []}
        elif (result["teacher_sha256"] != summary["teacher_sha256"]
              or result["model_sha256"] != summary["model_sha256"]
              or compare_configs(result["teacher_config_payload"], summary["teacher_config_payload"])):
            raise ValueError("cannot combine different teachers/models/configurations")
        result["probe_runtime"].append({key: summary.get(key) for key in (
            "jax_version", "mujoco_version", "devices", "noise", "args")})
        rows = read_trials(directory / "trials.csv")
        with np.load(directory / "candidate_features.npz", allow_pickle=False) as data:
            index = np.flatnonzero(np.isclose(data["sample_steps"] * summary["task"]["physics_timestep"]
                                             * summary["task"]["action_repeat"], time_s, atol=1e-7))
            if len(index) != 1:
                raise ValueError(f"{directory}: requested time not sampled")
            index = int(index[0])
            for source in range(data["qpos"].shape[1]):
                selected = [r for r in rows if r["source_id"] == source
                            and abs(r["sample_time_s"] - time_s) < 1e-7]
                exact = [r for r in selected if r["case"] == "exact"]
                if not exact or not all(r["source_success"] and r["success"] for r in selected):
                    continue
                errors = [r[k] for r in exact for k in (
                    "exact_replay_qpos_max_error", "exact_replay_qvel_max_error")]
                if not np.isfinite(errors).all() or max(errors) > 1e-5:
                    raise ValueError("divergent exact replay")
                if {r["case"] for r in selected} != {
                        "exact", "state_noise", "phase_noise", "history_noise", "combined"}:
                    raise ValueError("candidate must have all four perturbation cases plus exact")
                candidates.append({"source_run": directory.name, "source_id": source,
                    "probe_trial_count": len(selected), "sample_time_s": time_s,
                    **{key: data[key][index, source].tolist() for key in (
                        "qpos", "qvel", "ctrl", "rolling_phase", "oscillator_phase", "time")}})
    if not candidates:
        raise ValueError("no qualifying candidates; do not silently use failed source states")
    return {**result, "candidates": candidates}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", nargs="+", required=True, type=Path)
    p.add_argument("--time-s", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("output exists; choose a new bank file")
    result = export_bank(args.probe, args.time_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[bank] {len(result['candidates'])} candidates saved to {args.out}")


if __name__ == "__main__":
    main()
