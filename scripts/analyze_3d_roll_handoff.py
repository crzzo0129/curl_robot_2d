"""Re-score completed probe CSVs and audit a subsequently supplied config.

Never changes original experiment metadata or raw trial results. A matching
base config does not imply matching runtime or curriculum randomization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig
from curl_robot_2d_mjx.config_3d import Rolling3DConfig
from curl_robot_2d_mjx.handoff_probe_3d import summarize_probes
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig


def read_trials(path):
    with path.open(encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for row in raw:
        decoded = {}
        for key, value in row.items():
            if key == "case":
                decoded[key] = value
            elif value in ("True", "False"):
                decoded[key] = value == "True"
            elif value == "":
                decoded[key] = None
            elif key in ("source_id", "sample_step", "trial_index"):
                decoded[key] = int(value)
            else:
                decoded[key] = float(value)
        rows.append(decoded)
    return rows


def compare_configs(recorded, supplied):
    """Compare effective base settings, filling backward-compatible defaults."""
    diffs = []
    for section, cls in (("task", Rolling3DConfig), ("reference", CEMReferenceConfig),
                         ("reward", Rolling3DRewardConfig)):
        left = json.loads(json.dumps(asdict(cls(**recorded[section]))))
        right = json.loads(json.dumps(asdict(cls(**supplied[section]))))
        for key in sorted(left.keys() | right.keys()):
            if section == "reference" and key == "source":
                continue  # Path provenance only; coefficients are compared.
            if left.get(key) != right.get(key):
                diffs.append({"field": f"{section}.{key}", "recorded": left.get(key),
                              "supplied": right.get(key)})
    defaults = {"reflection_equivariant_policy": False}
    for key in ("hidden_layers", "activation", "zero_residual_policy_init",
                "initial_policy_std", "reflection_equivariant_policy"):
        left, right = recorded.get(key, defaults.get(key)), supplied.get(key, defaults.get(key))
        if left != right:
            diffs.append({"field": key, "recorded": left, "supplied": right})
    return diffs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--out", type=Path, required=True,
                        help="new JSON file; raw experiment files are never overwritten")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; choose a new analysis file")
    if not (args.run / "summary.json").is_file():
        parser.error("run is not completed (summary.json missing)")
    if (args.run / "invalid_replay.json").exists():
        parser.error("invalid exact replay; do not interpret perturbation results")
    metadata = json.loads((args.run / "experiment.json").read_text(encoding="utf-8"))
    rows = read_trials(args.run / "trials.csv")
    errors = [max(r["exact_replay_qpos_max_error"], r["exact_replay_qvel_max_error"])
              for r in rows if r["case"] == "exact"]
    if not errors or not all(math.isfinite(x) for x in errors) or max(errors) > 1e-5:
        parser.error("missing or divergent exact replay")
    report = {
        "run": str(args.run.resolve()), "teacher_sha256": metadata["teacher_sha256"],
        "original_teacher_configuration_assumed": metadata["teacher_configuration_assumed"],
        "model_randomization_during_probe": metadata["model_randomization_during_probe"],
        "sources": json.loads((args.run / "sources.json").read_text(encoding="utf-8")),
        "trial_count": len(rows), "exact_replay_max_error": max(errors),
        "groups": summarize_probes(rows),
        "interpretation": "local continuation sensitivity only; not stand reachability or gate certification",
    }
    if args.teacher_config:
        supplied = json.loads(args.teacher_config.read_text(encoding="utf-8"))
        diffs = compare_configs(metadata["teacher_config_payload"], supplied)
        report["subsequent_config_audit"] = {
            "path": str(args.teacher_config.resolve()),
            "sha256": hashlib.sha256(args.teacher_config.read_bytes()).hexdigest(),
            "effective_base_settings_match": not diffs, "differences": diffs,
            "ignored_reference_source_paths": [metadata["reference"]["source"],
                                               supplied["reference"]["source"]],
            "training_runtime": supplied.get("runtime"),
            "probe_runtime": {k: metadata.get(k) for k in ("mujoco_version", "jax_version", "devices")},
            "curriculum_not_replayed": supplied.get("curriculum"),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(f"trials={len(rows)}, exact_replay_max_error={max(errors):.3g}")
    print("time case trials success failure_free slow_only min_turns max_abs_y")
    for g in report["groups"]:
        print(f"{g['sample_time_s']:.2f} {g['case']:13s} {g['trials']:3d} "
              f"{g['success_rate']:.3f} {g['failure_free_rate']:.3f} "
              f"{g['slow_but_failure_free_rate']:.3f} "
              f"{g['minimum_turns_after_handoff']:.3f} {g['max_abs_y_m']:.4f}")
    if args.teacher_config:
        print(f"effective_base_settings_match={not diffs}")
    print(f"[saved] {args.out}")
    return report


if __name__ == "__main__":
    main()
