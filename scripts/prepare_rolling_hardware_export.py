"""Export the selected rolling PPO student on the training host; no simulation.

Uses checkpoints/<step>/student_params, which already contains the expanded
12-action actor and its frozen student normalizer. Never exports PPO params.
The result is an inference candidate, not authorization to switch live motors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from scripts.export_rtneural import (
    _activation, _array, _dense_layers, _field, _load_checkpoint,
    _run_layers, _split_params, convert,
)
from curl_robot_2d_mjx.deployment_rolling_3d import HARDWARE_CONTROLLER_JOINT_NAMES_3D


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(config, training, dense):
    if training.get("actor_observation") != "real_controller_36x20":
        raise ValueError("Expected the real_controller_36x20 actor")
    args, task = training["args"], training["task"]
    if not args.get("command_conditioned") or not args.get("rolling_snapshots"):
        raise ValueError("This exporter requires command-conditioned rolling snapshot training")
    if task.get("geometry") != "rollingquad_2_abd10_no_self_collision":
        raise ValueError("Unexpected geometry; review the hardware contract before exporting")
    dt = float(task["physics_timestep"]) * int(task["action_repeat"])
    if not math.isclose(dt, 0.02, abs_tol=1e-9):
        raise ValueError("Expected 50 Hz training")
    if config.get("observation_history") != 20 or config.get("use_imu") is not True:
        raise ValueError("Expected 20 observation frames with IMU enabled")
    if config.get("control_orientation") is not False:
        raise ValueError("Expected fixed desired world-Z observation")
    if [k.shape for _, k, _ in dense] != [(720, 512), (512, 256), (256, 128), (128, 12)]:
        raise ValueError("Expected expanded 720 -> 512 -> 256 -> 128 -> 12 student_params")
    arrays = {key: _array(config[key], key) for key in (
        "action_scale", "default_joint_pos", "joint_lower_limits", "joint_upper_limits")}
    if any(v.shape != (12,) for v in arrays.values()):
        raise ValueError("All joint metadata must contain 12 entries")
    if not np.allclose(arrays["action_scale"], [0, 0.8, 1.2] * 4, rtol=0, atol=1e-6):
        raise ValueError("Unexpected effective motor command scales")
    center, low, high = (arrays[k] for k in (
        "default_joint_pos", "joint_lower_limits", "joint_upper_limits"))
    if np.any(low >= high) or np.any(center < low) or np.any(center > high):
        raise ValueError("Invalid joint limits or action center")
    abd = [0, 3, 6, 9]
    if not np.allclose(center[abd], np.deg2rad([-10, -10, 10, 10]), atol=1e-6):
        raise ValueError("Expected front -10 / rear +10 degree abduction centers")
    if np.any(dense[-1][1][:, abd]) or np.any(dense[-1][2][abd]):
        raise ValueError("Abduction output columns must be exactly zero")
    for key in ("kp", "kd"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"Invalid {key}")


def verify_export(checkpoint, document):
    normalizer, policy = _split_params(checkpoint)
    mean = _array(_field(normalizer, "mean"), "mean").reshape(-1)
    std = _array(_field(normalizer, "std"), "std").reshape(-1)
    # Probe normalization near its learned mean, including nearly constant
    # channels. These are arithmetic probes, NOT physical rollout evidence.
    rng = np.random.default_rng(20260912)
    probes = np.concatenate((mean[None],
        mean + rng.standard_normal((63, 720)).astype(np.float32) * std), axis=0)
    expected = (probes - mean) / std
    dense = _dense_layers(policy)
    for index, (_, kernel, bias) in enumerate(dense):
        expected = _activation("tanh" if index == len(dense) - 1 else "elu",
                               expected @ kernel + bias)
    actual = _run_layers(probes, document["layers"])
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("Nonfinite inference during export verification")
    error = float(np.max(np.abs(expected - actual)))
    if error > 2e-5:
        raise ValueError(f"Export arithmetic differs from checkpoint: max action error={error}")
    return {
        "kind": "synthetic_normalizer_neighborhood_not_robot_observations",
        "samples": len(probes), "max_action_error": error, "tolerance": 2e-5,
        "normalizer_std_min": float(std.min()),
        "normalizer_channels_below_1e_minus_5": np.flatnonzero(std < 1e-5).tolist(),
        "physical_handoff_validated": False, "cpp_runtime_validated": False,
    }, {"observations": probes.tolist(), "expected_actions": expected.tolist(),
        "tolerance": 2e-5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="PPO actor run directory")
    parser.add_argument("--step", type=int, help="Pin a step; default is best_fixed_checkpoint.json")
    parser.add_argument("--out", type=Path, required=True, help="New export directory; creates sibling .zip")
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    archive = out.with_name(out.name + ".zip")
    if out.exists() or archive.exists():
        parser.error("Output directory or ZIP exists; choose a new --out")
    selection = read_json(run / "best_fixed_checkpoint.json")
    step = int(selection["step"] if args.step is None else args.step)
    if step < 0:
        parser.error("Step must be nonnegative")
    # Resolve locally instead of trusting the original host's absolute path.
    student = run / "checkpoints" / f"{step:012d}" / "student_params"
    if not student.is_file():
        parser.error(f"Missing actual actor weights: {student}. Diagnostics ZIPs do not include them.")
    training = read_json(run / "training_config.json")
    config = read_json(run / "controller_config.json")
    records = [r for r in read_json(run / "fixed_eval_history.json") if int(r["step"]) == step]
    if len(records) != 1 or records[0].get("diagnostics_finite") is not True:
        parser.error("Expected one finite fixed evaluation record for the selected checkpoint")
    checkpoint = _load_checkpoint(str(student))
    _, actor = _split_params(checkpoint)
    validate_contract(config, training, _dense_layers(actor))
    document = convert(checkpoint, config, activation="elu", observation_history=20,
                       normalization="batchnorm")
    verification, vectors = verify_export(checkpoint, document)
    task = training["task"]
    requirements = {
        "policy_period_s": 0.02, "single_observation_size": 36,
        "observation_history": 20, "history_order": "newest_first",
        "command_encoding": "raw_vx_vy_yaw_rate", "vy_m_s": 0.0,
        "vx_range_m_s": [task["forward_command_min_m_s"], task["forward_command_max_m_s"]],
        "yaw_abs_nonzero_range_rad_s": [task["turn_command_min_rad_s"], task["turn_command_max_rad_s"]],
        "straight_yaw_rad_s": 0.0, "desired_world_z": [0, 0, 1],
        "last_action": "previous_raw_tanh_action_in_rolling_action_coordinates",
        "action_type": "position", "initial_state": "mature_rolling_after_stand_to_roll",
        "requires_separate_startup_policy": True,
        "requires_live_history_and_action_coordinate_handoff": True,
        "note": "Metadata only; the existing ROS controller does not enforce these requirements.",
    }
    document.update(joint_names=list(HARDWARE_CONTROLLER_JOINT_NAMES_3D),
                    export_contract="rolling_command_ppo_36x20_batchnorm_v1",
                    student_sha256=sha256(student), controller_requirements=requirements)
    model_name = f"rolling_command_ppo_{step:012d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rolling_export_", dir=out.parent) as temporary:
        staging = Path(temporary) / out.name
        staging.mkdir()
        write_json(staging / model_name, document)
        write_json(staging / "controller_config.json", config)
        write_json(staging / "fixed_eval.json", records[0])
        write_json(staging / "verification.json", verification)
        write_json(staging / "inference_vectors.json", vectors)
        write_json(staging / "controller_requirements.json", requirements)
        manifest = {
            "status": "export_candidate_handoff_not_validated",
            "run": str(run), "step": step, "student_params": str(student),
            "student_sha256": sha256(student), "model": model_name,
            "selection": "best_fixed_panel" if args.step is None else "explicit_step",
            "independent_evaluation": False,
            "dr_strength": training["args"]["dr_strength"],
            "source_hashes": {name: sha256(run / name) for name in (
                "controller_config.json", "training_config.json", "fixed_eval_history.json",
                "best_fixed_checkpoint.json")},
            "files_sha256": {p.name: sha256(p) for p in sorted(staging.iterdir())},
        }
        write_json(staging / "manifest.json", manifest)
        # Neither result is exposed until inference and all metadata checks pass.
        zip_path = Path(temporary) / archive.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(staging.iterdir()):
                zipped.write(path, arcname=f"{out.name}/{path.name}")
        shutil.move(str(staging), str(out))
        shutil.move(str(zip_path), str(archive))
    print(f"Selected actor step: {step:,}")
    print(f"Fixed-panel success: {records[0]['success_rate']:.1%} (not independent validation)")
    print(f"Export max action error: {verification['max_action_error']:.9g}")
    print(f"Model: {out / model_name}\nTransfer ZIP: {archive}")
    print("Candidate exported. Robot runtime parity and stand-to-roll handoff still require validation.")


if __name__ == "__main__":
    main()
