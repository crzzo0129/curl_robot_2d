"""Package an existing cloud PPO run for review; no JAX import or GPU work."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="PPO output directory")
    parser.add_argument("--out", type=Path, required=True, help="new .zip output")
    parser.add_argument("--log", type=Path, help="optional training stdout/stderr log")
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error("run directory does not exist")
    if args.out.exists():
        parser.error("output already exists; use a new zip path")
    if args.log is not None and not args.log.is_file():
        parser.error("log file does not exist")
    project = Path(__file__).resolve().parents[1]
    versions = {}
    for package in ("jax", "jaxlib", "brax", "flax", "optax", "mujoco", "mujoco-mjx", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = []
    for name in ("training_config.json", "metrics_history.json", "summary.json",
                 "fixed_eval_manifest.json", "fixed_eval_history.json", "stopped.json",
                 "best_fixed_checkpoint.json"):
        path = args.run / name
        if path.is_file():
            files.append((path, f"run/{name}"))
    for name in ("scripts/train_mjx_3d_roll_student_dr_ppo.py",
                 "scripts/train_mjx_3d_roll_distillation.py",
                 "curl_robot_2d_mjx/environment_3d.py",
                 "curl_robot_2d_mjx/environment_rolling_student_dr_3d.py",
                 "curl_robot_2d_mjx/wrappers_rolling_student_dr_3d.py",
                 "curl_robot_2d_mjx/rolling_student_snapshot_pool.py",
                 "curl_robot_2d_mjx/rolling_student_dr_ppo_3d.py",
                 "curl_robot_2d_mjx/rolling_ppo_diagnostics.py",
                 "curl_robot_2d_mjx/reward_3d.py",
                 "curl_robot_2d_mjx/config_3d.py",
                 "curl_robot_2d_mjx/steering_calibration.py", "requirements-mjx.txt"):
        path = project / name
        if path.is_file():
            files.append((path, f"source/{name}"))
    signature = None
    try:
        brax = importlib.metadata.distribution("brax")
        for name in ("training/agents/ppo/train.py", "training/agents/ppo/losses.py",
                     "training/agents/ppo/optimizer.py", "training/agents/ppo/networks.py",
                     "training/acting.py", "training/distribution.py", "envs/wrappers/training.py"):
            path = Path(brax.locate_file(f"brax/{name}"))
            if path.is_file():
                files.append((path, f"installed_brax/{name}"))
                if name.endswith("ppo/train.py"):
                    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                                if isinstance(n, ast.FunctionDef) and n.name == "train")
                    signature = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
    except (importlib.metadata.PackageNotFoundError, OSError, SyntaxError, StopIteration):
        pass
    if args.log:
        files.append((args.log, "run/training.log"))
    config_path = args.run / "training_config.json"
    if config_path.is_file():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        calibration = saved_config.get("task", {}).get("steering_calibration_path")
        if calibration:
            calibration_path = Path(calibration)
            if not calibration_path.is_absolute():
                calibration_path = project / calibration_path
            if calibration_path.is_file():
                files.append((calibration_path, "run/steering_calibration.json"))
    checkpoint_files = []
    for name in ("params_final", "student_params"):
        path = args.run / name
        if path.is_file():
            checkpoint_files.append({"path": name, "bytes": path.stat().st_size})
    for directory in (args.run / "checkpoints", args.run / "ppo_checkpoint"):
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    checkpoint_files.append({"path": str(path.relative_to(args.run)), "bytes": path.stat().st_size})
    manifest = {
        "run": str(args.run.resolve()), "python": platform.python_version(),
        "platform": platform.platform(), "versions": versions,
        "ppo_train_parameters": signature, "checkpoint_files": checkpoint_files,
        "source_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path, name in files if name.startswith(("source/", "installed_brax/"))},
        "note": "No policy weights included. Source reflects files at collection time, not necessarily run start.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for path, name in files:
            archive.write(path, name)
    print(f"Saved {args.out}; versions={versions}")
    print("No training, simulator rollout or GPU initialization was performed.")


if __name__ == "__main__":
    main()
