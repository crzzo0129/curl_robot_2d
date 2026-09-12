"""Preview or launch four independent, single-GPU DR experiments on Linux.

Standard library only: previewing never imports the trainer or JAX.
All runs restore the same checkpoint; only symmetry/action-rate weights vary.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("sym0_rate008", 0.0, 0.08),
    ("sym0_rate010", 0.0, 0.10),
    ("sym001_rate008", 0.01, 0.08),
    ("sym001_rate010", 0.01, 0.10),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, help="Shared initial checkpoint")
    parser.add_argument("--prefix", default="reward_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gpus", default="0,1,2,3", help="Four distinct physical GPU indices or UUIDs")
    parser.add_argument("--num-envs", type=int, default=1024, help="Environments per experiment")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per experiment")
    parser.add_argument("--collision-model", choices=("cad", "foot-spheres"), default="cad")
    parser.add_argument("--launch", action="store_true", help="Start detached jobs; otherwise only print commands")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("--prefix must contain only letters, digits, underscores or hyphens")
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(gpus):
        parser.error("--gpus must specify four distinct devices")
    if args.num_envs <= 0 or args.batch_size <= 0 or args.batch_size * 32 % args.num_envs:
        parser.error("positive sizes required; batch_size * 32 must be divisible by num_envs")
    checkpoint = Path(args.resume).expanduser().resolve()
    output = PROJECT_ROOT / "results" / args.prefix
    runs = []
    for gpu, (label, symmetry, rate) in zip(gpus, CASES):
        name = args.prefix + "_" + label
        command = [sys.executable, "-u", "-m", "scripts.train_ppo_deploy", "dr",
                   "--resume", str(checkpoint), "--run-name", name,
                   "--num-envs", str(args.num_envs), "--batch-size", str(args.batch_size),
                   "--fb-symmetry-weight", str(symmetry), "--action-rate-weight", str(rate),
                   "--collision-model", args.collision_model]
        runs.append(dict(name=name, gpu=gpu, symmetry_weight=symmetry,
                         action_rate_weight=rate, command=command,
                         log=str(output / (label + ".log")), pid=None))
        print(f"GPU {gpu}: symmetry={symmetry:g}, action_rate={rate:g}")
        print("  " + shlex.join(["env", f"CUDA_VISIBLE_DEVICES={gpu}", *command]))
    if not args.launch:
        print("Preview only. Add --launch on the training server to start all four jobs.")
        return
    if sys.platform != "linux":
        parser.error("--launch is supported on the Linux training server only")
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    # Preflight every destination before starting any child process.
    for run in runs:
        stem = "rollingquad_2_deploy_" + run["name"]
        for suffix in ("_policy.bin", "_checkpoints", "_videos", "_policy.json"):
            candidate = PROJECT_ROOT / (stem + suffix)
            if candidate.exists():
                parser.error(f"output already exists: {candidate}; choose a new --prefix")
    if output.exists():
        parser.error(f"output already exists: {output}; choose a new --prefix")
    output.mkdir(parents=True)
    manifest = dict(checkpoint=str(checkpoint), domain_randomization=True, terrain=False,
                    collision_model=args.collision_model,
                    num_envs_per_run=args.num_envs, batch_size_per_run=args.batch_size,
                    runs=runs)
    manifest_path = output / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    for run in runs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = run["gpu"]
        try:
            with open(run["log"], "x", encoding="utf-8") as log:
                process = subprocess.Popen(run["command"], cwd=PROJECT_ROOT, env=env,
                                           stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            run["pid"] = process.pid
        except OSError as error:
            run["launch_error"] = str(error)
            save_manifest()
            raise RuntimeError(f"Launch interrupted; inspect {manifest_path} for jobs already started") from error
        save_manifest()
        (output / (run["name"] + ".pid")).write_text(str(process.pid) + "\n", encoding="utf-8")
        print(f"Started GPU {run['gpu']}: PID {process.pid}, log {run['log']}")
    print(f"Launch records: {manifest_path}")
    print("Jobs run independently in the background. Inspect each log for training startup/errors.")


if __name__ == "__main__":
    main()
