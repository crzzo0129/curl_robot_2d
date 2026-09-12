"""Preview or launch four independent, single-GPU DR experiments on Linux.

Standard library only: previewing never imports the trainer or JAX.
New sweeps share one checkpoint. Continuations restore each run's latest file.
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
SYMMETRY_CASES = (
    # label, front/back loss, action rate, left/right loss, phase reward, balance cost
    ("baseline", 0.01, 0.10, 0.0, 0.0, 0.0),
    ("lr_only", 0.01, 0.10, 0.01, 0.0, 0.0),
    ("cycle_only", 0.01, 0.10, 0.0, 0.05, 0.02),
    ("lr_cycle", 0.01, 0.10, 0.01, 0.05, 0.02),
)


def sweep_cases(name):
    if name == "reward":
        return tuple((*case, 0.0, 0.0, 0.0) for case in CASES)
    if name == "symmetry":
        return SYMMETRY_CASES
    raise ValueError(f"unknown sweep: {name}")


def continuation_sources(path):
    """Resolve all four checkpoints before any output is created or job starts."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("domain_randomization") is not True or manifest.get("terrain") is not False:
        raise ValueError("continuation requires a flat DR reward-sweep manifest")
    old_runs = manifest["runs"]
    if len(old_runs) != 4:
        raise ValueError("continuation requires exactly four recorded runs")
    cases = sweep_cases(manifest.get("sweep", "reward"))
    by_weights = {(run["symmetry_weight"], run["action_rate_weight"],
                   run.get("lr_symmetry_weight", 0.0), run.get("trot_phase_weight", 0.0),
                   run.get("cycle_balance_weight", 0.0)): run for run in old_runs}
    if set(by_weights) != {case[1:] for case in cases}:
        raise ValueError("recorded reward settings do not match the four sweep cases")
    sources = []
    for label, *weights in cases:
        old = by_weights[tuple(weights)]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", old["name"]):
            raise ValueError("invalid source run name")
        directory = PROJECT_ROOT / ("rollingquad_2_deploy_" + old["name"] + "_checkpoints")
        files = [p for p in directory.glob("*.bin") if p.is_file() and p.stem.isdigit()]
        if not files:
            raise FileNotFoundError(f"no numbered checkpoints for {label}: {directory}")
        checkpoint = max(files, key=lambda p: int(p.stem))
        if checkpoint.stat().st_size == 0:
            raise ValueError(f"latest checkpoint is empty: {checkpoint}")
        sources.append((checkpoint, old))
    return path, manifest, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--resume", help="Shared initial checkpoint for a new sweep")
    source.add_argument("--continue-from", help="Prior sweep results directory or manifest.json")
    parser.add_argument("--prefix", default="reward_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--sweep", choices=("reward", "symmetry"),
                        help="reward: original rate/FB grid; symmetry: baseline/LR/cycle/both")
    parser.add_argument("--gpus", help="Four devices; default 0,1,2,3 or the previous assignments")
    parser.add_argument("--num-envs", type=int, help="Environments per experiment; default 1024 or previous value")
    parser.add_argument("--batch-size", type=int, help="Batch size per experiment; default 64 or previous value")
    parser.add_argument("--collision-model", choices=("cad", "foot-spheres"), help="Default cad or previous model")
    parser.add_argument("--launch", action="store_true", help="Start detached jobs; otherwise only print commands")
    args = parser.parse_args()
    previous = {}
    previous_path = None
    if args.continue_from:
        try:
            previous_path, previous, sources = continuation_sources(args.continue_from)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    else:
        sources = [(Path(args.resume).expanduser().resolve(), {}) for _ in CASES]
    if args.sweep is None:
        args.sweep = previous.get("sweep", "reward")
    elif previous and args.sweep != previous.get("sweep", "reward"):
        parser.error("continuation must keep its previous sweep; use --resume for a new comparison")
    cases = sweep_cases(args.sweep)
    if args.num_envs is None:
        args.num_envs = previous.get("num_envs_per_run", 1024)
    if args.batch_size is None:
        args.batch_size = previous.get("batch_size_per_run", 64)
    if args.collision_model is None:
        args.collision_model = previous.get("collision_model", "cad")
    if args.collision_model not in ("cad", "foot-spheres"):
        parser.error("invalid recorded collision model")
    if args.gpus is None:
        args.gpus = ",".join(str(old.get("gpu", i)) for i, (_, old) in enumerate(sources))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("--prefix must contain only letters, digits, underscores or hyphens")
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(gpus):
        parser.error("--gpus must specify four distinct devices")
    if args.num_envs <= 0 or args.batch_size <= 0 or args.batch_size * 32 % args.num_envs:
        parser.error("positive sizes required; batch_size * 32 must be divisible by num_envs")
    output = PROJECT_ROOT / "results" / args.prefix
    runs = []
    for gpu, (label, symmetry, rate, lr, phase, balance), (checkpoint, old) in zip(gpus, cases, sources):
        name = args.prefix + "_" + label
        command = [sys.executable, "-u", "-m", "scripts.train_ppo_deploy", "dr",
                   "--resume", str(checkpoint), "--run-name", name,
                   "--num-envs", str(args.num_envs), "--batch-size", str(args.batch_size),
                   "--fb-symmetry-weight", str(symmetry), "--action-rate-weight", str(rate),
                   "--lr-symmetry-weight", str(lr), "--trot-phase-weight", str(phase),
                   "--cycle-balance-weight", str(balance),
                   "--collision-model", args.collision_model]
        runs.append(dict(name=name, gpu=gpu, symmetry_weight=symmetry,
                         lr_symmetry_weight=lr, trot_phase_weight=phase, cycle_balance_weight=balance,
                         checkpoint=str(checkpoint), source_run=old.get("name"),
                         action_rate_weight=rate, command=command,
                         log=str(output / (label + ".log")), pid=None))
        print(f"GPU {gpu}: FB={symmetry:g}, LR={lr:g}, action_rate={rate:g}, "
              f"trot_phase={phase:g}, cycle_balance={balance:g}")
        print("  " + shlex.join(["env", f"CUDA_VISIBLE_DEVICES={gpu}", *command]))
    if not args.launch:
        print("Preview only. Add --launch on the training server to start all four jobs.")
        return
    if sys.platform != "linux":
        parser.error("--launch is supported on the Linux training server only")
    for checkpoint, old in sources:
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
        # A matching live source process could still be writing its checkpoint.
        if old.get("pid") is not None:
            try:
                command_line = Path(f"/proc/{int(old['pid'])}/cmdline").read_bytes().split(b"\0")
            except FileNotFoundError:
                command_line = []
            if b"--run-name" in command_line:
                i = command_line.index(b"--run-name")
                if i + 1 < len(command_line) and command_line[i + 1].decode() == old["name"]:
                    parser.error(f"source run is still active: {old['name']} (PID {old['pid']})")
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
    manifest = dict(checkpoint=str(sources[0][0]) if not previous else None,
                    sweep=args.sweep,
                    continued_from=str(previous_path) if previous_path is not None else None,
                    domain_randomization=True, terrain=False,
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
