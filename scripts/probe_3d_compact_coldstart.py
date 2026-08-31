"""Real frozen teacher cold-start sweep, NOT stand-to-compact reachability.

Each episode starts near compact at teacher age/phase zero, with the original
reference/residual ramps. Only initial conditions vary; no curriculum physics
randomization, policy training, failure relaxation, or candidate state snap.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from curl_robot_2d_mjx.autonomous_startup_3d import model_fingerprint, sha256
from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, validate_3d_config
from curl_robot_2d_mjx.environment_3d import model_path_3d
from curl_robot_2d_mjx.handoff_probe_3d import FAILURES, HandoffNoise, perturbation_batch
from scripts.probe_3d_roll_handoff import MJXRunner, write_csv, write_json


FIXED_VX = {f"vx_{sign}_{label}": multiplier * value
            for sign, multiplier in (("neg", -1), ("pos", 1))
            for label, value in (("001", .01), ("003", .03), ("005", .05), ("010", .10))}


def sweep_cases(task):
    """Amplitudes are per component, independent uniform [-bound, bound]."""
    def noise(q=task.reset_joint_noise_rad, qd=task.reset_velocity_noise,
              v=0., w=0., tilt=0.):
        return HandoffNoise(q, qd, v, w, tilt, 0., 0.)
    return {
        "exact": noise(0., 0.),
        "training_noise": noise(v=task.reset_root_velocity_noise,
                                w=task.reset_root_velocity_noise, tilt=task.reset_axis_tilt_noise_rad),
        "joint_002": noise(qd=.02),
        "joint_005": noise(qd=.05),
        "joint_010": noise(qd=.10),
        "linear_001": noise(v=.01),
        "linear_003": noise(v=.03),
        "angular_005": noise(w=.05),
        "angular_010": noise(w=.10),
        "combined_low": noise(qd=.02, v=.01, w=.05),
        "combined_medium": noise(qd=.05, v=.03, w=.10),
        "combined_high": noise(qd=.10, v=.05, w=.20),
        "nearcompact_low": noise(q=.01, qd=.02, v=.01, w=.05, tilt=.02),
        **{name: noise() for name in FIXED_VX},
    }


def initial_offsets(group, noise, seed, count):
    offsets = perturbation_batch("state_noise", noise, seed, count)
    if group in FIXED_VX:
        # Same q/qdot draws across +/- tests; isolate world X, all other root
        # velocity components exactly zero. This is not an added random bound.
        offsets["dv"][:] = 0
        offsets["dv"][:, 0] = FIXED_VX[group]
    return offsets


def report_rows(first, current, max_y, max_axis, group, horizon_s, dt, minimum_turns):
    translation = ((current["qpos"][:, 0] - first["qpos"][:, 0])
                   / (2 * np.pi * first["radius"]))
    rotation = (current["absolute_rotation"] - first["absolute_rotation"]) / (2 * np.pi)
    signed = (current["rolling_phase"] - first["rolling_phase"]) / (2 * np.pi)
    turns = np.minimum(translation, rotation)
    rows = []
    for i in range(len(turns)):
        full = bool(current["time"][i] >= horizon_s - dt * .1)
        free = full and not bool(current["failed"][i])
        rows.append({
            "group": group, "trial": i, "horizon_s": horizon_s,
            "failure_free": free, "success": bool(free and turns[i] >= minimum_turns and signed[i] > 0),
            "continued_s": float(current["time"][i]), "turns": float(turns[i]),
            "translation_turns": float(translation[i]), "rotation_turns": float(rotation[i]),
            "signed_turns": float(signed[i]), "end_y_m": float(current["y"][i]),
            "max_abs_y_m": float(max_y[i]), "max_axis_tilt_rad": float(max_axis[i]),
            **{f"failure_{k}": bool(current[f"failure_{k}"][i]) for k in FAILURES},
        })
    return rows


def aggregate(rows):
    return [{"group": group, "episodes": len(selected),
             "failure_free_rate": float(np.mean([r["failure_free"] for r in selected])),
             "success_rate": float(np.mean([r["success"] for r in selected])),
             "mean_turns": float(np.mean([r["turns"] for r in selected])),
             "minimum_turns": min(r["turns"] for r in selected),
             "maximum_abs_y_m": max(r["max_abs_y_m"] for r in selected),
             "failure_rates": {k: float(np.mean([r[f"failure_{k}"] for r in selected])) for k in FAILURES}}
            for group in dict.fromkeys(r["group"] for r in rows)
            for selected in [[r for r in rows if r["group"] == group]]]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--teacher-config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--groups", nargs="+")
    p.add_argument("--duration-s", type=float, default=10.)
    p.add_argument("--minimum-turns", type=float, default=5.)
    p.add_argument("--memory-fraction", type=float, default=.8)
    p.add_argument("--mujoco-gl", default="disable")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.teacher_config = a.teacher_config or a.teacher.parent / "training_config.json"
    for path in (a.teacher, a.teacher_config):
        if not path.is_file():
            p.error(f"missing input: {path}")
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        p.error("output must be new/empty; never overwrite an experiment")
    if a.trials < 1 or not np.isfinite(a.duration_s) or a.duration_s < 3:
        p.error("positive trials and at least 3 seconds required")
    if not np.isfinite(a.minimum_turns) or a.minimum_turns < 0:
        p.error("minimum-turns must be finite and nonnegative")
    return a


def main(argv=None):
    args = parse_args(argv)
    payload = json.loads(args.teacher_config.read_text(encoding="utf-8"))
    original = Rolling3DConfig(**payload["task"])
    if (original.geometry != "rollingquad_2" or original.reset_pose != "compact"
            or original.direct_effective_action or not original.explicit_phase_observation
            or original.lateral_command_enabled or original.lateral_command_fixed not in (None, 0.)
            or original.reset_pair_differential_scale is not None):
        raise ValueError("requires the accepted independent-reset compact straight rolling teacher")
    dt = original.control_timestep
    steps = round(args.duration_s / dt)
    if not np.isclose(steps * dt, args.duration_s, atol=1e-8, rtol=0):
        raise ValueError("duration must align with the control timestep")
    task = replace(original, episode_length=steps + 2, reset_joint_noise_rad=0.,
                   reset_velocity_noise=0., reset_root_velocity_noise=0., reset_axis_tilt_noise_rad=0.)
    validate_3d_config(task)
    cases = sweep_cases(original)
    groups = args.groups or list(cases)
    if len(set(groups)) != len(groups) or any(g not in cases for g in groups):
        raise ValueError(f"unique groups from {list(cases)} required")
    # Exact and training-distribution controls always accompany the sweep.
    groups = list(dict.fromkeys(["exact", "training_noise", *groups]))
    audit_names = ("reset_joint_noise_rad", "reset_velocity_noise", "reset_root_velocity_noise",
                   "reset_pair_differential_scale", "reset_axis_tilt_noise_rad")
    audit = [{"stage": "base", **{k: getattr(original, k) for k in audit_names}}]
    for stage in payload.get("curriculum", {}).get("stages", []):
        settings = Rolling3DConfig(**stage["task"])
        audit.append({"stage": stage["name"], **{k: getattr(settings, k) for k in audit_names}})
    metadata = {
        "status": "planned", "teacher_tested": False, "teacher_sha256": sha256(args.teacher),
        "teacher_config_payload": payload, **model_fingerprint(model_path_3d(task.geometry)),
        "task": asdict(task), "reset_noise_audit": audit,
        "groups": {g: asdict(cases[g]) for g in groups},
        "fixed_root_vx_m_s": {g: FIXED_VX[g] for g in groups if g in FIXED_VX},
        "trials_per_noisy_group": args.trials, "exact_unique_trials": 1, "seed": args.seed,
        "duration_s": args.duration_s, "minimum_success_turns": args.minimum_turns,
        "success_definition": "full horizon, no configured failure, conservative turns >= threshold, positive signed turns",
        "nominal_physics_only": True,
        "sample_distribution": "independent uniform per component, paired draws across groups; fixed_root_vx_m_s overrides all root velocity draws for signed-X groups",
        "abduction_position_and_velocity_noise": 0.,
        "teacher_clock_and_phase": "zero; original cold-start ramps preserved",
        "scope": "compact cold-start sensitivity only; not stand reachability, stationary equilibrium, or a certified handoff gate",
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return metadata
    started = time.perf_counter()
    runner = MJXRunner(task, CEMReferenceConfig(**payload["reference"]), payload,
                       SimpleNamespace(**vars(args), backend="mjx-teacher"))
    import mujoco
    metadata.update(jax_version=runner.jax.__version__, mujoco_version=mujoco.__version__,
                    devices=[str(x) for x in runner.jax.devices()], status="running", teacher_tested=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "experiment.json", metadata)
    print(f"[coldstart] compiling reset, teacher and physics; batch={args.trials}", flush=True)
    reset = runner.reset(args.trials, args.seed)
    baseline = runner.features(reset)
    np.testing.assert_allclose(baseline["qvel"], 0, atol=0)
    for key in ("time", "rolling_phase", "oscillator_phase", "last_action"):
        np.testing.assert_allclose(baseline[key], 0, atol=0)
    reports, rows, checkpoints = [], [], []
    checkpoints_steps = {round(t / dt) for t in (3., 6., args.duration_s) if t <= args.duration_s}
    for group in groups:
        # Use the same random directions across groups to isolate amplitude effects.
        offsets = initial_offsets(group, cases[group], args.seed, args.trials)
        state = runner.branch(reset, np.arange(args.trials), offsets, "exact" if group == "exact" else "state_noise")
        first = runner.features(state)
        # Perturbations only change physical initial conditions, never controller memory.
        for key in ("ctrl", "time", "last_action", "oscillator_phase", "rolling_phase"):
            np.testing.assert_array_equal(first[key], baseline[key])
        n = 1 if group == "exact" else args.trials
        max_y, max_axis = np.abs(first["y"]), first["axis_tilt"].copy()
        traces = {k: [first[k][:n]] for k in ("qpos", "qvel", "ctrl", "time", "rolling_phase")}
        print(f"[coldstart] {group}: n={n}", flush=True)
        group_checkpoints = []
        for i in range(1, steps + 1):
            state = runner.step(state)
            current = runner.features(state)
            max_y, max_axis = np.maximum(max_y, np.abs(current["y"])), np.maximum(max_axis, current["axis_tilt"])
            for k in traces:
                traces[k].append(current[k][:n])
            if i in checkpoints_steps:
                # Intermediate checkpoints report progress, not the final success criterion.
                checkpoint = report_rows(first, current, max_y, max_axis, group, i * dt, dt, args.minimum_turns)[:n]
                if i < steps:
                    for row in checkpoint:
                        row.pop("success")
                group_checkpoints.extend(checkpoint)
            if i % 50 == 0:
                print(f"[coldstart] {group} t={i * dt:g}s alive={int(np.sum(~current['failed'][:n]))}/{n}", flush=True)
        result = report_rows(first, current, max_y, max_axis, group, args.duration_s, dt, args.minimum_turns)[:n]
        rows.extend(result)
        checkpoints.extend(group_checkpoints)
        report = aggregate(result)[0]
        reports.append(report)
        np.savez_compressed(args.out / f"{group}.npz", **{f"offset_{k}": v[:n] for k, v in offsets.items()},
                            **{f"start_{k}": v[:n] for k, v in first.items()},
                            **{f"end_{k}": v[:n] for k, v in current.items()},
                            **{f"trace_{k}": np.stack(v) for k, v in traces.items()})
        write_csv(args.out / "trials.csv", rows)
        write_csv(args.out / "checkpoints.csv", checkpoints)
        write_json(args.out / "partial_summary.json", {**metadata, "results": reports})
        print(f"[result] {group} success={report['success_rate']:.1%} free={report['failure_free_rate']:.1%} "
              f"turns={report['mean_turns']:.3f} min={report['minimum_turns']:.3f} "
              f"max|y|={report['maximum_abs_y_m']:.4f}m", flush=True)
    summary = {**metadata, "status": "completed", "results": reports,
               "total_episodes": len(rows), "elapsed_wall_s": time.perf_counter() - started}
    write_json(args.out / "summary.json", summary)
    print(f"[saved] {args.out / 'summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    main()
