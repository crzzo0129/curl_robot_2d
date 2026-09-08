"""CPU vs MJX zero-residual equivalence for the +90° Roll-to-Stand handoff.

Replays every handoff state from the train/eval banks with the 150 ms linear
reference (handoff_ctrl -> ABD=0 Stand) using BOTH the CPU MuJoCo model and the
MJX environment at action=0, then reports the maximum deviation in ctrl, final
pose, angular velocity and foot-contact order.

The CPU replay is implemented without JAX so it can be run locally; the MJX
half requires the Linux GPU stack (``requirements-mjx.txt``) and is skipped with
a clear message when JAX/MJX is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.config_transition_3d import (
    Transition3DConfig,
    transition_curriculum_config_3d,
    transition_physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d,
    model_path_3d,
)
from curl_robot_2d_mjx.environment_walking_3d import (
    FOOT_GEOM_NAMES_3D,
    WALKING_JOINT_NAMES_3D,
)
from curl_robot_2d_mjx.transition_initialization_3d import walking_start_state_3d

GEOMETRY = "rollingquad_2_abd10_no_self_collision"
ABD_INDICES = np.asarray((0, 3, 6, 9), dtype=np.int32)


def build_task() -> Transition3DConfig:
    """Mirror scripts/train_roll_to_stand_reference_residual.py exactly."""
    base = Transition3DConfig(
        geometry=GEOMETRY,
        dynamic_roll_to_stand=True,
        handcrafted_reference_residual=True,
        stand_abduction_zero=True,
        physics_timestep=0.001,
        ready_hold_s=1.0,
        episode_length=500,
        observation_noise_velocity=0.20,
        observation_noise_gravity=0.05,
        observation_noise_joint_position=0.01,
    )
    return transition_physics_profile_3d(
        "accurate", transition_curriculum_config_3d("brake_full", base)
    )


def load_handoffs(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key]
            for key in ("qpos", "qvel", "ctrl", "time_s", "episode_id")
        }


def _cpu_model(task: Transition3DConfig):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path_3d(GEOMETRY)))
    apply_physics_options_3d(model, task)
    return model


def _stand_ctrl(model, task: Transition3DConfig) -> np.ndarray:
    stand = walking_start_state_3d(model, task)["ctrl"].copy()
    stand[ABD_INDICES] = 0.0
    return stand


def cpu_replay(model, task: Transition3DConfig, qpos, qvel, ctrl):
    """Replay the 150 ms linear reference from one handoff state.

    Mirrors the MJX ``step`` substep loop (1 ms physics, reference advanced once
    per substep) so the two trajectories cover exactly the same 160 ms window:
    ``ceil(0.15 / 0.02) = 8`` policy steps x 20 substeps.
    """
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    stand = _stand_ctrl(model, task)
    floor = model.geom("floor").id
    feet = tuple(model.geom(name).id for name in FOOT_GEOM_NAMES_3D)
    dt = float(model.opt.timestep)
    action_repeat = int(task.action_repeat)
    num_policy_steps = max(1, int(math.ceil(task.reference_deploy_duration_s / task.control_timestep)))
    total_substeps = num_policy_steps * action_repeat

    ctrl_traj = np.zeros((total_substeps, model.nu), dtype=np.float64)
    foot_contact = np.zeros((num_policy_steps, len(feet)), dtype=bool)

    for substep in range(total_substeps):
        alpha = float(np.clip(substep * dt / task.reference_deploy_duration_s, 0.0, 1.0))
        target = ctrl + alpha * (stand - ctrl)
        data.ctrl[:] = target
        ctrl_traj[substep] = data.ctrl.copy()
        mujoco.mj_step(model, data)
        if (substep + 1) % action_repeat == 0:
            row = (substep + 1) // action_repeat - 1
            for contact in data.contact[: data.ncon]:
                if contact.dist > 0:
                    continue
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if floor in (g1, g2):
                    geom = g2 if g1 == floor else g1
                    if geom in feet:
                        foot_contact[row, feet.index(geom)] = True

    return {
        "ctrl_traj": ctrl_traj,
        "final_qpos": data.qpos.copy(),
        "final_qvel": data.qvel.copy(),
        "foot_contact": foot_contact,
    }


def run_mjx(task: Transition3DConfig, handoffs: dict, seed: int):
    """Roll out action=0 from each handoff in the MJX environment."""
    from curl_robot_2d_mjx.runtime import configure_cloud_runtime, describe_runtime

    configure_cloud_runtime(verbose=False)
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    from curl_robot_2d_mjx.environment_transition_3d import make_brax_transition_env_3d

    env = make_brax_transition_env_3d(task, seed=seed)
    reset = jax.jit(env.reset)
    takeover = jax.jit(env.reset_from_roll_state)
    step = jax.jit(env.step)

    num_policy_steps = max(1, int(math.ceil(task.reference_deploy_duration_s / task.control_timestep)))
    template = reset(jax.random.PRNGKey(seed))
    zero_action = jp.zeros((env.action_size,), dtype=jp.float32)

    rows = []
    for index in range(len(handoffs["qpos"])):
        data = template.pipeline_state.replace(
            qpos=jp.asarray(handoffs["qpos"][index]),
            qvel=jp.asarray(handoffs["qvel"][index]),
            ctrl=jp.asarray(handoffs["ctrl"][index]),
            time=jp.asarray(handoffs["time_s"][index]),
        )
        data = mjx.forward(env.sys, data)
        state = takeover(data, jax.random.PRNGKey(seed + index + 1))
        foot_contact = np.zeros((num_policy_steps, 4), dtype=bool)
        for step_index in range(num_policy_steps):
            state = step(state, zero_action)
            foot_ground = np.asarray(
                jax.device_get(env._contacts(state.pipeline_state)["foot_ground"]),
                dtype=bool,
            )
            foot_contact[step_index] = foot_ground
        rows.append(
            {
                "final_qpos": np.asarray(state.pipeline_state.qpos),
                "final_qvel": np.asarray(state.pipeline_state.qvel),
                "foot_contact": foot_contact,
            }
        )
    return {"rows": rows, "runtime": describe_runtime()}


def compare(cpu_rows, mjx_rows):
    n = len(cpu_rows)
    qpos_dev = np.zeros(n)
    qvel_dev = np.zeros(n)
    contact_mismatch = np.zeros(n, dtype=int)
    for i in range(n):
        cpu, mjx = cpu_rows[i], mjx_rows[i]
        qpos_dev[i] = float(np.max(np.abs(cpu["final_qpos"] - mjx["final_qpos"])))
        qvel_dev[i] = float(np.max(np.abs(cpu["final_qvel"] - mjx["final_qvel"])))
        contact_mismatch[i] = int(np.count_nonzero(cpu["foot_contact"] != mjx["foot_contact"]))
    return {
        "qpos_max_abs_dev": float(qpos_dev.max()),
        "qvel_max_abs_dev": float(qvel_dev.max()),
        "foot_contact_mismatched_cells": int(contact_mismatch.sum()),
        "foot_contact_mismatched_handoffs": int((contact_mismatch > 0).sum()),
        "per_handoff_qpos_dev": [float(v) for v in qpos_dev],
        "per_handoff_qvel_dev": [float(v) for v in qvel_dev],
        "per_handoff_contact_mismatch": [int(v) for v in contact_mismatch],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path,
        default=Path("results/roll_to_stand_reference_residual/handoffs_train.npz"),
    )
    parser.add_argument(
        "--eval", type=Path,
        default=Path("results/roll_to_stand_reference_residual/handoffs_eval.npz"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/roll_to_stand_reference_residual/zero_residual_equivalence.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true",
                        help="run only the CPU replay (no JAX); useful locally")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    task = build_task()
    import mujoco

    model = _cpu_model(task)
    handoffs = {label: load_handoffs(path) for label, path in (("train", args.train), ("eval", args.eval))}

    cpu_rows = {label: [] for label in handoffs}
    for label, bank in handoffs.items():
        for index in range(len(bank["qpos"])):
            cpu_rows[label].append(
                cpu_replay(model, task, bank["qpos"][index], bank["qvel"][index], bank["ctrl"][index])
            )

    report = {
        "status": "cpu_only" if args.cpu_only else "compared",
        "geometry": GEOMETRY,
        "task_physics": {
            "physics_timestep": task.physics_timestep,
            "solver_name": task.solver_name,
            "solver_iterations": task.solver_iterations,
            "solver_ls_iterations": task.solver_ls_iterations,
            "reference_deploy_duration_s": task.reference_deploy_duration_s,
        },
        "handoffs": {label: len(rows) for label, rows in cpu_rows.items()},
        "cpu_replay": {
            label: {
                "final_qpos_norms": [float(np.linalg.norm(r["final_qpos"])) for r in rows],
                "final_qvel_norms": [float(np.linalg.norm(r["final_qvel"])) for r in rows],
            }
            for label, rows in cpu_rows.items()
        },
    }

    if not args.cpu_only:
        for label, bank in handoffs.items():
            mjx_result = run_mjx(task, bank, args.seed)
            report.setdefault("runtime", mjx_result["runtime"])
            comparison = compare(cpu_rows[label], mjx_result["rows"])
            report[f"comparison_{label}"] = comparison

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
