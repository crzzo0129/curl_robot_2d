"""Collect balanced, unmodified CEM handoffs at requested Roll-to-Stand phases."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS
from curl_robot_2d_mjx.cem_reference import advance_oscillator, load_cem_reference
from curl_robot_2d_mjx.config_transition_3d import (
    Transition3DConfig, transition_curriculum_config_3d,
    transition_physics_profile_3d,
)
from curl_robot_2d_mjx.environment_3d import model_path_3d
from curl_robot_2d_mjx.transition_initialization_3d import save_roll_snapshots_3d
from scripts.evaluate_3d_symmetric_cem_reference import activate_planar_geometry
from scripts.run_handcrafted_roll_to_stand import MODEL, REFERENCE, near_target
from scripts.view_3d_cem_reference import _target_for_phase


GEOMETRY = "rollingquad_2_abd10_no_self_collision"


def _collect_handoff(model, reference, *, minimum_turns, target_pitch_deg=90.0,
                     max_time_s=30.0, replay_until_s=None, preroll_s=0.0):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("compact").id)
    joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES_3D])
    qpos_ids = model.jnt_qposadr[joint_ids]
    actuator_ids = np.asarray([
        model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D
    ])
    low, high = model.actuator_ctrlrange[actuator_ids].T
    data.ctrl[actuator_ids] = _target_for_phase(
        0.0, reference, 1.0, 0.0, 0.25, 0.0, 0.25, 0.0, low, high
    )
    data.qpos[qpos_ids] = data.ctrl[actuator_ids]
    mujoco.mj_forward(model, data)

    torso = model.body("torso").id
    dt = float(model.opt.timestep)
    phase = 0.0
    rolled = 0.0
    previous_in_window = False
    history = deque(maxlen=max(1, int(round(preroll_s / dt)) + 1))
    while data.time < max_time_s:
        if preroll_s > 0:
            history.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()))
        # mj_step's xmat can lag the integrated qpos by one physics step.
        # The free torso quaternion is the state that will actually be restored.
        rotation_flat = np.empty(9)
        mujoco.mju_quat2Mat(rotation_flat, data.qpos[3:7])
        rotation = rotation_flat.reshape(3, 3)
        pitch = math.atan2(rotation[2, 0], rotation[2, 2])
        pitch_rate = -float(data.qvel[4])
        gate_target = math.radians(target_pitch_deg + np.sign(pitch_rate) * 15.0)
        in_window = near_target(
            pitch, pitch_rate, gate_target, math.radians(15.0)
        )
        reached = (data.time >= replay_until_s - dt * .25 if replay_until_s is not None else
                   abs(rolled) >= minimum_turns * 2.0 * math.pi and in_window and not previous_in_window)
        if reached:
            return {
                "history": list(history),
                "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(), "time_s": float(data.time),
                "pitch_deg": math.degrees(pitch),
                "target_pitch_deg": target_pitch_deg,
                "pitch_rate_rad_s": pitch_rate,
                "minimum_turns": minimum_turns,
                "turns": rolled / (2.0 * math.pi),
            }
        previous_in_window = in_window
        phase = float(advance_oscillator(np, rolled, phase, dt, reference))
        data.ctrl[actuator_ids] = _target_for_phase(
            phase, reference, 1.0, 0.0, 0.25, 0.0, 0.25,
            float(data.time), low, high,
        )
        mujoco.mj_step(model, data)
        rolled += float(data.qvel[4]) * dt
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("nonfinite CEM reference rollout")
    raise RuntimeError(f"no +90 degree handoff after {minimum_turns:g} turns")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--xml", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8, help="cycles per target pitch")
    parser.add_argument("--pitch-targets-deg", type=float, nargs="+", default=[90.0],
                        help="balanced real handoffs at each requested pitch (80..100 degrees)")
    parser.add_argument("--first-turn", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0,
                        help="provenance/split id; collection is deterministic")
    args = parser.parse_args(argv)
    if (len(set(args.pitch_targets_deg)) != len(args.pitch_targets_deg)
            or any(not math.isfinite(p) or not 80 <= p <= 100 for p in args.pitch_targets_deg)):
        parser.error("pitch targets must be distinct finite angles in [80, 100]")
    window_bank = args.pitch_targets_deg != [90.0]
    summary_path = args.out.with_suffix(".summary.json")
    if args.samples < 1 or args.first_turn < 1:
        parser.error("samples and first-turn must be positive")
    if args.out.suffix != ".npz" or args.out.exists() or summary_path.exists():
        parser.error("--out must be a new .npz bank and summary path")
    if args.xml.resolve() != model_path_3d(GEOMETRY).resolve():
        parser.error("--xml must remain the abd10 no-self-collision model")

    activate_planar_geometry(PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    reference = load_cem_reference(args.reference)
    task = transition_physics_profile_3d(
        "accurate",
        transition_curriculum_config_3d(
            "brake_full",
            Transition3DConfig(
                geometry=GEOMETRY, dynamic_roll_to_stand=True,
                handcrafted_reference_residual=not window_bank,
                stand_abduction_zero=True, physics_timestep=0.001,
                ready_hold_s=1.0, episode_length=500,
                observation_noise_velocity=0.20,
                observation_noise_gravity=0.05,
                observation_noise_joint_position=0.01,
            ),
        ),
    )
    rows = []
    for offset in range(args.samples):
        for target in sorted(args.pitch_targets_deg):
            row = _collect_handoff(model, reference,
                minimum_turns=args.first_turn + offset, target_pitch_deg=target)
            if abs(row["pitch_deg"] - target) > 1.0:
                raise RuntimeError("handoff missed requested phase by more than one degree")
            rows.append(row)
            print(json.dumps({"sample": len(rows)-1, **{k: row[k] for k in
                  ("time_s", "pitch_deg", "target_pitch_deg", "turns")}}), flush=True)

    digest = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    provenance = (
        f"cem_reference:{args.reference.resolve()}#sha256={digest};"
        "residual=0;trigger_pitch_deg=90;stand_abduction_deg=0"
    )
    if window_bank:
        provenance = (f"cem_reference:{args.reference.resolve()}#sha256={digest};"
                      "residual=0;phase_window_deg=80:100;stand_abduction_deg=0")
    save_roll_snapshots_3d(
        args.out, model, task,
        qpos=[row["qpos"] for row in rows],
        qvel=[row["qvel"] for row in rows],
        ctrl=[row["ctrl"] for row in rows],
        time_s=[row["time_s"] for row in rows],
        episode_id=np.arange(len(rows), dtype=np.int32),
        source_policy=provenance,
    )
    report = {
        "source_kind": ("roll_to_stand_phase_window_reference" if window_bank else
                        "handcrafted_roll_to_stand_90_reference"),
        "status": "ok", "reference_path": str(args.reference.resolve()),
        "reference_sha256": digest, "model": str(args.xml.resolve()),
        "mesh_used": True, "trigger_pitch_deg": None if window_bank else 90.0,
        "deploy_duration_s": None if window_bank else 0.15, "stand_abduction_deg": [0.0] * 4,
        "seed": args.seed, "first_turn": args.first_turn,
        "samples": len(rows), "cycles_per_pitch": args.samples,
        "pitch_targets_deg": sorted(args.pitch_targets_deg),
        "task": asdict(task),
        "handoffs": [{k: row[k] for k in ("time_s", "pitch_deg", "turns",
                     "target_pitch_deg", "pitch_rate_rad_s", "minimum_turns")}
                     for row in rows],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
