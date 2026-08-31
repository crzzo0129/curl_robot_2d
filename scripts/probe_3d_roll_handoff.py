"""Probe takeover sensitivity in the first three seconds of 3-D rolling.

Default backend is the REAL MJX teacher and requires its checkpoint/config.
cpu-reference is an explicitly labelled control experiment, never a teacher
substitute. No startup policy is trained here and no deployment gate is set.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d_mjx.autonomous_startup_3d import model_fingerprint
from curl_robot_2d_mjx.cem_reference import (
    CEMReferenceConfig, CEMReferenceGeometry, advance_oscillator,
    load_cem_reference, reference_action,
)
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d, validate_3d_config
from curl_robot_2d_mjx.environment_3d import (
    apply_physics_options_3d, axis_tilted_quaternion_3d, cem_controller_path_3d,
    duplicate_planar_action_3d, geometry_parameters_3d, model_path_3d,
    reference_startup_scale_3d, rolling_target_ctrl_3d,
)
from curl_robot_2d_mjx.handoff_probe_3d import (
    FAILURES, PROBE_CASES, HandoffNoise, blank_failures, continuation_rows,
    perturbation_batch, sampling_steps, summarize_probes,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjx-teacher", "mjx-reference", "cpu-reference"),
                        default="mjx-teacher")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--assume-accepted-gain-config", action="store_true",
                        help="explicitly use the documented rollingquad_2/cg20 robust_recovery_v15 ABI; report it as assumed")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--sample-every-s", type=float, default=0.2)
    parser.add_argument("--source-duration-s", type=float, default=10.0)
    parser.add_argument("--continuation-s", type=float, default=3.0)
    parser.add_argument("--minimum-source-turns", type=float, default=5.0)
    parser.add_argument("--minimum-turn-rate", type=float, default=0.5,
                        help="minimum conservative turns/s AFTER handoff; not survival alone")
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--trials", type=int, default=32, help="per donor per perturbed case; exact uses one")
    parser.add_argument("--cases", nargs="+", choices=PROBE_CASES, default=list(PROBE_CASES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-noise-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument("--mujoco-gl", default="disable")
    args = parser.parse_args(argv)
    if not 0 < args.window_s <= 3:
        parser.error("--window-s must be in (0,3], the requested candidate window")
    for name in ("source_duration_s", "continuation_s", "minimum_turn_rate", "sample_every_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("minimum_source_turns", "reset_noise_scale", "noise_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.donors < 1 or args.trials < 1:
        parser.error("--donors and --trials must be positive")
    if args.source_duration_s < args.window_s:
        parser.error("source duration must cover the candidate window")
    if len(set(args.cases)) != len(args.cases):
        parser.error("duplicate cases")
    if "exact" not in args.cases:
        parser.error("include exact to verify snapshot replay before interpreting noise results")
    if args.backend == "mjx-teacher":
        if args.teacher is None or not args.teacher.is_file():
            parser.error("mjx-teacher requires the actual --teacher checkpoint")
        if args.assume_accepted_gain_config and args.teacher_config is not None:
            parser.error("choose matching --teacher-config OR --assume-accepted-gain-config")
        if args.teacher_config is None and not args.assume_accepted_gain_config:
            args.teacher_config = args.teacher.parent / "training_config.json"
        if not args.assume_accepted_gain_config and not args.teacher_config.is_file():
            parser.error("supply matching --teacher-config training_config.json")
    elif args.teacher is not None:
        parser.error("reference backends do not load --teacher; use mjx-teacher")
    if args.teacher_config is not None and not args.teacher_config.is_file():
        parser.error("teacher config does not exist")
    if args.assume_accepted_gain_config and args.backend != "mjx-teacher":
        parser.error("--assume-accepted-gain-config is for the teacher backend only")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory is not empty; choose a new experiment directory")
    return args


def experiment_config(args):
    payload = {}
    if args.teacher_config is not None:
        payload = json.loads(args.teacher_config.read_text(encoding="utf-8"))
        task = Rolling3DConfig(**payload["task"])
        reference = CEMReferenceConfig(**payload["reference"])
    else:
        task = physics_profile_3d("cg20", Rolling3DConfig(
            explicit_phase_observation=True, residual_pair_differential_scale=0.25))
        reference = load_cem_reference(cem_controller_path_3d("rollingquad_2"),
                                       reference_weight=1, minimum_residual_gain=0.15)
        if args.assume_accepted_gain_config:
            from scripts.train_mjx_3d_residual_ppo import RECIPES_3D
            from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
            payload = {
                "configuration_source": "ASSUMED from documented accepted gain teacher; not embedded in checkpoint",
                "recipe": "robust_recovery_v15", "task": asdict(task), "reference": asdict(reference),
                "reward": asdict(Rolling3DRewardConfig(**RECIPES_3D["robust_recovery_v15"]["reward"])),
                "hidden_layers": [256, 256, 128], "activation": "elu",
                "zero_residual_policy_init": True, "initial_policy_std": 0.10,
                "reflection_equivariant_policy": False,
            }
    # This probes the accepted compact-start teacher, NOT the new interpolation.
    if task.geometry != "rollingquad_2" or task.reset_pose != "compact" or task.direct_effective_action:
        raise ValueError("probe expects the corrected rollingquad_2 compact-start residual teacher")
    if task.lateral_command_enabled or task.lateral_command_fixed not in (None, 0.0):
        raise ValueError("probe expects straight rolling, without lateral commands")
    if task.reference_phase_rate_scale <= 0:
        raise ValueError("this experiment scores forward rolling (positive phase rate)")
    dt = task.control_timestep
    steps = sampling_steps(args.window_s, args.sample_every_s, dt)
    for duration in (args.source_duration_s, args.continuation_s):
        if not np.isclose(round(duration / dt) * dt, duration, atol=1e-8, rtol=0):
            raise ValueError("durations must be exact control-step multiples")
    # Never rewind episode time/counters at takeover. Extend only timeout so
    # every candidate can be followed for the requested horizon.
    task = replace(task, episode_length=math.ceil(max(
        args.source_duration_s, args.window_s + args.continuation_s) / dt) + 2,
        reset_joint_noise_rad=task.reset_joint_noise_rad * args.reset_noise_scale,
        reset_velocity_noise=task.reset_velocity_noise * args.reset_noise_scale,
        reset_root_velocity_noise=task.reset_root_velocity_noise * args.reset_noise_scale,
        reset_axis_tilt_noise_rad=task.reset_axis_tilt_noise_rad * args.reset_noise_scale)
    validate_3d_config(task)
    noise = HandoffNoise(**{k: v * args.noise_scale for k, v in asdict(HandoffNoise()).items()})
    noise.validate()
    return task, reference, payload, noise, steps


def physical_features(data, info, failures, radius, torso, penetration):
    rotation = np.asarray(data.xmat[torso]).reshape(3, 3)
    return {
        "qpos": np.asarray(data.qpos).copy(), "qvel": np.asarray(data.qvel).copy(),
        "ctrl": np.asarray(data.ctrl).copy(), "time": float(data.time),
        "radius": radius, "y": float(data.qpos[1]) - float(info["initial_root_y"]),
        "heading": float(np.arctan2(-rotation[0, 1], rotation[1, 1])),
        "axis_tilt": float(np.arccos(np.clip(abs(rotation[1, 1]), 0, 1))),
        "torque": float(np.max(np.abs(data.actuator_force))),
        "penetration": penetration,
        "oscillator_phase": float(info["oscillator_phase"]),
        "rolling_phase": float(info["rolling_phase"]),
        "absolute_rotation": float(info["cumulative_rotation"]),
        "last_action": np.asarray(info["last_action"]).copy(),
        "failed": any(failures.values()),
        **{f"failure_{name}": bool(failures[name]) for name in FAILURES},
    }


class CPUReference:
    """Actual CPU dynamics with the same CEM equations and termination checks.

    This is a reference-only control experiment, not numerical parity with MJX.
    All copies include complete MuJoCo integration state and controller memory.
    """
    def __init__(self, task, reference, _payload, _args):
        import mujoco
        self.mj = mujoco
        self.task, self.reference = task, reference
        self.model = mujoco.MjModel.from_xml_path(str(model_path_3d(task.geometry)))
        apply_physics_options_3d(self.model, task)
        self.aids = np.array([self.model.actuator(f"{n}_servo").id for n in JOINT_NAMES_3D])
        jids = np.array([self.model.joint(n).id for n in JOINT_NAMES_3D])
        self.qids, self.vids = self.model.jnt_qposadr[jids], self.model.jnt_dofadr[jids]
        self.low, self.high = self.model.jnt_range[jids].T
        self.compact = self.model.key("compact").ctrl.copy()
        self.torso = self.model.body("torso").id
        self.floor = self.model.geom("floor").id
        p = geometry_parameters_3d(task.geometry)
        self.radius = p.shell_contact_radius
        self.geometry = CEMReferenceGeometry(p.torso_length, p.edge_length,
            2 * p.foot_radius, p.upper_length, p.lower_length)
        self.planar_compact = np.array((p.compact_hip_angle, p.compact_knee_angle) * 2)
        self.planar_low = np.array((p.hip.shell_compatible_range[0], p.knee.shell_compatible_range[0]) * 2)
        self.planar_high = np.array((p.hip.shell_compatible_range[1], p.knee.shell_compatible_range[1]) * 2)

    def action(self, phase, elapsed):
        planar = reference_action(np, phase, self.reference, compact_ctrl=self.planar_compact,
            action_scales=np.array((0.8, 1.2) * 2), joint_low=self.planar_low,
            joint_high=self.planar_high, geometry=self.geometry)
        return np.clip(reference_startup_scale_3d(np, elapsed, self.task)
                       * duplicate_planar_action_3d(np, planar), -1, 1)

    def target(self, action):
        return rolling_target_ctrl_3d(np, self.compact, self.aids, action,
                                     np.array(self.task.action_scales), self.low, self.high)

    def reset(self, count, seed):
        states = []
        for index in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            d = self.mj.MjData(self.model)
            self.mj.mj_resetDataKeyframe(self.model, d, self.model.key("compact").id)
            action = self.action(0, 0)
            d.ctrl[:] = self.target(action)
            d.qpos[self.qids] = np.clip(d.ctrl[self.aids] + rng.uniform(
                -self.task.reset_joint_noise_rad, self.task.reset_joint_noise_rad, 8), self.low, self.high)
            d.qvel[:] = 0
            d.qvel[self.vids] = rng.uniform(-self.task.reset_velocity_noise, self.task.reset_velocity_noise, 8)
            d.qvel[:6] = rng.uniform(-self.task.reset_root_velocity_noise,
                                    self.task.reset_root_velocity_noise, 6)
            tilt = rng.uniform(-self.task.reset_axis_tilt_noise_rad, self.task.reset_axis_tilt_noise_rad, 2)
            d.qpos[3:7] = axis_tilted_quaternion_3d(np, d.qpos[3:7], *tilt)
            self.mj.mj_forward(self.model, d)
            info = {"initial_root_y": float(d.qpos[1]), "oscillator_phase": 0.0,
                "rolling_phase": 0.0, "cumulative_rotation": 0.0, "last_action": action,
                "root_low_step_count": 0, "axis_tilt_step_count": 0,
                "forbidden_contact_step_count": 0, "step_count": 0}
            states.append({"data": d, "info": info, "failures": blank_failures()})
        return states

    def clone(self, states):
        return [{"data": copy.copy(s["data"]), "info": copy.deepcopy(s["info"]),
                 "failures": s["failures"].copy()} for s in states]

    def contacts(self, d):
        penetration = max((max(0, -float(c.dist)) for c in d.contact), default=0.0)
        # The corrected model is floor-only. Do not claim self-collision
        # coverage if its contact masks do not produce robot-robot contacts.
        forbidden = [c for c in d.contact if self.floor not in c.geom and c.dist <= 0]
        return penetration, len(forbidden), max((float(-c.dist) for c in forbidden), default=0.0)

    def step(self, states):
        t = self.task
        for s in states:
            if any(s["failures"].values()):
                continue
            d, i, fail = s["data"], s["info"], s["failures"]
            for _ in range(t.action_repeat):
                i["oscillator_phase"] = float(advance_oscillator(np, i["rolling_phase"],
                    i["oscillator_phase"], t.physics_timestep, self.reference,
                    rate_scale=t.reference_phase_rate_scale))
                a = self.action(i["oscillator_phase"], d.time)
                d.ctrl[:] = self.target(self.reference.reference_weight * a)
                self.mj.mj_step(self.model, d)
                i["rolling_phase"] += float(d.qvel[4]) * t.physics_timestep
            i["last_action"] = self.reference.reference_weight * a
            i["cumulative_rotation"] += abs(float(d.qvel[4])) * t.control_timestep
            i["step_count"] += 1
            _, forbidden, depth = self.contacts(d)
            tilt = float(np.arccos(np.clip(abs(d.xmat[self.torso, 4]), 0, 1)))
            for key, active in (("root_low_step_count", t.terminate_root_z_min is not None
                                and d.qpos[2] < t.terminate_root_z_min),
                               ("axis_tilt_step_count", tilt > t.terminate_axis_tilt_rad),
                               ("forbidden_contact_step_count", forbidden > 0)):
                i[key] = i[key] + 1 if active else 0
            fail.update(nonfinite=not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
                                       and np.isfinite(d.qacc).all() and np.isfinite(d.actuator_force).all())
                        or any(w.number for w in d.warning),
                root_low=i["root_low_step_count"] >= math.ceil(t.terminate_root_z_low_duration_s / t.control_timestep),
                root_high=d.qpos[2] > t.terminate_root_z_max,
                lateral_drift=abs(d.qpos[1] - i["initial_root_y"]) > t.terminate_lateral_drift_m,
                axis_tilt=i["axis_tilt_step_count"] >= math.ceil(t.terminate_axis_tilt_duration_s / t.control_timestep),
                forbidden_depth=depth > t.terminate_forbidden_depth_m,
                forbidden_contact=i["forbidden_contact_step_count"] >= math.ceil(
                    t.terminate_forbidden_contact_duration_s / t.control_timestep))
        return states

    def features(self, states):
        rows = [physical_features(s["data"], s["info"], s["failures"], self.radius,
                                  self.torso, self.contacts(s["data"])[0]) for s in states]
        return {key: np.asarray([r[key] for r in rows]) for key in rows[0]}

    def branch(self, snapshot, ids, offsets, case):
        states = self.clone([snapshot[int(i)] for i in ids])
        if case == "exact":
            return states
        for n, s in enumerate(states):
            d, i = s["data"], s["info"]
            if case in ("state_noise", "combined"):
                d.qpos[self.qids] = np.clip(d.qpos[self.qids] + offsets["dq"][n], self.low, self.high)
                d.qvel[self.vids] += offsets["dqd"][n]
                d.qvel[:6] += offsets["dv"][n]
                d.qpos[3:7] = axis_tilted_quaternion_3d(np, d.qpos[3:7], *offsets["daxis"][n])
            i["oscillator_phase"] += float(offsets["dphase"][n])
            i["last_action"] = np.clip(i["last_action"] + offsets["dhistory"][n], -1, 1)
            # History noise is an intentional memory mismatch, not a ctrl change.
            if case in ("state_noise", "combined"):
                self.mj.mj_forward(self.model, d)
        return states

    def snapshot_arrays(self, snapshot):
        spec = self.mj.mjtState.mjSTATE_INTEGRATION
        vectors = []
        for s in snapshot:
            v = np.empty(self.mj.mj_stateSize(self.model, spec))
            self.mj.mj_getState(self.model, s["data"], v, spec)
            vectors.append(v)
        return {"physics_integration_state": np.stack(vectors),
                **{f"info_{k}": np.asarray([s["info"][k] for s in snapshot]) for k in snapshot[0]["info"]}}


class MJXRunner:
    def __init__(self, task, reference, payload, args):
        from curl_robot_2d_mjx.runtime import configure_cloud_runtime
        configure_cloud_runtime(memory_fraction=args.memory_fraction, preallocate=False,
                                xla_triton=False, mujoco_gl=args.mujoco_gl, verbose=False)
        import jax
        import jax.numpy as jp
        from mujoco import mjx
        from curl_robot_2d_mjx.environment_3d import make_brax_env_3d
        from curl_robot_2d_mjx.config_3d import smoothstep_ramp
        from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig

        self.jax, self.jp, self.task = jax, jp, task
        self.env = env = make_brax_env_3d(task, cem_reference=reference, seed=args.seed,
            reward_config=Rolling3DRewardConfig(**payload.get("reward", {})))
        self.model = env.mj_model
        self.policy = None
        if args.backend == "mjx-teacher":
            from brax.io import model as model_io
            from brax.training.acme import running_statistics
            from brax.training.agents.ppo import networks
            from scripts.train_mjx_3d_residual_ppo import _zero_centered_residual_network_factory
            from scripts.train_mjx_ppo import _network_factory
            hidden, activation = payload["hidden_layers"], payload["activation"]
            factory = (_zero_centered_residual_network_factory(hidden, activation,
                payload["initial_policy_std"], reflection_equivariant=payload.get("reflection_equivariant_policy", False))
                if payload["zero_residual_policy_init"] else _network_factory(hidden, activation))
            network = factory(env.observation_size, env.action_size,
                              preprocess_observations_fn=running_statistics.normalize)
            self.policy = networks.make_inference_fn(network)(
                model_io.load_params(args.teacher), deterministic=True)
        self._reset = jax.jit(jax.vmap(env.reset))

        def step_one(s):
            action = (self.policy(s.obs, jax.random.PRNGKey(0))[0]
                      if self.policy is not None else jp.zeros((8,)))
            return jax.lax.cond(s.done < 0.5, lambda _: env.step(s, action), lambda _: s, None)
        self._step = jax.jit(jax.vmap(step_one))

        def features_one(s):
            d, info = s.pipeline_state, s.info
            rotation = d.xmat[env.torso_body_id].reshape(3, 3)
            _, _, distances = env._contact_arrays(d)
            return {
                "qpos": d.qpos, "qvel": d.qvel, "ctrl": d.ctrl, "time": d.time,
                "radius": jp.asarray(env.rolling_radius),
                "y": d.qpos[1] - info["initial_root_y"],
                "heading": jp.arctan2(-rotation[0, 1], rotation[1, 1]),
                "axis_tilt": env._rolling_axis_tilt(d),
                "torque": jp.max(jp.abs(d.actuator_force)),
                "penetration": jp.maximum(0, -jp.min(distances)),
                "oscillator_phase": info["oscillator_phase"], "rolling_phase": info["rolling_phase"],
                "absolute_rotation": info["cumulative_rotation"], "last_action": info["last_action"],
                "failed": s.metrics["failed"] > 0.5,
                **{f"failure_{name}": s.metrics[f"failure_{name}"] > 0.5 for name in FAILURES},
            }
        self._features = jax.jit(jax.vmap(features_one))

        def perturb_one(s, o):
            d, info = s.pipeline_state, dict(s.info)
            qpos = d.qpos.at[env.joint_qpos_indices].set(jp.clip(
                d.qpos[env.joint_qpos_indices] + o["dq"], env.joint_low, env.joint_high))
            qpos = qpos.at[3:7].set(axis_tilted_quaternion_3d(jp, d.qpos[3:7], *o["daxis"]))
            qvel = d.qvel.at[env.joint_dof_indices].add(o["dqd"]).at[:6].add(o["dv"])
            physical_change = (jp.any(o["dq"] != 0) | jp.any(o["dqd"] != 0)
                               | jp.any(o["dv"] != 0) | jp.any(o["daxis"] != 0))
            d = jax.lax.cond(physical_change,
                lambda _: mjx.forward(env.mjx_model, d.replace(qpos=qpos, qvel=qvel)), lambda _: d, None)
            info["oscillator_phase"] = info["oscillator_phase"] + o["dphase"]
            info["last_action"] = jp.clip(info["last_action"] + o["dhistory"], -1, 1)
            # Existing observations describe the last substep's reference/ramp.
            ref_time = jp.maximum(0, d.time - task.physics_timestep)
            info["last_reference_action"] = jp.where(o["dphase"] != 0,
                env._scaled_reference_action_8d(info["oscillator_phase"], ref_time),
                info["last_reference_action"])
            obs = env._observation(d, info["last_action"], env._contact_metrics(d),
                axis_tilt=env._rolling_axis_tilt(d), reference_action_value=info["last_reference_action"],
                oscillator_phase=info["oscillator_phase"], rolling_phase=info["rolling_phase"],
                action_ramp=smoothstep_ramp(jp, ref_time, task.startup_action_ramp_s),
                lateral_drift=d.qpos[1] - info["initial_root_y"],
                lateral_velocity_command=info["lateral_velocity_command"])
            # Preserve global coordinates, time, counters, warm start, and ALL
            # unchanged teacher history; never reset at a proposed handoff.
            return s.replace(pipeline_state=d, info=info, obs=obs)
        self._perturb = jax.jit(jax.vmap(perturb_one))

    def reset(self, count, seed):
        return self._reset(self.jax.random.split(self.jax.random.PRNGKey(seed), count))

    def clone(self, state):
        return state  # Immutable JAX arrays: later steps allocate new states.

    def step(self, state):
        return self._step(state)

    def features(self, state):
        return self.jax.tree_util.tree_map(np.asarray, self.jax.device_get(self._features(state)))

    def branch(self, snapshot, ids, offsets, case):
        state = self.jax.tree_util.tree_map(lambda x: x[self.jp.asarray(ids)], snapshot)
        return state if case == "exact" else self._perturb(state, self.jax.tree_util.tree_map(self.jp.asarray, offsets))

    def snapshot_arrays(self, snapshot):
        # Every dynamic leaf is saved (not only qpos): warm start, ctrl, act,
        # phase, counters, history, obs, metrics. Paths document the versioned
        # tree layout; these are not portable across arbitrary MJX versions.
        leaves, _ = self.jax.tree_util.tree_flatten_with_path(snapshot)
        return {f"leaf_{i:04d}_{self.jax.tree_util.keystr(path)}": np.asarray(value)
                for i, (path, value) in enumerate(leaves)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    task, reference, payload, noise, sample_steps = experiment_config(args)
    # Resolve all prerequisites before creating an output directory.
    runner = (CPUReference if args.backend == "cpu-reference" else MJXRunner)(task, reference, payload, args)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dt = task.control_timestep
    source_steps, horizon = round(args.source_duration_s / dt), round(args.continuation_s / dt)
    check_steps = {s + horizon for s in sample_steps}
    run_steps = max(source_steps, max(check_steps))
    metadata = {
        "backend": args.backend, "teacher_tested": args.backend == "mjx-teacher",
        "teacher_checkpoint": str(args.teacher.resolve()) if args.teacher else None,
        "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest() if args.teacher else None,
        "teacher_config": str(args.teacher_config.resolve()) if args.teacher_config else None,
        "teacher_config_sha256": (hashlib.sha256(args.teacher_config.read_bytes()).hexdigest()
                                  if args.teacher_config else None),
        "teacher_config_payload": payload,
        "teacher_configuration_assumed": args.assume_accepted_gain_config,
        **model_fingerprint(model_path_3d(task.geometry)),
        "task": asdict(task), "reference": asdict(reference), "noise": asdict(noise),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_times_s": [s * dt for s in sample_steps],
        "model_randomization_during_probe": False,
        "physics_state_reset_at_handoff": False, "time_and_global_origin_preserved": True,
        "history_noise_changes_ctrl": False,
        "interpretation": "local continuation sensitivity, NOT stand reachability or a certified takeover gate",
        "cpu_note": "CPU reference uses independent NumPy reset RNG, not seed parity with MJX; torque sampled at control boundaries",
        "source_qualification": "failure-free for source-duration with minimum-source-turns",
        "success_definition": "full continuation horizon without configured failure AND post-handoff conservative turn rate above threshold AND positive signed rotation",
    }
    import mujoco
    metadata["mujoco_version"] = mujoco.__version__
    if isinstance(runner, MJXRunner):
        metadata["jax_version"] = runner.jax.__version__
        metadata["devices"] = [str(x) for x in runner.jax.devices()]
    write_json(args.out / "experiment.json", metadata)
    print(f"[source] {args.backend} donors={args.donors}, sampling 0..{args.window_s:g}s; "
          "first MJX calls may compile", flush=True)
    state = runner.reset(args.donors, args.seed)
    initial = runner.features(state)
    snapshots, expected, candidate_frames = {}, {}, []
    source_end = None
    for step in range(run_steps + 1):
        if step in sample_steps:
            snapshots[step] = runner.clone(state)
            frame = runner.features(state)
            candidate_frames.append(frame)
            np.savez_compressed(args.out / f"snapshot_{step:04d}.npz", **runner.snapshot_arrays(state))
        if step in check_steps:
            expected[step] = runner.features(state)
        if step == source_steps:
            source_end = runner.features(state)
        if step and step % max(1, round(1 / dt)) == 0:
            print(f"[source] t={step * dt:g}s", flush=True)
        if step < run_steps:
            state = runner.step(state)
    source_turns = np.minimum((source_end["absolute_rotation"] - initial["absolute_rotation"]) / (2 * np.pi),
        (source_end["qpos"][:, 0] - initial["qpos"][:, 0]) / (2 * np.pi * initial["radius"]))
    qualified = (~source_end["failed"] & (source_end["time"] >= args.source_duration_s - dt * 0.1)
                 & (source_turns >= args.minimum_source_turns))
    source_rows = [{"source_id": n, "success": bool(qualified[n]), "turns": float(source_turns[n]),
                    "end_y_m": float(source_end["y"][n]),
                    **{f"failure_{k}": bool(source_end[f"failure_{k}"][n]) for k in FAILURES}}
                   for n in range(args.donors)]
    write_json(args.out / "sources.json", source_rows)
    np.savez_compressed(args.out / "candidate_features.npz", sample_steps=np.asarray(sample_steps),
        **{k: np.stack([f[k] for f in candidate_frames]) for k in initial})
    print(f"[source] qualified={int(np.sum(qualified))}/{args.donors}; "
          "failed-source results will be reported separately", flush=True)

    rows = []
    for step in sample_steps:
        snapshot = snapshots[step]
        eligible = np.flatnonzero(~runner.features(snapshot)["failed"])
        if not eligible.size:
            print(f"[probe] {step * dt:g}s: no live donors", flush=True)
            continue
        for case in args.cases:
            count = 1 if case == "exact" else args.trials
            ids = np.repeat(eligible, count)
            # Stable integer indices make runs reproducible across invocations.
            seed = np.random.SeedSequence([args.seed, step, PROBE_CASES.index(case), 727])
            offsets = perturbation_batch(case, noise, seed, len(ids))
            trial = runner.branch(snapshot, ids, offsets, case)
            first = runner.features(trial)
            maxima = {"y": np.abs(first["y"]), "axis_tilt": first["axis_tilt"].copy(),
                      "torque": first["torque"].copy(), "first_command_jump": np.zeros(len(ids))}
            for j in range(horizon):
                trial = runner.step(trial)
                current = runner.features(trial)
                maxima["y"] = np.maximum(maxima["y"], np.abs(current["y"]))
                maxima["axis_tilt"] = np.maximum(maxima["axis_tilt"], current["axis_tilt"])
                maxima["torque"] = np.maximum(maxima["torque"], current["torque"])
                if j == 0:
                    maxima["first_command_jump"] = np.max(np.abs(current["ctrl"] - first["ctrl"]), axis=1)
            new_rows = continuation_rows(first, current, source_ids=ids, source_success=qualified,
                case=case, sample_step=step, dt=dt, horizon_s=args.continuation_s,
                minimum_turn_rate=args.minimum_turn_rate, maxima=maxima,
                exact_expected=expected[step + horizon] if case == "exact" else None)
            rows.extend(new_rows)
            np.savez_compressed(args.out / f"probe_{step:04d}_{case}.npz", source_ids=ids,
                **offsets, **{f"start_{k}": v for k, v in first.items()},
                **{f"end_{k}": v for k, v in current.items()})
            write_csv(args.out / "trials.csv", rows)
            print(f"[probe] t={step * dt:.2f}s {case}: "
                  f"{sum(r['success'] for r in new_rows)}/{len(new_rows)} success", flush=True)
            if case == "exact":
                error = max(max(r["exact_replay_qpos_max_error"], r["exact_replay_qvel_max_error"]) for r in new_rows)
                if not math.isfinite(error) or error > 1e-5:
                    write_json(args.out / "invalid_replay.json", {
                        "sample_step": step, "max_error": error if math.isfinite(error) else None})
                    raise RuntimeError("exact snapshot replay diverged: discard perturbation conclusions")
    report = {**metadata, "source_success_count": int(np.sum(qualified)),
              "groups": summarize_probes(rows), "elapsed_wall_s": time.perf_counter() - started,
              "status": "teacher_continuation_probe_completed" if args.backend == "mjx-teacher" else "reference_control_only_teacher_unverified"}
    write_json(args.out / "summary.json", report)
    write_csv(args.out / "summary.csv", [{k: v for k, v in r.items() if k != "failure_rates"}
                                          for r in report["groups"]])
    print(f"[saved] {args.out / 'summary.json'}; teacher_tested={metadata['teacher_tested']}", flush=True)
    return report


if __name__ == "__main__":
    main()
