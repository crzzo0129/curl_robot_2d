"""MJX environment: 0.4 m/s walking snapshot -> compact pose, 12-DoF actor.

The episode resets to a real walking state captured with the deploy-interface
walking policy (rollingquad_2_deploy_robust_dr_policy_stable.json at 0.4 m/s).
Observation and action contracts are identical to the deploy neural
controller: 36 dims x 20 frames history in, 12 absolute position targets out
(pose + scale * action).  The terminal gate is pose-only and no rolling
teacher is activated; see walk_compact_3d.py and docs.

Physics parity with the CPU snapshot collector: mesh rollingquad_abd10.xml,
<option> replaced to 0.002 s implicitfast pyramidal Newton 20/10 impratio 10;
contacts left as authored (ground contact on, self-collision off).
"""

from __future__ import annotations

import numpy as np

from curl_robot_2d_mjx.walk_compact_3d import (
    ACTION_SIZE,
    COMMAND_M_S,
    CONTROL_TIMESTEP_S,
    HISTORY_SIZE,
    OBSERVATION_SIZE,
    PHYSICS_TIMESTEP_S,
    SINGLE_OBS_SIZE,
    WalkCompactConfig,
    compact_target_from_keyframe,
    dense_pose_reward,
    excess_height_cost,
    gate_errors,
    policy_actuator_names,
    policy_joint_names,
    pose_cost,
    pose_potential,
    validate_snapshot_bank,
)


def make_walk_compact_env(runtime_xml, snapshot_npz, snapshot_meta,
                          config=None, *, seed=0):
    """Create the walk->compact MJX environment.

    runtime_xml: prepared by walk_compact_3d.prepare_runtime_xml.
    snapshot_npz/snapshot_meta: walking snapshots (collect_walking_start_snapshots).
    """
    import jax
    import jax.numpy as jp
    import mujoco
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf

    cfg = config or WalkCompactConfig()
    cfg.validate(CONTROL_TIMESTEP_S)
    arrays, meta = validate_snapshot_bank(snapshot_npz, snapshot_meta)
    bank = {key: jp.asarray(value) for key, value in arrays.items()}
    from curl_robot_2d_mjx.walk_compact_3d import bank_action_arrays
    action = bank_action_arrays(meta)
    action = {key: jp.asarray(value) for key, value in action.items()}
    n_frames = round(CONTROL_TIMESTEP_S / PHYSICS_TIMESTEP_S)
    if not np.isclose(n_frames * PHYSICS_TIMESTEP_S, CONTROL_TIMESTEP_S):
        raise ValueError("runtime physics timestep does not align to 50 Hz control")

    mj = mujoco.MjModel.from_xml_path(str(runtime_xml))
    if (mj.nq, mj.nv, mj.nu) != (19, 18, ACTION_SIZE):
        raise ValueError(f"expected floating-base 12-DoF model, got {mj.nq} {mj.nv} {mj.nu}")
    actuator_names = tuple(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                           for i in range(mj.nu))
    if actuator_names != policy_actuator_names():
        raise ValueError(f"actuator order mismatch:\n{actuator_names}")
    joint_names = policy_joint_names()
    joint_qpos_idx = np.asarray([mj.jnt_qposadr[mujoco.mj_name2id(
        mj, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names], dtype=np.int32)
    joint_qpos_idx = jp.asarray(joint_qpos_idx)
    key_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, "compact")
    if key_id < 0:
        raise ValueError("compact keyframe not found in the runtime model")
    stand_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id < 0:
        raise ValueError("stand keyframe not found in the runtime model")
    target = compact_target_from_keyframe(
        np.asarray(mj.key_qpos[key_id]), np.asarray(joint_qpos_idx))
    target = {key: jp.asarray(value) for key, value in target.items()}
    stand_z = float(np.asarray(mj.key_qpos[stand_id])[2])
    # Action -> ctrl follows the repo transition convention: nominal = walking
    # default pose, asymmetric per-joint scale that reaches the full joint
    # range (so the compact hip/knee targets are inside [-1, 1] action space).
    nominal = action["default"]
    ctrl_low = jp.asarray(np.asarray(mj.actuator_ctrlrange[:, 0], dtype=np.float32))
    ctrl_high = jp.asarray(np.asarray(mj.actuator_ctrlrange[:, 1], dtype=np.float32))
    scale = jp.maximum(ctrl_high - nominal, nominal - ctrl_low)
    n_snapshots = int(bank["qpos"].shape[0])
    budget_steps = cfg.episode_steps(CONTROL_TIMESTEP_S)
    command = jp.asarray((COMMAND_M_S, 0.0, 0.0))
    desired_z = jp.asarray((0.0, 0.0, 1.0))

    sys = mjcf.load_model(mj)

    from brax import math as brax_math

    class WalkCompactEnv(PipelineEnv):
        """PipelineEnv with flat brax base State (like train_ppo_deploy)."""

        def __init__(self):
            super().__init__(sys=sys, backend="mjx", n_frames=n_frames)
            self.mj_model = mj
            self.config = cfg
            self.seed = seed
            self.budget_steps = budget_steps
            self.joint_qpos_idx = joint_qpos_idx

        @property
        def observation_size(self):
            return OBSERVATION_SIZE

        @property
        def action_size(self):
            return ACTION_SIZE

        @property
        def episode_length(self):
            return budget_steps

        def _frame(self, ps, info):
            inv_rot = brax_math.quat_inv(ps.x.rot[0])
            return jp.concatenate((
                brax_math.rotate(ps.xd.ang[0], inv_rot),               # gyro
                brax_math.rotate(jp.array((0.0, 0.0, -1.0)), inv_rot),  # proj gravity
                info["command"],                                        # unscaled cmd
                desired_z,
                ps.q[self.joint_qpos_idx] - nominal,                    # joint err
                info["last_act"]))

        def _push(self, hist, frame):
            # Newest first; drop the oldest frame (single obs = 36 dims).
            return jp.concatenate((frame, hist[:-36]))

        def _zero_metrics(self):
            names = ("reward", "success", "failed", "timeout", "gate_eligible",
                     "terminal_gate_error", "terminal_gate_joint",
                     "terminal_gate_axis_tilt", "terminal_gate_lateral",
                     "terminal_pose_quality", "pose_quality")
            return {name: jp.zeros((), dtype=jp.float32) for name in names}

        def reset(self, rng):
            key, = jax.random.split(rng, 1)
            index = jax.random.randint(key, (), 0, n_snapshots)
            qpos = bank["qpos"][index]
            qvel = bank["qvel"][index]
            ps = self.pipeline_init(qpos, qvel)
            # Reuse the recorded walking history but clear the previous-action
            # term: the transition actor starts with its own empty action
            # history (same as a controller hot-switch to a fresh policy).
            hist = bank["hist"][index].reshape((HISTORY_SIZE, SINGLE_OBS_SIZE))
            hist = hist.at[:, 24:36].set(0.0).reshape(-1)
            info = {
                "command": command,
                "last_act": jp.zeros(ACTION_SIZE),
                "hist": hist,
                "step": jp.asarray(0, dtype=jp.int32),
                "confirm": jp.asarray(0, dtype=jp.int32),
                "initial_y": qpos[1],
                "terminal": jp.asarray(False),
            }
            obs = jp.nan_to_num(hist)
            return State(ps, obs, jp.zeros(()), jp.zeros(()),
                         metrics=self._zero_metrics(), info=info)

        def step(self, state, action_in):
            return jax.lax.cond(state.info["terminal"],
                lambda _: state.replace(reward=jp.zeros(()), metrics=self._zero_metrics()),
                lambda _: self._step_live(state, action_in), operand=None)

        def _step_live(self, state, action_in):
            old = state.info
            action_in = jp.clip(action_in, -1.0, 1.0)
            ctrl = jp.clip(nominal + action_in * scale, ctrl_low, ctrl_high)
            ps = self.pipeline_step(state.pipeline_state, ctrl)
            frame = self._frame(ps, old)
            hist = self._push(old["hist"], frame)

            # ---------------- pose gate (pose-only, velocities ignored)
            quat = ps.x.rot[0]
            # Rolling-axis tilt: sideways lean of body-Y out of the horizontal,
            # invariant to the forward roll that curling onto the shell implies.
            body_y_world = brax_math.rotate(jp.array((0.0, 1.0, 0.0)), quat)
            axis_tilt = jp.arcsin(jp.clip(jp.abs(body_y_world[2]), 0.0, 1.0))
            joints = ps.q[self.joint_qpos_idx]
            root_z = ps.q[2]
            lateral = ps.q[1] - old["initial_y"]
            errors = gate_errors(jp, joints, axis_tilt, lateral, target, cfg)
            cost = pose_cost(jp, joints, target, cfg)
            quality = pose_potential(jp, joints, target, cfg)
            finite = jp.all(jp.isfinite(ps.q)) & jp.all(jp.isfinite(ps.qd))
            eligible = (jp.max(errors) <= 1.0) & finite
            confirm = jp.where(eligible, old["confirm"] + 1, 0)
            step_count = old["step"] + 1
            success = (confirm >= cfg.confirmation_steps) & finite
            failed = ~finite
            timeout = ~success & ~failed & (step_count >= budget_steps)
            terminal = failed | timeout | success

            # ---------------- rewards
            excess = excess_height_cost(jp, root_z, stand_z=stand_z,
                                        compact_z=target["root_z"], cfg=cfg)
            change = jp.mean(jp.square(action_in - old["last_act"]))
            torque = jp.mean(jp.square(ps.qfrc_actuator[6:] / 3.0))
            reward = (dense_pose_reward(jp, cost, cfg)
                      + cfg.success_bonus * success.astype(jp.float32)
                      - cfg.time_cost - cfg.action_change_cost * change
                      - cfg.torque_cost * torque - excess)
            reward = jp.nan_to_num(reward, nan=-1.0, posinf=-1.0, neginf=-1.0)

            info = {**old, "command": command, "last_act": action_in,
                    "hist": hist, "step": step_count, "confirm": confirm,
                    "terminal": terminal}
            metrics = {
                "reward": reward,
                "success": success.astype(jp.float32),
                "failed": failed.astype(jp.float32),
                "timeout": timeout.astype(jp.float32),
                "gate_eligible": eligible.astype(jp.float32),
                "terminal_gate_error": jp.where(terminal, jp.max(errors), 0.0),
                "terminal_gate_joint": jp.where(terminal, errors[0], 0.0),
                "terminal_gate_axis_tilt": jp.where(terminal, errors[1], 0.0),
                "terminal_gate_lateral": jp.where(terminal, errors[2], 0.0),
                "terminal_pose_quality": jp.where(terminal, quality, 0.0),
                "pose_quality": quality,
            }
            metrics = {k: jp.nan_to_num(v) for k, v in metrics.items()}
            return State(ps, jp.nan_to_num(hist), reward, terminal.astype(jp.float32),
                         metrics=metrics, info=info)

    return WalkCompactEnv()


def wrap_walk_compact(env, episode_length, action_repeat=1, randomization_fn=None):
    """Full autoreset wrapper; brax defaults only reset on truncation."""
    import jax
    import jax.numpy as jp
    from brax.envs.base import Wrapper
    from brax.envs.wrappers import training

    if action_repeat != 1:
        raise ValueError("walk compact owns its control cadence; action_repeat must be 1")
    if randomization_fn is not None:
        raise ValueError("walk compact v1 does not implement domain randomization")

    class FullResetWrapper(Wrapper):
        def reset(self, rng):
            state = self.env.reset(rng)
            state.info["reset_rng"] = rng
            return state

        def step(self, state, action):
            keys = jax.vmap(lambda k: jax.random.split(k, 2))(state.info["reset_rng"])
            next_state = self.env.step(state, action)
            fresh = self.env.reset(keys[:, 0])
            info = dict(next_state.info)

            def choose(new, old):
                mask = next_state.done.reshape(
                    next_state.done.shape + (1,) * (old.ndim - next_state.done.ndim))
                return jp.where(mask, new, old)

            for key in fresh.info:
                info[key] = jax.tree_util.tree_map(
                    choose, fresh.info[key], next_state.info[key])
            info["truncation"] = next_state.info["truncation"]
            info["steps"] = next_state.info["steps"]
            info["episode_metrics"] = next_state.info["episode_metrics"]
            info["episode_done"] = next_state.info["episode_done"]
            info["reset_rng"] = keys[:, 1]
            info["needs_step_reset"] = next_state.done > 0
            return next_state.replace(
                pipeline_state=jax.tree_util.tree_map(
                    choose, fresh.pipeline_state, next_state.pipeline_state),
                obs=jax.tree_util.tree_map(choose, fresh.obs, next_state.obs),
                info=info)

    class CounterResetWrapper(FullResetWrapper):
        def reset(self, rng):
            state = super().reset(rng)
            state.info["needs_step_reset"] = jp.zeros_like(state.done, dtype=bool)
            return state

        def step(self, state, action):
            info = dict(state.info)
            info["steps"] = jp.where(info["needs_step_reset"], 0.0, info["steps"])
            return super().step(state.replace(info=info, done=jp.zeros_like(state.done)),
                                action)

    return CounterResetWrapper(training.EpisodeWrapper(training.VmapWrapper(env),
                                                       episode_length, 1))
