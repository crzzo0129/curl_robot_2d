"""PPO MDP: stand actor -> state gate -> frozen rolling teacher -> outcome.

The actor has full 8-D effective-action authority only during startup. During
the teacher tail actions are ignored; rewards/values still propagate through
the same episode. No candidate qpos/qvel is ever assigned during a handoff.
"""

from dataclasses import replace

import numpy as np

from curl_robot_2d_mjx.autonomous_startup_3d import (
    AUTONOMOUS_STARTUP_OBSERVATION_SIZE, AutonomousStartupConfig, candidate_potential, confirmation_update,
    continuation_score, gate_errors, load_frozen_teacher,
)
from curl_robot_2d_mjx.config_3d import smoothstep_ramp
from curl_robot_2d_mjx.environment_3d import (
    make_brax_env_3d, pair_coupled_residual_action_3d, rolling_target_ctrl_3d,
)
from curl_robot_2d_mjx.cem_reference import advance_oscillator
from curl_robot_2d_mjx.startup_rolling_3d import reset_pose_arrays_3d
from curl_robot_2d_mjx.handoff_probe_3d import FAILURES
from curl_robot_2d_mjx.compact_startup_3d import CompactStartupConfig, compact_target, compact_potential


def make_autonomous_startup_env(task, reference, reward, bank, teacher_path, teacher_payload,
                                config=None, *, seed=0, teacher_policy=None):
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from brax.envs.base import Env, State

    cfg = config or AutonomousStartupConfig()
    compact_mode = isinstance(cfg, CompactStartupConfig)
    dt = task.control_timestep
    cfg.validate(dt)
    if (task.geometry != "rollingquad_2" or task.reset_pose != "compact"
            or task.direct_effective_action or not task.explicit_phase_observation
            or task.lateral_command_enabled or task.lateral_command_fixed not in (None, 0.0)
            or task.lateral_reflex_gain != 0):
        raise ValueError("requires the straight, compact-start, phase-observing residual teacher without reflex")
    if task.reference_ramp_start_scale != 0 or task.reference_phase_rate_scale <= 0:
        raise ValueError("requires the accepted forward teacher with initial zero reference ramp")
    # No timeout inside the base physics env; wrapper owns startup/tail limits.
    task = replace(task, episode_length=cfg.episode_steps(dt) + 2)
    base = make_brax_env_3d(task, cem_reference=reference, reward_config=reward, seed=seed)
    policy = teacher_policy or load_frozen_teacher(base, teacher_path, teacher_payload)
    if compact_mode:
        expected = compact_target(base.mj_model)
        if bank is not None:
            for key in expected:
                if np.shape(bank.get(key)) != expected[key].shape or not np.allclose(bank[key], expected[key], atol=1e-7, rtol=0):
                    raise ValueError("compact startup cannot use the old dynamic candidate bank")
        bank = expected
    targets = {key: jp.asarray(value) for key, value in bank.items()}
    stand_qpos, stand_action = reset_pose_arrays_3d(base.mj_model, replace(task, reset_pose="stand"))
    stand_qpos, stand_action = jp.asarray(stand_qpos), jp.asarray(stand_action)
    stand_ctrl = jp.asarray(base.mj_model.key("stand").ctrl)
    startup_steps = round(cfg.startup_budget_s / dt)
    tail_steps = round(cfg.continuation_s / dt)

    class AutonomousStartupEnv(Env):
        def __init__(self):
            self.base = base
            self.config, self.startup_config = task, cfg
            self.mj_model, self.mjx_model = base.mj_model, base.mjx_model
            self.stand_action = stand_action
            self.episode_length = cfg.episode_steps(dt)
            self.bank = targets

        @property
        def backend(self):
            return "mjx"

        @property
        def observation_size(self):
            return AUTONOMOUS_STARTUP_OBSERVATION_SIZE

        @property
        def action_size(self):
            return 8

        @property
        def sys(self):
            return base.mjx_model

        def _zero_metrics(self):
            names = ("reward", "handoff", "startup_success", "failed", "startup_timeout",
                     "tail_insufficient_progress", "handoff_time_s", "terminal_tail_turns",
                     "terminal_max_abs_y_m", "terminal_gate_error", "startup_control_step",
                     "teacher_control_step", "gate_eligible", "gate_error", "action_change",
                     "command_jump_rad", "shaping", "tail_progress_reward",
                     *[f"failure_{name}" for name in FAILURES])
            if compact_mode:
                names += ("foot_slip_squared_integral", "foot_slip_distance_m", "reward_foot_slip",
                          "reward_settling", "terminal_foot_slip_peak_m_s",
                          "handoff_joint_speed_rad_s", "handoff_root_linear_speed_m_s",
                          "handoff_root_angular_speed_rad_s")
            return {name: jp.zeros((), dtype=jp.float32) for name in names}

        def _unpack(self, state):
            i = state.info
            return State(state.pipeline_state, i["base_obs"], jp.zeros(()), i["base_done"],
                         metrics=i["base_metrics"], info=i["base_info"])

        def _pack_info(self, info, b):
            return {**info, "base_info": b.info, "base_obs": b.obs,
                    "base_metrics": b.metrics, "base_done": b.done}

        def _obs(self, d, info):
            matrix = d.xmat[base.torso_body_id].reshape(3, 3)
            phase = info["base_info"]["rolling_phase"]
            flags = jp.asarray((info["startup_steps"] / startup_steps,
                                info["teacher_active"].astype(jp.float32),
                                info["tail_steps"] / tail_steps,
                                info["confirmation"] / cfg.confirmation_steps))
            return jp.nan_to_num(jp.concatenate((
                jp.asarray((d.qpos[2], d.qpos[1] - info["base_info"]["initial_root_y"],
                            jp.arctan2(-matrix[0, 1], matrix[1, 1]))),
                matrix[:, 1], matrix[:, 2], d.qvel, d.qpos[7:],
                info["base_info"]["last_action"], jp.asarray((jp.sin(phase), jp.cos(phase))), flags)))

        def candidate_match(self, b):
            errors = gate_errors(jp, b.pipeline_state.qpos, b.pipeline_state.qvel,
                                 b.info["rolling_phase"], targets, cfg)
            # Minimax picks a single candidate whose worst dimension is closest.
            index = jp.argmin(jp.max(errors, axis=-1))
            potential = (compact_potential(jp, b.pipeline_state.qpos, b.pipeline_state.qvel, targets, cfg)[0]
                         if compact_mode else jp.max(candidate_potential(jp, errors)))
            return index, errors[index], potential

        def prepare_teacher_context(self, b, index):
            """Only controller context changes; actual physics/counters/history stay."""
            d, info = b.pipeline_state, dict(b.info)
            delta = info["rolling_phase"] - targets["rolling_phase"][index]
            delta = jp.arctan2(jp.sin(delta), jp.cos(delta))
            phase = targets["oscillator_phase"][index] + delta
            age = targets["time"][index]
            if compact_mode:
                # Rebase only the controller's phase, not physical state,
                # absolute roll history, global origins or failure counters.
                info["teacher_phase_offset"] = info["rolling_phase"]
                phase, age = jp.zeros(()), jp.zeros(())
            ref = base._scaled_reference_action_8d(phase, jp.maximum(age - task.physics_timestep, 0))
            info.update(oscillator_phase=phase, reference_time_offset=age - d.time,
                        last_reference_action=ref, direct_action_override=jp.asarray(False))
            obs = base._observation(d, info["last_action"], base._contact_metrics(d),
                axis_tilt=base._rolling_axis_tilt(d), reference_action_value=ref,
                oscillator_phase=phase, rolling_phase=info["rolling_phase"] - info.get("teacher_phase_offset", 0.),
                action_ramp=smoothstep_ramp(jp, jp.maximum(age - task.physics_timestep, 0),
                                           task.startup_action_ramp_s),
                lateral_drift=d.qpos[1] - info["initial_root_y"],
                lateral_velocity_command=info["lateral_velocity_command"])
            return b.replace(info=info, obs=obs)

        def first_teacher_command(self, prepared):
            action = policy(prepared.obs, jax.random.PRNGKey(0))[0]
            coupled = pair_coupled_residual_action_3d(jp, jp.clip(action, -1, 1),
                                                     task.residual_pair_differential_scale)
            info, d = prepared.info, prepared.pipeline_state
            age = d.time + info["reference_time_offset"]
            phase = advance_oscillator(jp, info["rolling_phase"] - info.get("teacher_phase_offset", 0.), info["oscillator_phase"],
                task.physics_timestep, reference, rate_scale=task.reference_phase_rate_scale)
            effective = jp.clip(reference.reference_weight * base._scaled_reference_action_8d(phase, age)
                + smoothstep_ramp(jp, age, task.startup_action_ramp_s) * reference.residual_gain * coupled, -1, 1)
            ctrl = rolling_target_ctrl_3d(jp, base.compact_ctrl, base.actuator_ids, effective,
                                         base.action_scales, base.joint_low, base.joint_high)
            return ctrl, jp.all(jp.isfinite(action))

        def reset(self, rng):
            b = base.reset(rng)
            # Stand is installed ONLY at reset. Preserve the sampled reset noise.
            d = b.pipeline_state
            noise = d.qpos[base.joint_qpos_indices] - base.compact_qpos[base.joint_qpos_indices]
            q = stand_qpos.at[base.joint_qpos_indices].set(jp.clip(
                stand_qpos[base.joint_qpos_indices] + noise, base.joint_low, base.joint_high))
            q = q.at[3:7].set(d.qpos[3:7])
            d = mjx.forward(base.mjx_model, d.replace(qpos=q, ctrl=stand_ctrl))
            bi = {**b.info, "last_action": stand_action, "last_policy_action": stand_action,
                  "reference_time_offset": jp.zeros(()), "direct_action_override": jp.asarray(True)}
            if compact_mode:
                bi.update(teacher_phase_offset=jp.zeros(()), startup_slip_squared=jp.zeros(()),
                          startup_slip_distance=jp.zeros(()), startup_slip_peak=jp.zeros(()))
            b = b.replace(pipeline_state=d, info=bi)
            _, _, potential = self.candidate_match(b)
            zero, integer = jp.zeros(()), jp.asarray(0, dtype=jp.int32)
            info = self._pack_info({
                "teacher_active": jp.asarray(False), "startup_steps": integer,
                "tail_steps": integer, "confirmation": integer, "candidate_id": jp.asarray(-1, jp.int32),
                "potential": potential, "handoff_x": zero, "handoff_rotation": zero,
                "handoff_phase": zero, "tail_turns": zero, "max_abs_y": zero,
                "handoff_time": zero, "terminal": jp.asarray(False),
            }, b)
            if compact_mode:
                info["max_startup_slip"] = zero
            return State(d, self._obs(d, info), zero, zero, metrics=self._zero_metrics(), info=info)

        def step(self, state, action):
            # The full-state autoreset wrapper clears terminal before a new episode.
            return jax.lax.cond(state.info["terminal"],
                lambda _: state.replace(reward=jp.zeros(()), metrics=self._zero_metrics()),
                lambda _: self._step_live(state, action), operand=None)

        def _step_live(self, state, action):
            old = state.info
            active = old["teacher_active"]
            b = self._unpack(state)
            teacher_action = policy(b.obs, jax.random.PRNGKey(0))[0]
            control = jp.where(active, teacher_action, action)
            bi = {**b.info, "direct_action_override": ~active}
            next_b = base.step(b.replace(info=bi), control)
            d = next_b.pipeline_state
            failed_physics = next_b.metrics["failed"] > 0.5
            startup_count = old["startup_steps"] + (~active).astype(jp.int32)
            tail_count = old["tail_steps"] + active.astype(jp.int32)
            index, errors, potential = self.candidate_match(next_b)
            prepared = self.prepare_teacher_context(next_b, index)
            first_ctrl, teacher_finite = self.first_teacher_command(prepared)
            jump = jp.max(jp.abs(first_ctrl - d.ctrl))
            y = d.qpos[1] - next_b.info["initial_root_y"]
            contacts = base._contact_metrics(d)
            eligible = ((jp.max(errors) <= 1) & (jp.abs(y) <= cfg.lateral_m)
                & (base._rolling_axis_tilt(d) <= cfg.axis_tilt_rad)
                & (contacts["forbidden_count"] == 0)
                & (jump <= cfg.first_command_jump_rad) & teacher_finite & ~failed_physics & ~active)
            confirm = confirmation_update(jp, old["candidate_id"], old["confirmation"], index, eligible)
            handoff = (~active & (confirm >= cfg.confirmation_steps) & (startup_count <= startup_steps))
            # Same actual state, never a bank snapshot. Scalar branch also avoids
            # accidentally changing history or phase while still approaching.
            next_b = jax.lax.cond(handoff, lambda _: prepared, lambda _: next_b, None)
            timeout = ~active & ~handoff & (startup_count >= startup_steps)
            hx = jp.where(handoff, d.qpos[0], old["handoff_x"])
            hr = jp.where(handoff, next_b.info["cumulative_rotation"], old["handoff_rotation"])
            hp = jp.where(handoff, next_b.info["rolling_phase"], old["handoff_phase"])
            turns, signed = continuation_score(jp, x=d.qpos[0], start_x=hx,
                rotation=next_b.info["cumulative_rotation"], start_rotation=hr,
                phase=next_b.info["rolling_phase"], start_phase=hp, radius=base.rolling_radius)
            turns = jp.where(active, turns, 0.0)
            tail_complete = active & (tail_count >= tail_steps)
            success = tail_complete & ~failed_physics & (turns >= cfg.minimum_turns) & (signed > 0)
            slow = tail_complete & ~failed_physics & ~success
            terminal = failed_physics | timeout | tail_complete
            failed = terminal & ~success
            change = jp.mean(jp.square(next_b.info["last_action"] - b.info["last_action"]))
            torque = jp.mean(jp.square(d.actuator_force[base.actuator_ids] / base.force_limits))
            shaping = cfg.potential_weight * (cfg.discounting * jp.where(handoff | terminal, 0., potential)
                                              - old["potential"])
            shaping = jp.where(active, 0., shaping)
            tail_reward = jp.where(active, cfg.turn_reward * (turns - old["tail_turns"]), 0.)
            reward_value = (shaping + tail_reward - jp.where(active, 0., cfg.time_cost)
                - cfg.action_change_cost * change - cfg.torque_cost * torque
                + cfg.handoff_bonus * handoff + cfg.success_bonus * success - cfg.failure_cost * failed)
            if compact_mode:
                slip_squared = jp.where(active, 0., next_b.info["startup_slip_squared"])
                slip_distance = jp.where(active, 0., next_b.info["startup_slip_distance"])
                peak_slip = jp.maximum(old["max_startup_slip"], jp.where(active, 0., next_b.info["startup_slip_peak"]))
                slip_penalty = cfg.foot_slip_weight * slip_squared / cfg.foot_slip_sigma_m_s**2
                settling = compact_potential(jp, d.qpos, d.qvel, targets, cfg)[1]
                settling_penalty = jp.where(active, 0., cfg.settling_velocity_weight * settling)
                reward_value -= slip_penalty + settling_penalty
            reward_value = jp.nan_to_num(reward_value, nan=-cfg.failure_cost,
                                         posinf=-cfg.failure_cost, neginf=-cfg.failure_cost)
            max_y = jp.maximum(old["max_abs_y"], jp.abs(y))
            info = self._pack_info({**old, "teacher_active": active | handoff,
                "startup_steps": startup_count, "tail_steps": tail_count,
                "candidate_id": jp.where(active, old["candidate_id"], index),
                "confirmation": jp.where(active, old["confirmation"], confirm),
                "potential": jp.where(active | handoff, 0., potential),
                "handoff_x": hx, "handoff_rotation": hr, "handoff_phase": hp,
                "handoff_time": jp.where(handoff, d.time, old["handoff_time"]),
                "tail_turns": turns, "max_abs_y": max_y, "terminal": terminal}, next_b)
            metrics = {
                "reward": reward_value, "handoff": handoff.astype(jp.float32),
                "startup_success": success.astype(jp.float32), "failed": failed.astype(jp.float32),
                "startup_timeout": timeout.astype(jp.float32),
                "tail_insufficient_progress": slow.astype(jp.float32),
                "handoff_time_s": jp.where(handoff, d.time, 0.),
                "terminal_tail_turns": jp.where(terminal, turns, 0.),
                "terminal_max_abs_y_m": jp.where(terminal, max_y, 0.),
                "terminal_gate_error": jp.where(terminal, jp.max(errors), 0.),
                "startup_control_step": (~active).astype(jp.float32),
                "teacher_control_step": active.astype(jp.float32),
                "gate_eligible": eligible.astype(jp.float32), "gate_error": jp.max(errors),
                "action_change": change, "command_jump_rad": jp.where(handoff, jump, 0.),
                "shaping": shaping, "tail_progress_reward": tail_reward,
                **{f"failure_{name}": next_b.metrics[f"failure_{name}"] for name in FAILURES},
            }
            if compact_mode:
                info["max_startup_slip"] = peak_slip
                metrics.update(foot_slip_squared_integral=slip_squared * dt,
                    foot_slip_distance_m=slip_distance, reward_foot_slip=-slip_penalty,
                    reward_settling=-settling_penalty,
                    terminal_foot_slip_peak_m_s=jp.where(terminal, peak_slip, 0.),
                    handoff_joint_speed_rad_s=jp.where(handoff, jp.max(jp.abs(d.qvel[6:])), 0.),
                    handoff_root_linear_speed_m_s=jp.where(handoff, jp.max(jp.abs(d.qvel[:3])), 0.),
                    handoff_root_angular_speed_rad_s=jp.where(handoff, jp.max(jp.abs(d.qvel[3:6])), 0.))
            metrics = jax.tree_util.tree_map(lambda x: jp.nan_to_num(x), metrics)
            return State(d, self._obs(d, info), reward_value, terminal.astype(jp.float32),
                         metrics=metrics, info=info)

    return AutonomousStartupEnv()


def wrap_autonomous_startup(env, episode_length, action_repeat=1, randomization_fn=None):
    """Reset ALL controller/task state; Brax's default resets only physics/obs.

    New reset RNG each episode, including when the previous episode timed out.
    Preserve terminal metrics/done for PPO/EvalWrapper, with next reset obs.
    """
    import jax
    import jax.numpy as jp
    from brax.envs.base import Wrapper
    from brax.envs.wrappers import training
    if action_repeat != 1 or randomization_fn is not None:
        raise ValueError("startup owns action repeat; model randomization not implemented in v1")

    class FullResetWrapper(Wrapper):
        def reset(self, rng):
            state = self.env.reset(rng)
            state.info["reset_rng"] = rng
            return state

        def step(self, state, action):
            keys = jax.vmap(lambda k: jax.random.split(k, 2))(state.info["reset_rng"])
            next_state = self.env.step(state, action)
            fresh = self.env.reset(keys[:, 0])
            # Preserve outer EvalWrapper metadata if present.
            info = dict(next_state.info)
            def choose(new, old):
                mask = next_state.done.reshape(next_state.done.shape + (1,) * (old.ndim - next_state.done.ndim))
                return jp.where(mask, new, old)
            for key in fresh.info:
                info[key] = jax.tree_util.tree_map(choose, fresh.info[key], next_state.info[key])
            # The terminal step's truncation/length must remain available to PPO/evaluator.
            info["truncation"] = next_state.info["truncation"]
            info["steps"] = next_state.info["steps"]
            info["episode_metrics"] = next_state.info["episode_metrics"]
            info["episode_done"] = next_state.info["episode_done"]
            info["reset_rng"] = keys[:, 1]
            info["needs_step_reset"] = next_state.done > 0
            return next_state.replace(
                pipeline_state=jax.tree_util.tree_map(choose, fresh.pipeline_state, next_state.pipeline_state),
                obs=jax.tree_util.tree_map(choose, fresh.obs, next_state.obs), info=info)

    # Reset step counters BEFORE the first next-episode step, but not before
    # EvalWrapper reads the previous terminal length.
    class CounterResetWrapper(FullResetWrapper):
        def reset(self, rng):
            state = super().reset(rng)
            state.info["needs_step_reset"] = jp.zeros_like(state.done, dtype=bool)
            return state

        def step(self, state, action):
            info = dict(state.info)
            info["steps"] = jp.where(info["needs_step_reset"], 0., info["steps"])
            return super().step(state.replace(info=info, done=jp.zeros_like(state.done)), action)

    return CounterResetWrapper(training.EpisodeWrapper(training.VmapWrapper(env), episode_length, 1))
