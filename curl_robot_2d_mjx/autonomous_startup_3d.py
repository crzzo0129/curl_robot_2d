"""Stand-to-rolling skill contracts. No fixed folding trajectory or state snap."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONTRACT = "autonomous_stand_to_roll_v1"
AUTONOMOUS_STARTUP_OBSERVATION_SIZE = 53


@dataclass(frozen=True)
class AutonomousStartupConfig:
    startup_budget_s: float = 3.0
    continuation_s: float = 3.0
    confirmation_steps: int = 3
    minimum_turns: float = 1.5
    gate_scale: float = 1.0
    # Provisional training tolerances, NOT a certified deployment gate.
    joint_position_rad: float = 0.10
    joint_velocity_rad_s: float = 1.0
    root_z_m: float = 0.035
    root_linear_velocity_m_s: float = 0.15
    root_angular_velocity_rad_s: float = 1.0
    orientation_rad: float = 0.15
    rolling_phase_rad: float = 0.15
    lateral_m: float = 0.05
    axis_tilt_rad: float = 0.10
    first_command_jump_rad: float = 0.18
    potential_weight: float = 5.0
    time_cost: float = 0.02
    action_change_cost: float = 0.02
    torque_cost: float = 0.005
    handoff_bonus: float = 1.0
    success_bonus: float = 20.0
    failure_cost: float = 10.0
    turn_reward: float = 2.0
    discounting: float = 0.995

    def validate(self, dt):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.startup_budget_s > 3:
            raise ValueError("startup budget must not exceed the requested 3 seconds")
        if not isinstance(self.confirmation_steps, int) or self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be a positive integer")
        if not 0 < self.discounting < 1:
            raise ValueError("discounting must be in (0,1)")
        for duration in (self.startup_budget_s, self.continuation_s):
            if not np.isclose(round(duration / dt) * dt, duration, atol=1e-8, rtol=0):
                raise ValueError("durations must align with the control timestep")
        if self.confirmation_steps > round(self.startup_budget_s / dt):
            raise ValueError("confirmation exceeds startup budget")

    def episode_steps(self, dt):
        return round((self.startup_budget_s + self.continuation_s) / dt)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_errors(xp, qpos, qvel, rolling_phase, bank, cfg):
    """One vector of normalized errors per candidate; x translation is free.

    Keep whole candidates intact: never mix one source's q with another's qd.
    All 12 joint positions/velocities, including passive abduction tracking,
    enter the gate even though the startup actor commands 8 hip/knee channels.
    """
    def max_abs(x):
        return xp.max(xp.abs(x), axis=-1)
    dots = xp.abs(xp.sum(bank["qpos"][:, 3:7] * qpos[3:7], axis=-1))
    orientation = 2 * xp.arccos(xp.clip(dots, 0, 1))
    phase_delta = rolling_phase - bank["rolling_phase"]
    phase_delta = xp.arctan2(xp.sin(phase_delta), xp.cos(phase_delta))
    errors = xp.stack((
        max_abs(qpos[7:] - bank["qpos"][:, 7:]) / cfg.joint_position_rad,
        max_abs(qvel[6:] - bank["qvel"][:, 6:]) / cfg.joint_velocity_rad_s,
        xp.abs(qpos[2] - bank["qpos"][:, 2]) / cfg.root_z_m,
        max_abs(qvel[:3] - bank["qvel"][:, :3]) / cfg.root_linear_velocity_m_s,
        max_abs(qvel[3:6] - bank["qvel"][:, 3:6]) / cfg.root_angular_velocity_rad_s,
        orientation / cfg.orientation_rad,
        xp.abs(phase_delta) / cfg.rolling_phase_rad,
    ), axis=-1) / cfg.gate_scale
    return errors


def candidate_potential(xp, errors):
    # Smooth bounded shaping even far from the candidate. Not a pose trajectory.
    return xp.exp(-0.5 * xp.mean(xp.log1p(xp.square(errors)), axis=-1))


def confirmation_update(xp, previous_id, previous_count, candidate_id, eligible):
    count = xp.where(eligible, xp.where(previous_id == candidate_id, previous_count + 1, 1), 0)
    return count.astype(xp.int32)


def continuation_score(xp, *, x, start_x, rotation, start_rotation,
                       phase, start_phase, radius):
    turns = xp.minimum((rotation - start_rotation) / (2 * xp.pi),
                       (x - start_x) / (2 * xp.pi * radius))
    signed = (phase - start_phase) / (2 * xp.pi)
    return turns, signed


def load_candidate_bank(path, *, teacher_path, teacher_payload, model_path):
    from scripts.analyze_3d_roll_handoff import compare_configs
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != CONTRACT or not payload.get("candidates"):
        raise ValueError("invalid or empty startup candidate bank")
    if payload["teacher_sha256"] != sha256(teacher_path):
        raise ValueError("candidate bank belongs to a different teacher checkpoint")
    if payload["model_sha256"] != sha256(model_path):
        raise ValueError("candidate bank MJCF does not match the current model")
    differences = compare_configs(payload["teacher_config_payload"], teacher_payload)
    if differences:
        raise ValueError(f"candidate bank/config mismatch: {differences}")
    bank = {key: np.asarray([c[key] for c in payload["candidates"]], dtype=np.float32)
            for key in ("qpos", "qvel", "ctrl", "rolling_phase", "oscillator_phase", "time")}
    for key, values in bank.items():
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite candidate {key}")
    count = len(payload["candidates"])
    for key, width in (("qpos", 19), ("qvel", 18), ("ctrl", 12)):
        if bank[key].shape != (count, width):
            raise ValueError(f"candidate {key} shape mismatch")
    for key in ("rolling_phase", "oscillator_phase", "time"):
        if bank[key].shape != (count,):
            raise ValueError(f"candidate {key} shape mismatch")
    if np.any(bank["time"] <= 0.25) or np.any(bank["time"] > 3.001):
        raise ValueError("candidate time must be after the initial ramp and within 3 seconds")
    if not np.allclose(np.linalg.norm(bank["qpos"][:, 3:7], axis=-1), 1, atol=1e-5):
        raise ValueError("candidate quaternion is not normalized")
    return bank, payload


def load_frozen_teacher(env, checkpoint, payload):
    """Load the exact rolling network ABI; never treat it as startup weights."""
    from brax.io import model as model_io
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    from scripts.train_mjx_3d_residual_ppo import _zero_centered_residual_network_factory
    from scripts.train_mjx_ppo import _network_factory
    factory = (_zero_centered_residual_network_factory(
        payload["hidden_layers"], payload["activation"], payload["initial_policy_std"],
        reflection_equivariant=payload.get("reflection_equivariant_policy", False))
        if payload["zero_residual_policy_init"] else
        _network_factory(payload["hidden_layers"], payload["activation"]))
    network = factory(env.observation_size, env.action_size,
                      preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(network)(model_io.load_params(checkpoint), deterministic=True)
