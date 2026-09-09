"""Configuration and curriculum for one-policy stand-to-roll training."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


STAND_TO_ROLL_CURRICULUM_STAGES = (
    "rolling_orbit",
    "mixed_75",
    "mixed_25",
    "compact",
    "slightly_open",
    "crouch",
    "semi_stand",
    "full_stand",
)


@dataclass(frozen=True)
class StandToRollConfig:
    """Nominal task configuration before teacher-shaping annealing.

    The actor always owns all twelve motor targets.  CEM is used only to
    compute distance, progress and a one-shot capture milestone.
    """

    geometry: str = "rollingquad_2_abd10"
    physics_profile: str = "cg20"
    self_collision_enabled: bool = False
    episode_length: int = 500

    # Stand-to-roll action ABI.  This deliberately does not reuse the walking
    # train_ppo_deploy range: that range cannot reach the compact rolling pose.
    action_center_hip_rad: float = 0.1108283051
    action_center_knee_rad: float = 0.9092586986
    action_scale_abduction_rad: float = 0.30
    action_scale_hip_rad: float = 0.80
    action_scale_knee_rad: float = 1.20

    # q_reset=(1-alpha)q_compact+alpha*q_walk_stand.  Alpha is sampled per
    # episode.  At alpha=1, the four abduction joints are exactly zero.
    reset_alpha_min: float = 0.0
    reset_alpha_max: float = 0.10
    reset_joint_noise_rad: float = 0.01
    reset_velocity_noise_rad_s: float = 0.05
    snapshot_reset_probability: float = 0.0
    reset_ground_clearance_m: float = 0.0005

    # Actor observation exactly follows train_ppo_deploy: a raw 36-value frame
    # stacked newest-first over 20 policy steps.
    observation_history: int = 20
    observation_noise_enabled: bool = True
    observation_noise_angular_velocity_rad_s: float = 0.20
    observation_noise_gravity: float = 0.05
    observation_noise_joint_position_rad: float = 0.01
    observation_limit: float = 100.0

    # Capture is a reward/metric milestone only.  It never changes actions.
    capture_d_threshold: float = 1.20
    capture_omega_min_rad_s: float = 0.50
    capture_sustain_s: float = 0.08
    # Zero retains the legacy full-episode task for teacher/data tools.
    post_capture_turns: int = 0
    reward_insurance_bonus: float = 2.0
    reward_wait_capture: float = 0.0
    torque_hard_limit_nm: float = 0.0  # opt-in; zero preserves old checkpoints
    torque_soft_limit_nm: float = 2.0
    reward_torque_excess: float = 0.0  # per second, sum over joints
    load_diagnostics: bool = False

    # Fixed shaping through full_stand; teacher annealing comes later.
    reward_roll_progress: float = 1.0
    reward_cem_progress: float = 0.25
    reward_cem_orbit: float = 0.02
    reward_compact_progress: float = 0.20
    reward_capture_bonus: float = 2.0
    reward_action_rate: float = 0.002
    reward_torque: float = 0.002
    reward_joint_limit: float = 0.05
    reward_forbidden_collision: float = 1.0
    # Per-second coefficients; step terms are multiplied by control_timestep.
    reward_lateral: float = 4.0
    reward_lateral_before_capture: float = 0.0  # additional per-second coefficient
    reward_sustain: float = 1.0
    sustain_forward_speed_min_m_s: float = 0.02

    # Falling and low torso height are valid rolling behavior.
    terminate_root_z_max_m: float = 0.50
    terminate_lateral_m: float = 0.60
    terminate_axis_tilt_rad: float = 1.20

    @property
    def control_timestep(self) -> float:
        return 0.02


def stand_to_roll_curriculum_config(
    stage: str, base: StandToRollConfig | None = None
) -> StandToRollConfig:
    """Return one reset-curriculum level without changing reward weights."""

    config = base or StandToRollConfig()
    bounds = {
        "rolling_orbit": (0.0, 0.10),
        "mixed_75": (0.0, 0.10),
        "mixed_25": (0.0, 0.10),
        "compact": (0.0, 0.10),
        "slightly_open": (0.0, 0.30),
        "crouch": (0.20, 0.60),
        "semi_stand": (0.50, 1.00),
        "full_stand": (1.00, 1.00),
    }
    if stage not in bounds:
        raise ValueError(f"unknown stand-to-roll curriculum stage: {stage}")
    lo, hi = bounds[stage]
    result = replace(config, reset_alpha_min=lo, reset_alpha_max=hi,
                     snapshot_reset_probability={
                         "rolling_orbit": 1.0, "mixed_75": 0.75,
                         "mixed_25": 0.25,
                     }.get(stage, 0.0))
    validate_stand_to_roll_config(result)
    return result


def validate_stand_to_roll_config(config: StandToRollConfig) -> None:
    if config.post_capture_turns not in (0, 1, 2):
        raise ValueError("post_capture_turns must be 0, 1 or 2")
    if not math.isfinite(config.torque_hard_limit_nm) or config.torque_hard_limit_nm < 0:
        raise ValueError("torque_hard_limit_nm must be finite and nonnegative")
    if not math.isfinite(config.torque_soft_limit_nm) or config.torque_soft_limit_nm <= 0:
        raise ValueError("torque_soft_limit_nm must be finite and positive")
    if 0 < config.torque_hard_limit_nm < config.torque_soft_limit_nm:
        raise ValueError("hard torque limit must be >= soft torque limit")
    if config.episode_length < 1:
        raise ValueError("episode_length must be positive")
    if not 0.0 <= config.snapshot_reset_probability <= 1.0:
        raise ValueError("snapshot_reset_probability must be in [0, 1]")
    if not 0.0 <= config.reset_alpha_min <= config.reset_alpha_max <= 1.0:
        raise ValueError("reset alpha bounds must satisfy 0 <= min <= max <= 1")
    if config.observation_history != 20:
        raise ValueError("deployment observation history must remain 20")
    for value, name in (
        (config.action_scale_abduction_rad, "action_scale_abduction_rad"),
        (config.action_scale_hip_rad, "action_scale_hip_rad"),
        (config.action_scale_knee_rad, "action_scale_knee_rad"),
        (config.capture_d_threshold, "capture_d_threshold"),
        (config.capture_sustain_s, "capture_sustain_s"),
        (config.observation_limit, "observation_limit"),
        (config.terminate_root_z_max_m, "terminate_root_z_max_m"),
        (config.terminate_lateral_m, "terminate_lateral_m"),
        (config.terminate_axis_tilt_rad, "terminate_axis_tilt_rad"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for value, name in (
        (config.reset_joint_noise_rad, "reset_joint_noise_rad"),
        (config.reset_ground_clearance_m, "reset_ground_clearance_m"),
        (config.sustain_forward_speed_min_m_s, "sustain_forward_speed_min_m_s"),
        (config.reset_velocity_noise_rad_s, "reset_velocity_noise_rad_s"),
        (config.capture_omega_min_rad_s, "capture_omega_min_rad_s"),
        (config.observation_noise_angular_velocity_rad_s, "observation noise"),
        (config.observation_noise_gravity, "observation noise"),
        (config.observation_noise_joint_position_rad, "observation noise"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in vars(config).items():
        if name.startswith("reward_") and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")
