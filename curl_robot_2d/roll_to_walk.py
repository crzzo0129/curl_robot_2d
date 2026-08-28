"""A deterministic roll-to-walk transition baseline for the 2-D curl robot.

The controller is intentionally small and inspectable.  It keeps the existing
phase-locked rolling controller, then performs a one-way transition through
braking and deployment before handing control to the existing planar walking
controller.  This is a runnable systems baseline, not a learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts.explore_walking_controller import (
    WalkingControllerConfig,
    walking_targets,
)
from scripts.optimize_phase_controller import controller_targets


JOINT_NAMES = ("front_hip", "front_knee", "rear_hip", "rear_knee")
DEFAULT_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "phase_controller"
    / "best_phase_controller.json"
)


class TransitionMode(str, Enum):
    ROLL = "roll"
    BRAKE = "brake"
    DEPLOY = "deploy"
    WALK = "walk"
    COMPLETE = "complete"


@dataclass(frozen=True)
class RollToWalkConfig:
    """Timing and safety parameters for one complete transition."""

    roll_duration_s: float = 1.4
    brake_duration_s: float = 1.2
    deploy_duration_s: float = 1.6
    walk_duration_s: float = 3.0
    brake_start_progress: float = 0.7557696520
    brake_end_rate_scale: float = 0.4356312723
    brake_end_amplitude_scale: float = 0.3852994846
    brake_midpoint_offsets_rad: tuple[float, ...] = (
        0.1463065639, 0.0437949843, 0.0815537638, 0.1429862062
    )
    brake_terminal_offsets_rad: tuple[float, ...] = (
        0.0831906869, 0.0011188551, 0.1635395515, 0.1674802041
    )
    walking: WalkingControllerConfig = WalkingControllerConfig()

    def __post_init__(self) -> None:
        for name in (
            "roll_duration_s",
            "brake_duration_s",
            "deploy_duration_s",
            "walk_duration_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.brake_start_progress < 1.0:
            raise ValueError("brake_start_progress must be in [0, 1)")
        if self.brake_end_rate_scale < 0.0:
            raise ValueError("brake_end_rate_scale must be nonnegative")
        if self.brake_end_amplitude_scale < 0.0:
            raise ValueError("brake_end_amplitude_scale must be nonnegative")
        if len(self.brake_midpoint_offsets_rad) != 4:
            raise ValueError("brake midpoint offsets must contain four values")
        if len(self.brake_terminal_offsets_rad) != 4:
            raise ValueError("brake terminal offsets must contain four values")

    @property
    def total_duration_s(self) -> float:
        return (
            self.roll_duration_s
            + self.brake_duration_s
            + self.deploy_duration_s
            + self.walk_duration_s
        )


@dataclass(frozen=True)
class RollController:
    coefficients: np.ndarray
    oscillator_rate_rad_s: float
    oscillator_coupling_per_s: float
    minimum_foot_surface_gap_m: float = 0.0
    foot_gap_tracking_margin_m: float = 0.004
    nominal_knee_bias_rad: float = 0.0


@dataclass(frozen=True)
class TransitionResult:
    mode_history: tuple[str, ...]
    columns: tuple[str, ...]
    rows: np.ndarray
    summary: dict[str, float | int | str | bool]


def load_roll_controller(path: Path | None = None) -> RollController:
    """Load the validated rolling reference, with a safe built-in fallback."""

    path = DEFAULT_CONTROLLER_PATH if path is None else Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload["raw_coefficients"]
        names = tuple(
            name
            for joint in JOINT_NAMES
            for name in (f"{joint}_sin", f"{joint}_cos")
        )
        coefficients = np.asarray([float(raw[name]) for name in names])
        return RollController(
            coefficients=coefficients,
            oscillator_rate_rad_s=float(payload["oscillator_rate_rad_s"]),
            oscillator_coupling_per_s=float(
                payload["oscillator_coupling_per_s"]
            ),
            minimum_foot_surface_gap_m=float(
                payload.get("minimum_foot_surface_gap_m", 0.0)
            ),
            foot_gap_tracking_margin_m=float(
                payload.get("foot_gap_tracking_margin_m", 0.004)
            ),
            nominal_knee_bias_rad=float(
                payload.get("nominal_knee_bias_rad", 0.0)
            ),
        )

    # This is the checked-in controller's numerical baseline.  Keeping a
    # fallback makes the transition demo runnable from a clean checkout.
    return RollController(
        coefficients=np.asarray(
            (
                -0.1346346636,
                0.0461473104,
                -0.6405481460,
                -0.0963308835,
                -0.1657090796,
                0.3782152312,
                -0.6854323089,
                -0.2924007688,
            ),
            dtype=float,
        ),
        oscillator_rate_rad_s=3.6565631998,
        oscillator_coupling_per_s=4.7961792215,
    )


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"missing MuJoCo object: {name}")
    return int(value)


def _quintic01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def _wrap_to_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _forward_phase_distance(current: float, target: float, direction: float) -> float:
    return ((target - current) * direction) % (2.0 * math.pi)


def _setup(model: mujoco.MjModel) -> dict[str, object]:
    joint_qpos = np.asarray(
        [
            model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in JOINT_NAMES
        ],
        dtype=int,
    )
    joint_dof = np.asarray(
        [
            model.jnt_dofadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in JOINT_NAMES
        ],
        dtype=int,
    )
    return {
        "root_x_qpos": _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x"),
        "root_z_qpos": _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_z"),
        "root_pitch_qpos": _id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch"
        ),
        "root_x_dof": _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x"),
        "root_pitch_dof": _id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch"
        ),
        "joint_qpos": joint_qpos,
        "joint_dof": joint_dof,
        "front_foot_site": _id(
            model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"
        ),
        "rear_foot_site": _id(
            model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"
        ),
        "floor_geom": _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        "foot_geoms": frozenset(
            (
                _id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_foot_proxy"),
                _id(model, mujoco.mjtObj.mjOBJ_GEOM, "rear_foot_proxy"),
            )
        ),
    }


def _contact_summary(
    data: mujoco.MjData,
    *,
    floor_geom: int,
    foot_geoms: frozenset[int],
) -> tuple[int, int]:
    foot_contacts: set[int] = set()
    nonfoot_contacts = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (int(contact.geom1), int(contact.geom2))
        if floor_geom not in pair:
            continue
        robot_geom = pair[1] if pair[0] == floor_geom else pair[0]
        if robot_geom in foot_geoms:
            foot_contacts.add(robot_geom)
        else:
            nonfoot_contacts += 1
    return len(foot_contacts), nonfoot_contacts


def _roll_targets(
    phase: float,
    time_s: float,
    control_phase: float,
    controller: RollController,
    amplitude: float,
) -> np.ndarray:
    raw = controller_targets(
        phase,
        time_s,
        controller.coefficients,
        oscillator_rate=controller.oscillator_rate_rad_s,
        control_phase=control_phase,
        knee_bias_rad=controller.nominal_knee_bias_rad,
        minimum_foot_surface_gap_m=controller.minimum_foot_surface_gap_m,
        foot_gap_tracking_margin_m=controller.foot_gap_tracking_margin_m,
    )
    compact = np.asarray(
        (
            FIXED_PARAMETERS.compact_hip_angle,
            FIXED_PARAMETERS.compact_knee_angle,
            FIXED_PARAMETERS.compact_hip_angle,
            FIXED_PARAMETERS.compact_knee_angle,
        ),
        dtype=float,
    )
    return compact + float(np.clip(amplitude, 0.0, 1.0)) * (raw - compact)


def simulate_roll_to_walk(
    model: mujoco.MjModel,
    config: RollToWalkConfig | None = None,
    controller: RollController | None = None,
    *,
    detailed: bool = True,
) -> TransitionResult:
    """Run ROLL -> BRAKE -> DEPLOY -> WALK in one MuJoCo simulation."""

    config = RollToWalkConfig() if config is None else config
    controller = load_roll_controller() if controller is None else controller
    setup = _setup(model)
    root_x_qpos = int(model.jnt_qposadr[setup["root_x_qpos"]])
    root_z_qpos = int(model.jnt_qposadr[setup["root_z_qpos"]])
    root_pitch_qpos = int(model.jnt_qposadr[setup["root_pitch_qpos"]])
    root_x_dof = int(model.jnt_dofadr[setup["root_x_dof"]])
    root_pitch_dof = int(model.jnt_dofadr[setup["root_pitch_dof"]])
    joint_qpos = np.asarray(setup["joint_qpos"])

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, _id(model, mujoco.mjtObj.mjOBJ_KEY, "compact"))
    data.qvel[:] = 0.0
    # The validated rolling reference was identified with free root DOFs.
    # Leaving the XML's small root damping enabled changes its phase and makes
    # the transition start from a falling, rather than rolling, state.
    for joint_name in ("root_x", "root_z", "root_pitch"):
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        model.dof_damping[int(model.jnt_dofadr[joint_id])] = 0.0
    mujoco.mj_forward(model, data)

    walking_start_targets, _ = walking_targets(
        0.0, 0.0, 0.0, 0.0, config.walking,
        root_height_m=FIXED_PARAMETERS.walk_root_height,
    )
    compact_targets = data.qpos[joint_qpos].copy()
    mode = TransitionMode.ROLL
    mode_history = [mode.value]
    oscillator_phase = 0.0
    brake_start_phase = 0.0
    brake_direction = 1.0
    brake_target_distance = 1.0
    deploy_start_targets = compact_targets.copy()
    transition_times = {
        TransitionMode.BRAKE: config.roll_duration_s,
        TransitionMode.DEPLOY: config.roll_duration_s + config.brake_duration_s,
        TransitionMode.WALK: config.roll_duration_s + config.brake_duration_s + config.deploy_duration_s,
        TransitionMode.COMPLETE: config.total_duration_s,
    }
    rows: list[list[float]] = []
    previous_mode = mode
    timestep = float(model.opt.timestep)
    steps = int(math.ceil(config.total_duration_s / timestep))
    previous_wrapped_phase = float(data.qpos[root_pitch_qpos])
    body_phase_unwrapped = previous_wrapped_phase

    for _ in range(steps):
        time_s = float(data.time)
        wrapped_phase = float(data.qpos[root_pitch_qpos])
        body_phase_unwrapped += _wrap_to_pi(
            wrapped_phase - previous_wrapped_phase
        )
        previous_wrapped_phase = wrapped_phase
        if mode == TransitionMode.ROLL and time_s >= transition_times[TransitionMode.BRAKE]:
            brake_start_phase = body_phase_unwrapped
            phase = wrapped_phase
            brake_direction = 1.0 if float(data.qvel[root_pitch_dof]) >= 0.0 else -1.0
            required_distance = (
                abs(float(data.qvel[root_pitch_dof])) ** 2 / (2.0 * 8.0)
                + math.radians(20.0)
            )
            nearest = _forward_phase_distance(
                brake_start_phase, 0.0, brake_direction
            )
            brake_target_distance = max(nearest, required_distance)
            mode = TransitionMode.BRAKE
        elif mode == TransitionMode.BRAKE and time_s >= transition_times[TransitionMode.DEPLOY]:
            deploy_start_targets = data.qpos[joint_qpos].copy()
            mode = TransitionMode.DEPLOY
        elif mode == TransitionMode.DEPLOY and time_s >= transition_times[TransitionMode.WALK]:
            deploy_start_targets = data.qpos[joint_qpos].copy()
            mode = TransitionMode.WALK
        elif mode == TransitionMode.WALK and time_s >= transition_times[TransitionMode.COMPLETE]:
            mode = TransitionMode.COMPLETE
        if mode != previous_mode:
            mode_history.append(mode.value)
            previous_mode = mode

        phase = float(data.qpos[root_pitch_qpos])
        pitch_rate = float(data.qvel[root_pitch_dof])
        root_velocity = float(data.qvel[root_x_dof])
        phase_locked_rate = controller.oscillator_rate_rad_s + (
            controller.oscillator_coupling_per_s
            * math.sin(phase - oscillator_phase)
        )
        if mode == TransitionMode.ROLL:
            rate = max(0.1, phase_locked_rate)
            oscillator_rate_scale = 1.0
            amplitude = 1.0
            walking_phase = 0.0
        elif mode == TransitionMode.BRAKE:
            elapsed = time_s - transition_times[TransitionMode.BRAKE]
            phase_progress = brake_direction * (
                body_phase_unwrapped - brake_start_phase
            ) / max(brake_target_distance, 1.0e-6)
            progress = _quintic01(
                (phase_progress - config.brake_start_progress)
                / max(1.0 - config.brake_start_progress, 1.0e-6)
            )
            rate = (
                (1.0 + progress * (config.brake_end_rate_scale - 1.0))
                * max(0.1, phase_locked_rate)
            )
            oscillator_rate_scale = 1.0 + progress * (
                config.brake_end_rate_scale - 1.0
            )
            amplitude = 1.0 + progress * (
                config.brake_end_amplitude_scale - 1.0
            )
            walking_phase = 0.0
        elif mode == TransitionMode.DEPLOY:
            elapsed = time_s - transition_times[TransitionMode.DEPLOY]
            fraction = _quintic01(elapsed / config.deploy_duration_s)
            rate = 0.0
            oscillator_rate_scale = 0.0
            amplitude = 0.0
            walking_phase = 0.0
        else:
            rate = 0.0
            oscillator_rate_scale = 0.0
            amplitude = 0.0
            walking_phase = time_s - transition_times[TransitionMode.WALK]

        oscillator_phase += timestep * rate
        if mode in (TransitionMode.ROLL, TransitionMode.BRAKE):
            targets = _roll_targets(
                phase, time_s, oscillator_phase, controller, amplitude
            )
            if mode == TransitionMode.BRAKE:
                phase_progress = brake_direction * (
                    body_phase_unwrapped - brake_start_phase
                ) / max(brake_target_distance, 1.0e-6)
                progress = _quintic01(
                    (phase_progress - config.brake_start_progress)
                    / max(1.0 - config.brake_start_progress, 1.0e-6)
                )
                offset_progress = _quintic01(progress * 2.0)
                if progress > 0.5:
                    offset_progress = 1.0 + _quintic01(
                        (progress - 0.5) * 2.0
                    )
                midpoint = np.asarray(config.brake_midpoint_offsets_rad)
                terminal = np.asarray(config.brake_terminal_offsets_rad)
                if progress <= 0.5:
                    targets = targets + offset_progress * midpoint
                else:
                    targets = targets + midpoint + (
                        offset_progress - 1.0
                    ) * (terminal - midpoint)
        elif mode == TransitionMode.DEPLOY:
            fraction = _quintic01(
                (time_s - transition_times[TransitionMode.DEPLOY])
                / config.deploy_duration_s
            )
            targets = deploy_start_targets + fraction * (
                walking_start_targets - deploy_start_targets
            )
        elif mode == TransitionMode.WALK:
            targets, _ = walking_targets(
                walking_phase,
                phase,
                pitch_rate,
                root_velocity,
                config.walking,
                root_height_m=float(data.qpos[root_z_qpos]),
            )
        else:
            targets = walking_start_targets

        data.ctrl[:] = targets
        mujoco.mj_step(model, data)

        foot_contacts, nonfoot_contacts = _contact_summary(
            data,
            floor_geom=int(setup["floor_geom"]),
            foot_geoms=setup["foot_geoms"],
        )
        rows.append(
            [
                float(data.time),
                float(list(TransitionMode).index(mode)),
                float(data.qpos[root_x_qpos]),
                float(data.qpos[root_z_qpos]),
                float(data.qpos[root_pitch_qpos]),
                float(data.qvel[root_x_dof]),
                float(data.qvel[root_pitch_dof]),
                float(foot_contacts),
                float(nonfoot_contacts),
                float(np.max(np.abs(data.actuator_force))),
                float(np.max(np.abs(data.qpos[joint_qpos] - targets))),
                float(oscillator_rate_scale),
            ]
        )

    full_array = np.asarray(rows, dtype=float)
    if mode_history[-1] != TransitionMode.COMPLETE.value:
        mode_history.append(TransitionMode.COMPLETE.value)
    mode_values = full_array[:, 1] if len(full_array) else np.empty(0)
    walk_rows = full_array[
        mode_values >= float(list(TransitionMode).index(TransitionMode.WALK))
    ]
    initial_x = float(rows[0][2])
    final_x = float(rows[-1][2])
    finite = bool(
        np.isfinite(full_array).all()
        and np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
    )
    walk_nonfoot_steps = (
        int(np.count_nonzero(walk_rows[:, 8] > 0.0))
        if len(walk_rows)
        else 0
    )
    walk_safe = bool(
        len(walk_rows)
        and np.min(walk_rows[:, 7]) >= 1.0
        and walk_nonfoot_steps == 0
        and np.max(np.abs(walk_rows[:, 4])) < 0.80
        and np.min(walk_rows[:, 3]) > 0.115
    )
    summary: dict[str, float | int | str | bool] = {
        "status": "ok" if finite else "numerical_failure",
        "completed_all_modes": mode_history[-1] == TransitionMode.COMPLETE.value,
        "walking_baseline_safe": walk_safe,
        "initial_mode": TransitionMode.ROLL.value,
        "final_mode": TransitionMode.COMPLETE.value,
        "mode_count": len(mode_history),
        "elapsed_s": float(data.time),
        "roll_duration_s": config.roll_duration_s,
        "brake_duration_s": config.brake_duration_s,
        "deploy_duration_s": config.deploy_duration_s,
        "walk_duration_s": config.walk_duration_s,
        "root_x_displacement_m": final_x - initial_x,
        "walk_displacement_m": (
            float(walk_rows[-1, 2] - walk_rows[0, 2]) if len(walk_rows) else 0.0
        ),
        "walk_mean_velocity_m_s": (
            float((walk_rows[-1, 2] - walk_rows[0, 2]) / config.walk_duration_s)
            if len(walk_rows) else 0.0
        ),
        "walk_nonfoot_contact_steps": walk_nonfoot_steps,
        "walk_maximum_abs_pitch_rad": (
            float(np.max(np.abs(walk_rows[:, 4]))) if len(walk_rows) else 0.0
        ),
        "walk_minimum_root_z_m": (
            float(np.min(walk_rows[:, 3])) if len(walk_rows) else 0.0
        ),
        "maximum_abs_pitch_rad": float(np.max(np.abs(full_array[:, 4]))),
        "minimum_root_z_m": float(np.min(full_array[:, 3])),
        "maximum_root_z_m": float(np.max(full_array[:, 3])),
        "minimum_walk_foot_contacts": int(np.min(walk_rows[:, 7])) if len(walk_rows) else 0,
        "nonfoot_contact_steps": int(np.count_nonzero(full_array[:, 8] > 0.0)),
        "maximum_actuator_force_N": float(np.max(full_array[:, 9])),
    }
    return TransitionResult(
        mode_history=tuple(mode_history),
        columns=(
            "time_s", "mode_index", "root_x_m", "root_z_m", "root_pitch_rad",
            "root_x_velocity_m_s", "root_pitch_rate_rad_s", "foot_contacts",
            "nonfoot_ground_contacts", "maximum_actuator_force_N",
            "maximum_joint_tracking_error_rad", "roll_rate_scale",
        ),
        rows=full_array if detailed else np.empty((0, 12)),
        summary=summary,
    )
