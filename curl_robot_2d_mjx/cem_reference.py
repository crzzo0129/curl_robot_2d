"""Dependency-light CEM reference controller for residual MJX policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CEM_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "collision_constrained_cem"
    / "best_phase_controller.json"
)
COEFFICIENT_NAMES = (
    "front_hip_sin",
    "front_hip_cos",
    "front_knee_sin",
    "front_knee_cos",
    "rear_hip_sin",
    "rear_hip_cos",
    "rear_knee_sin",
    "rear_knee_cos",
)


@dataclass(frozen=True)
class CEMReferenceGeometry:
    """Planar dimensions used by the foot-clearance projection."""

    torso_length_m: float = 0.15
    link_length_m: float = 0.15
    foot_diameter_m: float = 0.0399
    upper_link_length_m: float | None = None
    lower_link_length_m: float | None = None


BASELINE_CEM_REFERENCE_GEOMETRY = CEMReferenceGeometry()


@dataclass(frozen=True)
class CEMReferenceConfig:
    """Frozen phase-locked CEM controller and residual blending settings."""

    coefficients: tuple[float, ...]
    oscillator_rate_rad_s: float
    oscillator_coupling_per_s: float
    knee_bias_rad: float = 0.0
    minimum_foot_surface_gap_m: float = 0.0
    foot_gap_tracking_margin_m: float = 0.0
    reference_weight: float = 1.0
    minimum_residual_gain: float = 0.05
    source: str = ""

    def with_weight(self, weight: float) -> "CEMReferenceConfig":
        if not 0.0 <= weight <= 1.0:
            raise ValueError("reference weight must be in [0, 1]")
        return replace(self, reference_weight=float(weight))

    @property
    def residual_gain(self) -> float:
        return residual_gain(
            self.reference_weight, self.minimum_residual_gain
        )


def residual_gain(reference_weight: float, minimum_gain: float) -> float:
    """Increase policy authority as the CEM reference is withdrawn."""

    if not 0.0 <= reference_weight <= 1.0:
        raise ValueError("reference weight must be in [0, 1]")
    if not 0.0 <= minimum_gain <= 1.0:
        raise ValueError("minimum residual gain must be in [0, 1]")
    return minimum_gain + (1.0 - reference_weight) * (1.0 - minimum_gain)


def load_cem_reference(
    path: Path = DEFAULT_CEM_CONTROLLER,
    *,
    reference_weight: float = 1.0,
    minimum_residual_gain: float = 0.05,
    minimum_foot_surface_gap_m: float | None = None,
    foot_gap_tracking_margin_m: float | None = None,
) -> CEMReferenceConfig:
    """Load the frozen CEM phase-locked oscillator from its JSON artifact."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("controller") != "phase_locked_oscillator":
        raise ValueError(f"unsupported CEM controller in {path}")
    raw = payload["raw_coefficients"]
    coefficients = tuple(float(raw[name]) for name in COEFFICIENT_NAMES)
    if len(coefficients) != 8 or not np.isfinite(coefficients).all():
        raise ValueError(f"invalid CEM coefficients in {path}")
    config = CEMReferenceConfig(
        coefficients=coefficients,
        oscillator_rate_rad_s=float(payload["oscillator_rate_rad_s"]),
        oscillator_coupling_per_s=float(
            payload["oscillator_coupling_per_s"]
        ),
        knee_bias_rad=float(payload.get("nominal_knee_bias_rad", 0.0)),
        minimum_foot_surface_gap_m=float(
            payload.get("minimum_foot_surface_gap_m", 0.0)
            if minimum_foot_surface_gap_m is None
            else minimum_foot_surface_gap_m
        ),
        foot_gap_tracking_margin_m=float(
            payload.get("foot_gap_tracking_margin_m", 0.0)
            if foot_gap_tracking_margin_m is None
            else foot_gap_tracking_margin_m
        ),
        reference_weight=float(reference_weight),
        minimum_residual_gain=float(minimum_residual_gain),
        source=str(path),
    )
    config.with_weight(config.reference_weight)
    if not math.isfinite(config.knee_bias_rad):
        raise ValueError(f"invalid CEM knee bias in {path}")
    if (
        not math.isfinite(config.minimum_foot_surface_gap_m)
        or config.minimum_foot_surface_gap_m < 0.0
    ):
        raise ValueError(f"invalid CEM foot gap in {path}")
    if (
        not math.isfinite(config.foot_gap_tracking_margin_m)
        or config.foot_gap_tracking_margin_m < 0.0
    ):
        raise ValueError(f"invalid CEM foot-gap tracking margin in {path}")
    residual_gain(config.reference_weight, config.minimum_residual_gain)
    return config


def advance_oscillator(
    xp,
    body_phase,
    oscillator_phase,
    timestep,
    config: CEMReferenceConfig,
    *,
    rate_scale=1.0,
):
    """Match the feedback phase update used by the CPU CEM controller."""

    scale = xp.asarray(rate_scale)
    direction = xp.where(scale < 0.0, -1.0, 1.0)
    directed_error = direction * (body_phase - oscillator_phase)
    phase_speed = (
        xp.abs(scale) * config.oscillator_rate_rad_s
        + config.oscillator_coupling_per_s * xp.sin(directed_error)
    )
    phase_rate = direction * xp.maximum(0.1, phase_speed)
    phase_rate = xp.where(scale == 0.0, 0.0, phase_rate)
    return oscillator_phase + timestep * phase_rate


def wrapped_phase_error(xp, body_phase, oscillator_phase):
    """Return body-minus-oscillator phase error in [-pi, pi]."""

    difference = body_phase - oscillator_phase
    return xp.arctan2(xp.sin(difference), xp.cos(difference))


def reference_action(
    xp,
    oscillator_phase,
    config: CEMReferenceConfig,
    *,
    compact_ctrl,
    action_scales,
    joint_low,
    joint_high,
    geometry: CEMReferenceGeometry = BASELINE_CEM_REFERENCE_GEOMETRY,
):
    """Return the CEM target as a normalized action around compact."""

    coefficients = xp.asarray(config.coefficients)
    sine = coefficients[0::2]
    cosine = coefficients[1::2]
    nominal_offset = xp.asarray(
        [0.0, config.knee_bias_rad, 0.0, config.knee_bias_rad]
    )
    target = compact_ctrl + (
        nominal_offset
        + sine * xp.sin(oscillator_phase)
        + cosine * xp.cos(oscillator_phase)
    )
    if config.minimum_foot_surface_gap_m > 0.0:
        torso_length = geometry.torso_length_m
        upper_length = (
            geometry.link_length_m
            if geometry.upper_link_length_m is None
            else geometry.upper_link_length_m
        )
        lower_length = (
            geometry.link_length_m
            if geometry.lower_link_length_m is None
            else geometry.lower_link_length_m
        )
        target_distance = (
            geometry.foot_diameter_m
            + config.minimum_foot_surface_gap_m
            + config.foot_gap_tracking_margin_m
        )
        for _ in range(6):
            front_hip, front_knee, rear_hip, rear_knee = target
            delta_x = torso_length + (
                upper_length * xp.sin(front_hip)
                + lower_length * xp.sin(front_hip - front_knee)
                + upper_length * xp.sin(rear_hip)
                + lower_length * xp.sin(rear_hip - rear_knee)
            )
            delta_z = (
                -upper_length * xp.cos(front_hip)
                - lower_length * xp.cos(front_knee - front_hip)
                + upper_length * xp.cos(rear_hip)
                + lower_length * xp.cos(rear_knee - rear_hip)
            )
            distance = xp.sqrt(delta_x * delta_x + delta_z * delta_z)
            front_dx = -lower_length * xp.cos(front_hip - front_knee)
            front_dz = -lower_length * xp.sin(front_knee - front_hip)
            rear_dx = -lower_length * xp.cos(rear_hip - rear_knee)
            rear_dz = -lower_length * xp.sin(rear_knee - rear_hip)
            front_gradient = (
                delta_x * front_dx + delta_z * front_dz
            ) / xp.maximum(distance, 1.0e-6)
            rear_gradient = (
                delta_x * rear_dx + delta_z * rear_dz
            ) / xp.maximum(distance, 1.0e-6)
            gradient_norm_squared = (
                front_gradient * front_gradient
                + rear_gradient * rear_gradient
            )
            scale = xp.maximum(target_distance - distance, 0.0) / xp.maximum(
                gradient_norm_squared, 1.0e-8
            )
            zero = xp.zeros_like(scale)
            target = target + xp.stack(
                (
                    zero,
                    xp.clip(scale * front_gradient, -0.20, 0.20),
                    zero,
                    xp.clip(scale * rear_gradient, -0.20, 0.20),
                )
            )
    target = xp.clip(target, joint_low, joint_high)
    return xp.clip((target - compact_ctrl) / action_scales, -1.0, 1.0)


def effective_residual_action(
    xp,
    policy_action,
    cem_action,
    config: CEMReferenceConfig,
):
    """Blend CEM authority with a policy residual in normalized coordinates."""

    return xp.clip(
        config.reference_weight * cem_action
        + config.residual_gain * policy_action,
        -1.0,
        1.0,
    )


def expected_budget_steps(requested_steps: int, rollout_quantum: int) -> int:
    """Round a curriculum budget once, independently of stage transitions."""

    if requested_steps <= 0 or rollout_quantum <= 0:
        raise ValueError("requested steps and rollout quantum must be positive")
    return math.ceil(requested_steps / rollout_quantum) * rollout_quantum
