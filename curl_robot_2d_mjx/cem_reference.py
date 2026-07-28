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
class CEMReferenceConfig:
    """Frozen phase-locked CEM controller and residual blending settings."""

    coefficients: tuple[float, ...]
    oscillator_rate_rad_s: float
    oscillator_coupling_per_s: float
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
        reference_weight=float(reference_weight),
        minimum_residual_gain=float(minimum_residual_gain),
        source=str(path),
    )
    config.with_weight(config.reference_weight)
    residual_gain(config.reference_weight, config.minimum_residual_gain)
    return config


def advance_oscillator(
    xp,
    body_phase,
    oscillator_phase,
    timestep,
    config: CEMReferenceConfig,
):
    """Match the feedback phase update used by the CPU CEM controller."""

    phase_rate = (
        config.oscillator_rate_rad_s
        + config.oscillator_coupling_per_s
        * xp.sin(body_phase - oscillator_phase)
    )
    return oscillator_phase + timestep * xp.maximum(0.1, phase_rate)


def reference_action(
    xp,
    oscillator_phase,
    config: CEMReferenceConfig,
    *,
    compact_ctrl,
    action_scales,
    joint_low,
    joint_high,
):
    """Return the CEM target as a normalized action around compact."""

    coefficients = xp.asarray(config.coefficients)
    sine = coefficients[0::2]
    cosine = coefficients[1::2]
    target = compact_ctrl + (
        sine * xp.sin(oscillator_phase)
        + cosine * xp.cos(oscillator_phase)
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
