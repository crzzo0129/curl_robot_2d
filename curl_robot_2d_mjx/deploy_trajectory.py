"""State-conditioned deterministic trajectories for parking deployment.

The trajectory starts from the measured joint position and velocity at the
BRAKE_ALIGN -> PARK_DEPLOY transition.  This avoids the position and velocity
jump caused by replaying an absolute, pre-recorded parking motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DeploySample:
    position: object
    velocity: object
    acceleration: object
    progress: float
    finished: bool


def _validate_duration(duration_s: float) -> None:
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("trajectory duration must be finite and positive")


def quintic_boundary_sample(
    xp,
    start_position,
    start_velocity,
    end_position,
    end_velocity,
    *,
    elapsed_s: float,
    duration_s: float,
) -> DeploySample:
    """Sample a quintic satisfying position, velocity and zero-acceleration ends."""

    _validate_duration(duration_s)
    q0 = xp.asarray(start_position)
    v0 = xp.asarray(start_velocity)
    q1 = xp.asarray(end_position)
    v1 = xp.asarray(end_velocity)
    if q0.shape != v0.shape or q0.shape != q1.shape or q0.shape != v1.shape:
        raise ValueError("all trajectory boundary arrays must have the same shape")

    t = min(max(float(elapsed_s), 0.0), duration_s)
    duration = float(duration_s)
    # Coefficients for q(t)=a0+a1*t+...+a5*t^5 with a2=0 and qdd(T)=0.
    delta = q1 - q0
    a0 = q0
    a1 = v0
    a2 = xp.zeros_like(q0)
    a3 = (10.0 * delta - (6.0 * v0 + 4.0 * v1) * duration) / duration**3
    a4 = (-15.0 * delta + (8.0 * v0 + 7.0 * v1) * duration) / duration**4
    a5 = (6.0 * delta - (3.0 * v0 + 3.0 * v1) * duration) / duration**5
    position = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    velocity = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    acceleration = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return DeploySample(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        progress=t / duration,
        finished=elapsed_s >= duration_s,
    )


def deploy_trajectory_sample(
    xp,
    capture_position,
    capture_velocity,
    park_position,
    *,
    elapsed_s: float,
    duration_s: float,
    midpoint=None,
    midpoint_fraction: float = 0.5,
) -> DeploySample:
    """Generate a direct quintic or a deterministic two-segment deployment."""

    _validate_duration(duration_s)
    if midpoint is None:
        return quintic_boundary_sample(
            xp,
            capture_position,
            capture_velocity,
            park_position,
            xp.zeros_like(xp.asarray(park_position)),
            elapsed_s=elapsed_s,
            duration_s=duration_s,
        )
    if not 0.0 < midpoint_fraction < 1.0:
        raise ValueError("midpoint_fraction must lie strictly between zero and one")
    split_s = duration_s * midpoint_fraction
    if elapsed_s <= split_s:
        sample = quintic_boundary_sample(
            xp,
            capture_position,
            capture_velocity,
            midpoint,
            xp.zeros_like(xp.asarray(midpoint)),
            elapsed_s=elapsed_s,
            duration_s=split_s,
        )
    else:
        sample = quintic_boundary_sample(
            xp,
            midpoint,
            xp.zeros_like(xp.asarray(midpoint)),
            park_position,
            xp.zeros_like(xp.asarray(park_position)),
            elapsed_s=elapsed_s - split_s,
            duration_s=duration_s - split_s,
        )
    return DeploySample(
        position=sample.position,
        velocity=sample.velocity,
        acceleration=sample.acceleration,
        progress=min(max(elapsed_s / duration_s, 0.0), 1.0),
        finished=elapsed_s >= duration_s,
    )
