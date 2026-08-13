"""Analytic compact geometry using the original Pupper leg dimensions."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PupperShellDesign:
    hip_center_distance: float = 0.15040
    upper_leg_length: float = 0.0844547808
    lower_leg_length: float = 0.0880136895
    motor_envelope_radius: float = 0.032
    foot_radius: float = 0.0195
    compact_foot_center_distance: float = 0.043
    shell_outer_radius: float = 0.1275
    shell_capsule_radius: float = 0.003
    shell_segments: int = 48

    def __post_init__(self) -> None:
        if self.compact_foot_center_distance <= 2.0 * self.foot_radius:
            raise ValueError("compact feet must have positive surface clearance")
        if not 0.122 <= self.shell_outer_radius <= 0.155:
            raise ValueError("shell_outer_radius must be within 122--155 mm")
        if self.shell_capsule_radius <= 0.0:
            raise ValueError("shell_capsule_radius must be positive")

    @property
    def hip_half_distance(self) -> float:
        return self.hip_center_distance / 2.0

    @property
    def foot_half_distance(self) -> float:
        return self.compact_foot_center_distance / 2.0

    @property
    def foot_surface_gap(self) -> float:
        return self.compact_foot_center_distance - 2.0 * self.foot_radius

    @property
    def shell_centerline_radius(self) -> float:
        """Capsule centerline whose outer surface is the requested radius."""

        return self.shell_outer_radius - self.shell_capsule_radius


@dataclass(frozen=True)
class CompactSolution:
    shell_center_below_hip: float
    knee_x: float
    knee_below_hip: float
    foot_x: float
    foot_below_hip: float
    hip_angle: float
    knee_angle: float
    hip_motor_radial_clearance: float


def solve_compact_geometry(design: PupperShellDesign) -> CompactSolution:
    """Solve one symmetric side of the folded Pupper geometry.

    The design circle is treated as the nominal zero-thickness outer arc from
    the CAD sketch.  Feet and knee motor envelopes are internally tangent to
    it.  Coordinates use +x forward/right and positive ``below`` downward.
    """

    p = design
    foot_x = p.foot_half_distance
    foot_radius_from_center = p.shell_outer_radius - p.foot_radius
    if foot_radius_from_center <= foot_x:
        raise ValueError("shell is too small to contain the separated feet")
    foot_relative_below = math.sqrt(
        foot_radius_from_center**2 - foot_x**2
    )

    knee_radius_from_center = (
        p.shell_outer_radius - p.motor_envelope_radius
    )
    center_to_foot = foot_radius_from_center
    along = (
        knee_radius_from_center**2
        - p.lower_leg_length**2
        + center_to_foot**2
    ) / (2.0 * center_to_foot)
    perpendicular_squared = knee_radius_from_center**2 - along**2
    if perpendicular_squared < 0.0:
        raise ValueError("knee/foot tangency circles do not intersect")
    perpendicular = math.sqrt(max(perpendicular_squared, 0.0))
    ux = foot_x / center_to_foot
    uy = foot_relative_below / center_to_foot

    candidates = []
    for sign in (-1.0, 1.0):
        knee_x = along * ux - sign * perpendicular * uy
        knee_relative_below = along * uy + sign * perpendicular * ux
        vertical_squared = p.upper_leg_length**2 - (
            knee_x - p.hip_half_distance
        ) ** 2
        if vertical_squared < 0.0:
            continue
        knee_below_hip = math.sqrt(max(vertical_squared, 0.0))
        center_below_hip = knee_below_hip - knee_relative_below
        if knee_x <= 0.0 or knee_below_hip <= 0.0:
            continue
        foot_below_hip = center_below_hip + foot_relative_below
        lower_dx = foot_x - knee_x
        lower_down = foot_below_hip - knee_below_hip
        if lower_down <= 0.0:
            continue
        hip_angle = math.atan2(
            knee_x - p.hip_half_distance,
            knee_below_hip,
        )
        lower_absolute_angle = math.atan2(lower_dx, lower_down)
        knee_angle = hip_angle - lower_absolute_angle
        hip_center_radius = math.hypot(
            p.hip_half_distance, center_below_hip
        )
        hip_clearance = (
            p.shell_outer_radius
            - hip_center_radius
            - p.motor_envelope_radius
        )
        candidates.append(
            CompactSolution(
                shell_center_below_hip=center_below_hip,
                knee_x=knee_x,
                knee_below_hip=knee_below_hip,
                foot_x=foot_x,
                foot_below_hip=foot_below_hip,
                hip_angle=hip_angle,
                knee_angle=knee_angle,
                hip_motor_radial_clearance=hip_clearance,
            )
        )
    if not candidates:
        raise ValueError("no folded configuration satisfies all constraints")
    solution = max(candidates, key=lambda value: value.shell_center_below_hip)
    if solution.shell_center_below_hip < -1.0e-12:
        raise ValueError("shell center crosses above the hip-center line")
    if solution.hip_motor_radial_clearance < -1.0e-12:
        raise ValueError("shell intersects the hip motor protection envelope")
    return solution

