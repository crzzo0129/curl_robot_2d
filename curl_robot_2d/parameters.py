"""Fixed parameters for the first planar Pupper abstraction.

The values in this module are deliberately split into source-derived robot
parameters and provisional simulation parameters.  The first arc-shell
geometry is parameterized here, but it does not yet add mass or inertia.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


SOURCE_MODEL = (
    "../pupper_v3_description/description/mujoco_xml/"
    "pupper_v3_complete.mjx.position.xml"
)


@dataclass(frozen=True)
class BodyParameters:
    mass: float
    com_along_link: float
    planar_inertia: float


@dataclass(frozen=True)
class JointParameters:
    mechanical_range: tuple[float, float]
    safe_range: tuple[float, float]
    shell_compatible_range: tuple[float, float]
    damping: float
    armature: float
    force_limit: float
    kp: float
    kd: float


@dataclass(frozen=True)
class FixedParameters:
    # Active idealized geometry.  The five center-line edges use one common
    # length: torso hip-to-hip, two thighs, and two shanks.
    edge_length: float = 0.15
    geometry_mode: str = "regular_pentagon"
    torso_length_override: float | None = None
    upper_length_override: float | None = None
    lower_length_override: float | None = None
    shell_segments_full_circle: int = 48
    side_rail_half_width_override: float | None = None
    # Minimum free surface distance between a foot and shell capsules carried
    # by the same shank.  Only used by the Pupper full-circle shell mode.
    pupper_shank_shell_foot_clearance: float = 0.0
    # Optional compact-pose allocation of the full circle to the torso.  The
    # remaining covered angle is split symmetrically between the two thighs;
    # shank coverage and the foot opening are left unchanged.
    pupper_torso_shell_coverage_angle: float | None = None

    # Source Pupper geometry retained for provenance.  These values are not
    # the active link lengths in the equal-edge baseline.
    source_torso_length: float = 0.25943
    source_upper_length: float = 0.0844547808
    source_lower_projected_length: float = 0.0880136895

    torso_mass: float = 1.506
    torso_com_x: float = 0.025
    torso_com_z: float = 0.015
    torso_planar_inertia: float = 0.0085
    source_torso_height: float = 0.09014
    source_torso_width: float = 0.12758
    foot_radius: float = 0.01995

    # Each planar link represents the left/right pair.  The thigh combines
    # source bodies *_1 and *_2 because joint 2 is locked in the 2-D model.
    thigh: BodyParameters = BodyParameters(
        mass=0.732,
        com_along_link=0.0573059111,
        planar_inertia=0.002673,
    )
    shank: BodyParameters = BodyParameters(
        mass=0.1,
        com_along_link=0.0897967059,
        planar_inertia=0.000155075,
    )

    # Joint 1 is the sagittal hip joint.  The limits use the source model's
    # mechanical range and the hardware file's 0.1-rad safety margin.
    hip: JointParameters = JointParameters(
        mechanical_range=(-1.22, 2.51),
        safe_range=(-1.12, 2.41),
        shell_compatible_range=(-1.12, 2.41),
        damping=0.02,
        armature=0.0032,
        force_limit=6.0,
        kp=10.0,
        kd=0.2,
    )

    # The source right knee uses [-2.79, 0.71].  The effective planar knee
    # coordinate reverses that sign so positive values mean "curl inward".
    knee: JointParameters = JointParameters(
        mechanical_range=(-0.71, 2.79),
        safe_range=(-0.61, 2.69),
        shell_compatible_range=(-0.61, 2.69),
        damping=0.02,
        armature=0.0032,
        force_limit=6.0,
        kp=10.0,
        kd=0.2,
    )

    # Provisional simplified collision geometry.  These are not yet design
    # variables and do not claim to reproduce the CAD surface.
    upper_proxy_radius: float = 0.012
    lower_proxy_radius: float = 0.010

    # Optional detailed 2-D structure proxies used by the 180 mm geometry
    # study.  They do not alter the explicitly specified body inertias.
    torso_box_width: float = 0.120
    torso_box_height: float = 0.120
    torso_box_outward_offset: float = 0.0
    structure_half_thickness_y: float = 0.012
    motor_radius: float = 0.027
    motor_half_thickness_y: float = 0.012
    motor_link_clearance: float = 0.001
    # When set, size the shell so its inner surface clears the motor envelope
    # by this radial distance in the compact regular-pentagon pose.
    shell_motor_clearance: float | None = None
    # Controlled shell-only experiment overrides.  They intentionally leave
    # the 150 mm links and finite-size feet unchanged.
    shell_contact_radius_override: float | None = None
    shell_arc_coverage_angle_override: float | None = None
    # Arc length removed only from the foot-side end of each shank shell.
    shank_shell_foot_retreat: float = 0.0

    # First rolling-shell baseline.  Each nominal 72-degree arc is trimmed at
    # both ends, then approximated by short capsules.  In the compact pose,
    # the outside of the shell and the outside of the foot spheres share one
    # nominal contact circle.
    shell_segments_per_edge: int = 6
    shell_capsule_radius: float = 0.006
    # The endpoints are shortened enough to keep at least 2 mm clearance
    # throughout the source-derived safe hip and knee ranges.  Expressed at
    # the collinear pose, that requires a 28 mm endpoint-to-endpoint gap.
    shell_design_gap: float = 0.028

    nominal_ground_friction: float = 0.8
    # A 1 ms step keeps the deliberately stiff self-contact response above
    # MuJoCo's default refsafe threshold (timeconst >= 2 * timestep), reducing
    # penetration without disabling the solver safety guard.
    timestep: float = 0.001

    # First visual keyframes.  "compact" is only an inspection pose; it is not
    # yet an optimized circular shell pose or a rolling trajectory.
    compact_hip_angle_regular: float = 0.3141592653589793
    # The two finite-size foot proxies share one sagittal plane.  Their
    # surfaces may touch in compact, but their centers may not coincide.
    compact_foot_surface_gap: float = 0.0

    # Representative planar walking pose: the front leg is in stance while
    # the rear leg is flexed into swing.  This is a geometry and collision
    # test pose, not a prescribed walking controller trajectory.
    walk_front_hip_angle: float = 0.35
    walk_front_knee_angle: float = 0.52
    walk_rear_hip_angle: float = 0.35
    walk_rear_knee_angle: float = 0.78

    # Static four-foot support pose for direct 3-D locomotion training.  This
    # is a reset and PD action center, not a time-varying gait reference.
    stand_3d_front_hip_angle: float = 0.33532733
    stand_3d_front_knee_angle: float = 0.53234438
    stand_3d_rear_hip_angle: float = 0.28239372
    stand_3d_rear_knee_angle: float = 0.54939539

    @property
    def torso_length(self) -> float:
        return (
            self.edge_length
            if self.torso_length_override is None
            else self.torso_length_override
        )

    @property
    def hip_half_span(self) -> float:
        return self.torso_length / 2.0

    @property
    def upper_length(self) -> float:
        return (
            self.edge_length
            if self.upper_length_override is None
            else self.upper_length_override
        )

    @property
    def lower_length(self) -> float:
        return (
            self.edge_length
            if self.lower_length_override is None
            else self.lower_length_override
        )

    @property
    def uses_pupper_original_shell(self) -> bool:
        return self.geometry_mode == "pupper_original_shell"

    @property
    def pupper_shell_design(self):
        """Return the analytic compact design used by the Pupper mode."""

        if not self.uses_pupper_original_shell:
            raise ValueError("Pupper shell design requested for regular geometry")
        from .pupper_shell_geometry import PupperShellDesign

        return PupperShellDesign(
            hip_center_distance=self.torso_length,
            upper_leg_length=self.upper_length,
            lower_leg_length=self.lower_length,
            motor_envelope_radius=self.motor_radius,
            foot_radius=self.foot_radius,
            compact_foot_center_distance=self.compact_foot_center_distance,
            shell_outer_radius=self.shell_contact_radius,
            shell_capsule_radius=self.shell_capsule_radius,
            shell_segments=self.shell_segments_full_circle,
        )

    @property
    def pupper_compact_solution(self):
        from .pupper_shell_geometry import solve_compact_geometry

        return solve_compact_geometry(self.pupper_shell_design)

    @property
    def regular_pentagon_radius(self) -> float:
        return self.edge_length / (2.0 * math.sin(math.pi / 5.0))

    @property
    def regular_pentagon_apothem(self) -> float:
        return self.regular_pentagon_radius * math.cos(math.pi / 5.0)

    @property
    def shell_contact_radius(self) -> float:
        """Radius of the compact pose's intended external contact circle."""
        if self.shell_contact_radius_override is not None:
            return self.shell_contact_radius_override
        if self.shell_motor_clearance is not None:
            return (
                self.regular_pentagon_radius
                + self.motor_radius
                + self.shell_motor_clearance
                + 2.0 * self.shell_capsule_radius
            )
        return self.regular_pentagon_radius + self.foot_radius

    @property
    def shell_centerline_radius(self) -> float:
        """Radius traced by the centers of the shell capsules."""

        return self.shell_contact_radius - self.shell_capsule_radius

    @property
    def shell_centerline_offset(self) -> float:
        """Radial offset from a pentagon vertex to a shell arc endpoint."""

        return self.shell_centerline_radius - self.regular_pentagon_radius

    @property
    def shell_arc_trim_angle(self) -> float:
        """Trim at each arc end to preserve the collinear-link shell gap."""

        required_centerline_gap = (
            2.0 * self.shell_capsule_radius + self.shell_design_gap
        )
        sine_argument = (
            self.edge_length - required_centerline_gap
        ) / (2.0 * self.shell_centerline_radius)
        if not 0.0 < sine_argument < 1.0:
            raise ValueError("shell gap cannot be realized by trimming the arc")
        trim = math.pi / 5.0 - math.asin(sine_argument)
        if trim < 0.0:
            raise ValueError("shell design gap would require extending the arc")
        return trim

    @property
    def shell_arc_coverage_angle(self) -> float:
        if self.shell_arc_coverage_angle_override is not None:
            return self.shell_arc_coverage_angle_override
        return 2.0 * (math.pi / 5.0 - self.shell_arc_trim_angle)

    @property
    def compact_foot_center_distance(self) -> float:
        return 2.0 * self.foot_radius + self.compact_foot_surface_gap

    @property
    def compact_hip_angle(self) -> float:
        if self.uses_pupper_original_shell:
            return self.pupper_compact_solution.hip_angle
        return self.compact_hip_angle_regular

    @property
    def compact_knee_angle(self) -> float:
        """Symmetric knee angle making the two foot surfaces just touch."""

        if self.uses_pupper_original_shell:
            return self.pupper_compact_solution.knee_angle

        target_front_foot_x = self.compact_foot_center_distance / 2.0
        sine_lower_angle = (
            self.hip_half_span
            + self.upper_length * math.sin(self.compact_hip_angle)
            - target_front_foot_x
        ) / self.lower_length
        if not -1.0 <= sine_lower_angle <= 1.0:
            raise ValueError("compact foot spacing is unreachable")
        lower_absolute_angle = math.asin(sine_lower_angle)
        return self.compact_hip_angle + lower_absolute_angle

    @property
    def total_mass(self) -> float:
        return self.torso_mass + 2.0 * self.thigh.mass + 2.0 * self.shank.mass

    @property
    def open_root_height(self) -> float:
        return self.upper_length + self.lower_length + self.foot_radius

    @property
    def compact_root_height(self) -> float:
        if self.uses_pupper_original_shell:
            if self.pupper_shank_shell_foot_clearance > 0.0:
                return (
                    self.pupper_compact_solution.foot_below_hip
                    + self.foot_radius
                )
            return (
                self.shell_contact_radius
                + self.pupper_compact_solution.shell_center_below_hip
            )
        lower_absolute_angle = self.compact_knee_angle - self.compact_hip_angle
        foot_supported_height = (
            self.upper_length * math.cos(self.compact_hip_angle)
            + self.lower_length * math.cos(lower_absolute_angle)
            + self.foot_radius
        )
        shell_supported_height = (
            self.regular_pentagon_apothem + self.shell_contact_radius
        )
        return max(foot_supported_height, shell_supported_height)

    def leg_extension_height(self, hip_angle: float, knee_angle: float) -> float:
        """Vertical hip-to-foot distance for the planar joint convention."""

        return (
            self.upper_length * math.cos(hip_angle)
            + self.lower_length * math.cos(knee_angle - hip_angle)
        )

    @property
    def walk_root_height(self) -> float:
        """Root height placing the lower foot of the walk pose on the floor."""

        front_height = self.leg_extension_height(
            self.walk_front_hip_angle,
            self.walk_front_knee_angle,
        )
        rear_height = self.leg_extension_height(
            self.walk_rear_hip_angle,
            self.walk_rear_knee_angle,
        )
        return max(front_height, rear_height) + self.foot_radius

    @property
    def stand_3d_root_height(self) -> float:
        """Root height placing all four 3-D feet on the floor."""

        front_height = self.leg_extension_height(
            self.stand_3d_front_hip_angle,
            self.stand_3d_front_knee_angle,
        )
        rear_height = self.leg_extension_height(
            self.stand_3d_rear_hip_angle,
            self.stand_3d_rear_knee_angle,
        )
        return max(front_height, rear_height) + self.foot_radius

    @property
    def side_rail_half_width(self) -> float:
        """Half-width of the first 3-D curl side-rail layout."""

        return (
            self.source_torso_width / 2.0
            if self.side_rail_half_width_override is None
            else self.side_rail_half_width_override
        )


FIXED_PARAMETERS = FixedParameters()

# Geometry-development branch.  Keep FIXED_PARAMETERS unchanged so existing
# 2-D references, policies, and friction experiments continue to use the
# validated 150 mm model.
REAL_GEOMETRY_PARAMETERS = replace(
    FIXED_PARAMETERS,
    edge_length=0.180,
    torso_box_outward_offset=0.020,
    upper_proxy_radius=0.025,
    lower_proxy_radius=0.025,
    foot_radius=0.030,
    shell_design_gap=0.0,
    shell_motor_clearance=0.003,
    shank_shell_foot_retreat=0.010,
)

# Pupper-link geometry with the analytically shifted circular shell.  This is
# a separate mode so the established 150 mm and real-geometry references keep
# their original kinematics and collision model.
PUPPER_ORIGINAL_SHELL_PARAMETERS = replace(
    FIXED_PARAMETERS,
    geometry_mode="pupper_original_shell",
    torso_length_override=0.15040,
    upper_length_override=FIXED_PARAMETERS.source_upper_length,
    lower_length_override=FIXED_PARAMETERS.source_lower_projected_length,
    foot_radius=0.0195,
    motor_radius=0.032,
    motor_half_thickness_y=0.0165,
    compact_foot_surface_gap=0.004,
    shell_contact_radius_override=0.1275,
    shell_capsule_radius=0.003,
    pupper_shank_shell_foot_clearance=0.033,
    torso_box_height=0.09014,
)

PUPPER_ORIGINAL_SHELL_60_PARAMETERS = replace(
    PUPPER_ORIGINAL_SHELL_PARAMETERS,
    pupper_shank_shell_foot_clearance=0.010,
    side_rail_half_width_override=0.060,
    pupper_torso_shell_coverage_angle=math.radians(150.0),
)
