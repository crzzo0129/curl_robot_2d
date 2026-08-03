"""Fixed parameters for the first planar Pupper abstraction.

The values in this module are deliberately split into source-derived robot
parameters and provisional simulation parameters.  The first arc-shell
geometry is parameterized here, but it does not yet add mass or inertia.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    compact_hip_angle: float = 0.3141592653589793
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
        return self.edge_length

    @property
    def hip_half_span(self) -> float:
        return self.edge_length / 2.0

    @property
    def upper_length(self) -> float:
        return self.edge_length

    @property
    def lower_length(self) -> float:
        return self.edge_length

    @property
    def regular_pentagon_radius(self) -> float:
        return self.edge_length / (2.0 * math.sin(math.pi / 5.0))

    @property
    def regular_pentagon_apothem(self) -> float:
        return self.regular_pentagon_radius * math.cos(math.pi / 5.0)

    @property
    def shell_contact_radius(self) -> float:
        """Radius of the compact pose's intended external contact circle."""

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
        return 2.0 * (math.pi / 5.0 - self.shell_arc_trim_angle)

    @property
    def compact_foot_center_distance(self) -> float:
        return 2.0 * self.foot_radius + self.compact_foot_surface_gap

    @property
    def compact_knee_angle(self) -> float:
        """Symmetric knee angle making the two foot surfaces just touch."""

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
        lower_absolute_angle = self.compact_knee_angle - self.compact_hip_angle
        return (
            self.upper_length * math.cos(self.compact_hip_angle)
            + self.lower_length * math.cos(lower_absolute_angle)
            + self.foot_radius
        )

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

        return self.source_torso_width / 2.0


FIXED_PARAMETERS = FixedParameters()
