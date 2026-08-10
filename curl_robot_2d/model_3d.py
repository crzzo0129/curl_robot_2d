"""Generate the first 3-D curl-robot MuJoCo model."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import dedent

from .parameters import FIXED_PARAMETERS, FixedParameters


JOINT_NAMES_3D = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
FOOT_SITE_NAMES_3D = (
    "front_left_foot_site",
    "front_right_foot_site",
    "rear_left_foot_site",
    "rear_right_foot_site",
)


def _f(value: float) -> str:
    return f"{value:.10g}"


def _arc_shell_geoms_3d(
    prefix: str,
    start: tuple[float, float],
    end: tuple[float, float],
    outward: tuple[float, float],
    parameters: FixedParameters,
    *,
    y: float = 0.0,
    indent: int = 14,
    end_retreat: float = 0.0,
) -> str:
    """Return capsule geoms for one shell arc in a body's local x-z plane."""

    p = parameters
    dx = end[0] - start[0]
    dz = end[1] - start[1]
    length = math.hypot(dx, dz)
    tangent = (dx / length, dz / length)
    normal_length = math.hypot(*outward)
    normal = (outward[0] / normal_length, outward[1] / normal_length)
    if not math.isclose(length, p.edge_length, abs_tol=1e-12):
        raise ValueError(f"{prefix} shell chord does not match edge_length")
    if not math.isclose(
        tangent[0] * normal[0] + tangent[1] * normal[1],
        0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{prefix} shell outward normal is not perpendicular")

    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    circle_center = (
        midpoint[0] - p.regular_pentagon_apothem * normal[0],
        midpoint[1] - p.regular_pentagon_apothem * normal[1],
    )
    trimmed_half_angle = math.pi / 5.0 - p.shell_arc_trim_angle
    end_angle = trimmed_half_angle - end_retreat / p.shell_centerline_radius
    if end_angle <= -trimmed_half_angle:
        raise ValueError(f"{prefix} shell end retreat removes the full arc")
    points: list[tuple[float, float]] = []
    for index in range(p.shell_segments_per_edge + 1):
        fraction = index / p.shell_segments_per_edge
        angle = (
            -trimmed_half_angle
            + (end_angle + trimmed_half_angle) * fraction
        )
        point = (
            circle_center[0]
            + p.shell_centerline_radius
            * (math.sin(angle) * tangent[0] + math.cos(angle) * normal[0]),
            circle_center[1]
            + p.shell_centerline_radius
            * (math.sin(angle) * tangent[1] + math.cos(angle) * normal[1]),
        )
        points.append(point)

    pad = " " * indent
    lines = []
    for index, (point_a, point_b) in enumerate(zip(points, points[1:])):
        lines.append(
            f'{pad}<geom name="{prefix}_shell_{index:02d}" '
            f'class="rolling_shell"\n'
            f'{pad}      fromto="{_f(point_a[0])} {_f(y)} {_f(point_a[1])} '
            f'{_f(point_b[0])} {_f(y)} {_f(point_b[1])}"/>'
        )
    return "\n".join(lines)


def _motor_geoms_3d(
    prefix: str,
    parameters: FixedParameters,
    *,
    indent: int,
) -> str:
    """Return an exact motor visual plus MJX-compatible collision proxies.

    JAX MJX does not implement cylinder-box contacts.  Four capsules arranged
    as diameters in the motor's x-z disk approximate the cylindrical collision
    volume while preserving its 24 mm y thickness.  The largest radial deficit
    is about 1.1 mm for the current 54 x 24 mm motor.
    """

    p = parameters
    capsule_radius = p.motor_half_thickness_y
    segment_half_length = p.motor_radius - capsule_radius
    if segment_half_length <= 0.0:
        raise ValueError("motor radius must exceed its axial half-thickness")

    pad = " " * indent
    lines = [
        f'{pad}<geom name="{prefix}_motor" type="cylinder"',
        f'{pad}      size="{_f(p.motor_radius)} {_f(p.motor_half_thickness_y)}"',
        f'{pad}      euler="1.570796327 0 0" contype="0" conaffinity="0"',
        f'{pad}      rgba="0.30 0.32 0.36 1"/>',
    ]
    for index in range(4):
        angle = index * math.pi / 4.0
        dx = segment_half_length * math.cos(angle)
        dz = segment_half_length * math.sin(angle)
        lines.extend(
            [
                f'{pad}<geom name="{prefix}_motor_collision_{index:02d}" type="capsule"',
                f'{pad}      class="structure_collision"',
                f'{pad}      fromto="{_f(-dx)} 0 {_f(-dz)} {_f(dx)} 0 {_f(dz)}"',
                f'{pad}      size="{_f(capsule_radius)}" rgba="0 0 0 0"/>',
            ]
        )
    return "\n".join(lines)


def _leg_chain(
    *,
    side: str,
    front: bool,
    y: float,
    parameters: FixedParameters,
    detailed_structure: bool = False,
) -> str:
    p = parameters
    x = p.hip_half_span if front else -p.hip_half_span
    prefix = f"{'front' if front else 'rear'}_{side}"
    thigh_shell_outward = (1.0, 0.0) if front else (-1.0, 0.0)
    shank_shell_outward = thigh_shell_outward
    hip_axis = "0 -1 0" if front else "0 1 0"
    knee_axis = "0 1 0" if front else "0 -1 0"
    thigh_mass = p.thigh.mass / 2.0
    shank_mass = p.shank.mass / 2.0
    thigh_inertia = (
        max(p.thigh.planar_inertia / 2.0, 1.0e-6),
        max(p.thigh.planar_inertia / 2.0, 1.0e-6),
        0.0001,
    )
    shank_inertia = (
        max(p.shank.planar_inertia / 2.0, 1.0e-6),
        max(p.shank.planar_inertia / 2.0, 1.0e-6),
        0.00002,
    )
    thigh_start = p.motor_radius + p.upper_proxy_radius + p.motor_link_clearance
    thigh_end = p.upper_length - thigh_start
    shank_start = p.motor_radius + p.lower_proxy_radius + p.motor_link_clearance
    shank_end = (
        p.lower_length
        - p.foot_radius
        - p.lower_proxy_radius
        - p.motor_link_clearance
    )
    thigh_fromto = (
        f"0 0 -{_f(thigh_start)} 0 0 -{_f(thigh_end)}"
        if detailed_structure
        else f"0 0 0 0 0 -{_f(p.upper_length)}"
    )
    shank_fromto = (
        f"0 0 -{_f(shank_start)} 0 0 -{_f(shank_end)}"
        if detailed_structure
        else f"0 0 0 0 0 -{_f(p.lower_length)}"
    )
    hip_motor = (
        "\n" + _motor_geoms_3d(f"{prefix}_hip", p, indent=16)
        if detailed_structure else ""
    )
    knee_motor = (
        "\n" + _motor_geoms_3d(f"{prefix}_knee", p, indent=18)
        if detailed_structure else ""
    )
    return f"""\
              <body name="{prefix}_thigh" pos="{_f(x)} {_f(y)} 0">
                <joint name="{prefix}_hip" type="hinge" axis="{hip_axis}"
                       class="single_hip_joint"/>
                <inertial pos="0 0 -{_f(p.thigh.com_along_link)}"
                          mass="{_f(thigh_mass)}"
                          diaginertia="{' '.join(_f(v) for v in thigh_inertia)}"/>
                <geom name="{prefix}_thigh_proxy" type="capsule"
                      class="structure_collision"
                      fromto="{thigh_fromto}"
                      size="{_f(p.upper_proxy_radius)}"
                      rgba="0.95 0.45 0.12 1"/>{hip_motor}
{_arc_shell_geoms_3d(prefix + "_thigh", (0.0, 0.0), (0.0, -p.upper_length), thigh_shell_outward, p, indent=16)}
                <body name="{prefix}_shank" pos="0 0 -{_f(p.upper_length)}">
                  <joint name="{prefix}_knee" type="hinge" axis="{knee_axis}"
                         class="single_knee_joint"/>
                  <inertial pos="0 0 -{_f(p.shank.com_along_link)}"
                            mass="{_f(shank_mass)}"
                            diaginertia="{' '.join(_f(v) for v in shank_inertia)}"/>
                  <geom name="{prefix}_shank_proxy" type="capsule"
                        class="structure_collision"
                        fromto="{shank_fromto}"
                        size="{_f(p.lower_proxy_radius)}"
                        rgba="0.98 0.70 0.18 1"/>{knee_motor}
{_arc_shell_geoms_3d(prefix + "_shank", (0.0, 0.0), (0.0, -p.lower_length), shank_shell_outward, p, indent=18, end_retreat=(p.shank_shell_foot_retreat if detailed_structure else 0.0))}
                  <geom name="{prefix}_foot_proxy" type="sphere"
                        class="structure_collision"
                        pos="0 0 -{_f(p.lower_length)}"
                        size="{_f(p.foot_radius)}"
                        rgba="0.85 0.85 0.88 1"/>
                  <site name="{prefix}_foot_site"
                        pos="0 0 -{_f(p.lower_length)}" size="0.004"/>
                </body>
              </body>"""


def build_mjcf_3d(
    parameters: FixedParameters = FIXED_PARAMETERS,
    *,
    detailed_structure: bool = False,
) -> str:
    p = parameters
    torso_half_length = p.torso_length / 2.0
    side_y = (
        p.torso_box_width / 2.0
        if detailed_structure
        else p.side_rail_half_width
    )
    single_force_limit = p.hip.force_limit / 2.0
    single_kp = p.hip.kp / 2.0
    single_kd = p.hip.kd / 2.0
    single_armature = p.hip.armature / 2.0
    single_damping = p.hip.damping / 2.0
    torso_inertia = (
        max(p.torso_mass * p.source_torso_width * p.source_torso_width / 12.0, 1.0e-5),
        p.torso_planar_inertia,
        max(p.torso_planar_inertia, 1.0e-5),
    )
    compact = (p.compact_hip_angle, p.compact_knee_angle)
    stand_front = (
        p.stand_3d_front_hip_angle,
        p.stand_3d_front_knee_angle,
    )
    stand_rear = (
        p.stand_3d_rear_hip_angle,
        p.stand_3d_rear_knee_angle,
    )
    park_key = ""
    if detailed_structure:
        # Validated by scripts/validate_3d_park_pose.py on the real-geometry
        # candidate.  Keep the original 150 mm baseline keyframes unchanged.
        park_root = 0.3671041158
        park_front = (0.4983797927, 0.3491569025)
        park_rear = (0.4340952810, 0.0107701707)
        park_key = f'''
            <key name="park"
                 qpos="0 0 {_f(park_root)} 1 0 0 0 {_f(park_front[0])} {_f(park_front[1])} {_f(park_front[0])} {_f(park_front[1])} {_f(park_rear[0])} {_f(park_rear[1])} {_f(park_rear[0])} {_f(park_rear[1])}"
                 ctrl="{_f(park_front[0])} {_f(park_front[1])} {_f(park_front[0])} {_f(park_front[1])} {_f(park_rear[0])} {_f(park_rear[1])} {_f(park_rear[0])} {_f(park_rear[1])}"/>'''
    torso_geoms = (
        f'''<geom name="torso_box_proxy" type="box"
                    class="structure_collision"
                    pos="0 0 {_f(p.torso_box_outward_offset - p.torso_box_height / 2.0)}"
                    size="{_f(p.torso_box_width / 2.0)} {_f(p.torso_box_width / 2.0)} {_f(p.torso_box_height / 2.0)}"
                    rgba="0.12 0.48 0.88 1"/>'''
        if detailed_structure
        else f'''<geom name="torso_spine_proxy" type="capsule"
                    class="structure_collision"
                    fromto="-{_f(torso_half_length)} 0 0 {_f(torso_half_length)} 0 0"
                    size="{_f(p.upper_proxy_radius)}"
                    rgba="0.12 0.48 0.88 1"/>
              <geom name="front_crossbar_proxy" type="capsule"
                    class="structure_collision"
                    fromto="{_f(torso_half_length)} -{_f(side_y)} 0 {_f(torso_half_length)} {_f(side_y)} 0"
                    size="{_f(p.lower_proxy_radius)}"
                    rgba="0.12 0.48 0.88 1"/>
              <geom name="rear_crossbar_proxy" type="capsule"
                    class="structure_collision"
                    fromto="-{_f(torso_half_length)} -{_f(side_y)} 0 -{_f(torso_half_length)} {_f(side_y)} 0"
                    size="{_f(p.lower_proxy_radius)}"
                    rgba="0.12 0.48 0.88 1"/>'''
    )
    explicit_foot_pairs = "" if detailed_structure else f'''          <contact>
            <pair name="left_foot_pair"
                  geom1="front_left_foot_proxy" geom2="rear_left_foot_proxy"
                  condim="3" friction="{_f(p.nominal_ground_friction)} 0.02 0.01"
                  solref="0.002 1" solimp="0.97 0.995 0.001"/>
            <pair name="right_foot_pair"
                  geom1="front_right_foot_proxy" geom2="rear_right_foot_proxy"
                  condim="3" friction="{_f(p.nominal_ground_friction)} 0.02 0.01"
                  solref="0.002 1" solimp="0.97 0.995 0.001"/>
          </contact>'''

    return dedent(
        f"""\
        <mujoco model="curl_robot_3d">
          <!--
            Generated 3-D curl robot baseline.

            This is not the disk_robot/Pupper disk model.  It directly lifts the
            2-D equal-edge curl mechanism into 3-D by duplicating the sagittal
            front/rear chains onto left and right side rails.
          -->

          <compiler angle="radian" autolimits="true"/>
          <option timestep="{_f(p.timestep)}" gravity="0 0 -9.81"
                  integrator="implicitfast" cone="elliptic"
                  iterations="20" ls_iterations="10"/>

          <visual>
            <global azimuth="120" elevation="-12" offwidth="960" offheight="720"/>
            <headlight ambient="0.35 0.35 0.35"
                       diffuse="0.65 0.65 0.65"
                       specular="0.20 0.20 0.20"/>
          </visual>

          <asset>
            <texture type="skybox" builtin="gradient" rgb1=".18 .23 .30"
                     rgb2=".02 .03 .05" width="512" height="512"/>
            <texture name="grid" type="2d" builtin="checker" width="512"
                     height="512" rgb1=".18 .22 .27" rgb2=".10 .13 .17"/>
            <material name="ground" texture="grid" texrepeat="1 1"
                      texuniform="true" reflectance="0.05"/>
          </asset>

          <default>
            <joint limited="true"/>
            <geom contype="0" conaffinity="1" condim="3"
                  friction="{_f(p.nominal_ground_friction)} 0.02 0.01"
                  solref="0.01 1" solimp="0.9 0.95 0.001"/>
            <default class="single_hip_joint">
              <joint damping="{_f(single_damping)}"
                     armature="{_f(single_armature)}"
                     range="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            </default>
            <default class="single_knee_joint">
              <joint damping="{_f(single_damping)}"
                     armature="{_f(single_armature)}"
                     range="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
            </default>
            <default class="single_servo">
              <general forcelimited="true"
                       forcerange="-{_f(single_force_limit)} {_f(single_force_limit)}"
                       ctrllimited="true"
                       gainprm="{_f(single_kp)} 0 0"
                       biastype="affine"
                       biasprm="0 -{_f(single_kp)} -{_f(single_kd)}"/>
            </default>
            <default class="rolling_shell">
              <geom type="capsule" size="{_f(p.shell_capsule_radius)}"
                    contype="4" conaffinity="7"
                    solref="0.003 1" solimp="0.95 0.99 0.001"
                    group="1" rgba="0.55 0.78 0.95 0.88"/>
            </default>
            <default class="structure_collision">
              <geom contype="2" conaffinity="7"
                    solref="0.003 1" solimp="0.95 0.99 0.001"/>
            </default>
          </default>

          <worldbody>
            <light name="key" directional="true" castshadow="true"
                   pos="-1 -2 4" dir="0.25 0.15 -1"
                   diffuse="0.85 0.85 0.82" specular="0.20 0.20 0.20"/>
            <light name="fill" directional="true" castshadow="false"
                   pos="1 2 3" dir="-0.20 -0.30 -1"
                   diffuse="0.30 0.36 0.45" specular="0.05 0.05 0.05"/>
            <camera name="tracking" mode="targetbody" target="torso"
                    pos="0.45 -0.55 0.35"/>
            <geom name="floor" type="plane" size="12 6 0.05"
                  contype="1" conaffinity="1" material="ground"/>

            <body name="torso" pos="0 0 0">
              <freejoint name="root"/>
              <inertial pos="{_f(p.torso_com_x)} 0 {_f(p.torso_com_z)}"
                        mass="{_f(p.torso_mass)}"
                        diaginertia="{' '.join(_f(v) for v in torso_inertia)}"/>
              {torso_geoms}
{_arc_shell_geoms_3d("torso_left", (-torso_half_length, 0.0), (torso_half_length, 0.0), (0.0, 1.0), p, y=side_y)}
{_arc_shell_geoms_3d("torso_right", (-torso_half_length, 0.0), (torso_half_length, 0.0), (0.0, 1.0), p, y=-side_y)}

{_leg_chain(side="left", front=True, y=side_y, parameters=p, detailed_structure=detailed_structure)}
{_leg_chain(side="right", front=True, y=-side_y, parameters=p, detailed_structure=detailed_structure)}
{_leg_chain(side="left", front=False, y=side_y, parameters=p, detailed_structure=detailed_structure)}
{_leg_chain(side="right", front=False, y=-side_y, parameters=p, detailed_structure=detailed_structure)}
            </body>
          </worldbody>

{explicit_foot_pairs}

          <actuator>
            <general name="front_left_hip_servo" joint="front_left_hip"
                     class="single_servo"
                     ctrlrange="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            <general name="front_left_knee_servo" joint="front_left_knee"
                     class="single_servo"
                     ctrlrange="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
            <general name="front_right_hip_servo" joint="front_right_hip"
                     class="single_servo"
                     ctrlrange="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            <general name="front_right_knee_servo" joint="front_right_knee"
                     class="single_servo"
                     ctrlrange="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
            <general name="rear_left_hip_servo" joint="rear_left_hip"
                     class="single_servo"
                     ctrlrange="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            <general name="rear_left_knee_servo" joint="rear_left_knee"
                     class="single_servo"
                     ctrlrange="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
            <general name="rear_right_hip_servo" joint="rear_right_hip"
                     class="single_servo"
                     ctrlrange="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            <general name="rear_right_knee_servo" joint="rear_right_knee"
                     class="single_servo"
                     ctrlrange="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
          </actuator>

          <sensor>
            <framequat name="torso_quat" objtype="body" objname="torso"/>
            <framepos name="torso_position" objtype="body" objname="torso"/>
            <framelinvel name="torso_velocity" objtype="body" objname="torso"/>
            <frameangvel name="torso_angular_velocity" objtype="body" objname="torso"/>
            <jointpos name="front_left_hip_position" joint="front_left_hip"/>
            <jointpos name="front_left_knee_position" joint="front_left_knee"/>
            <jointpos name="front_right_hip_position" joint="front_right_hip"/>
            <jointpos name="front_right_knee_position" joint="front_right_knee"/>
            <jointpos name="rear_left_hip_position" joint="rear_left_hip"/>
            <jointpos name="rear_left_knee_position" joint="rear_left_knee"/>
            <jointpos name="rear_right_hip_position" joint="rear_right_hip"/>
            <jointpos name="rear_right_knee_position" joint="rear_right_knee"/>
          </sensor>

          <keyframe>
            <key name="open"
                 qpos="0 0 {_f(p.open_root_height)} 1 0 0 0 0 0 0 0 0 0 0 0"
                 ctrl="0 0 0 0 0 0 0 0"/>
            <key name="stand"
                 qpos="0 0 {_f(p.stand_3d_root_height)} 1 0 0 0 {_f(stand_front[0])} {_f(stand_front[1])} {_f(stand_front[0])} {_f(stand_front[1])} {_f(stand_rear[0])} {_f(stand_rear[1])} {_f(stand_rear[0])} {_f(stand_rear[1])}"
                 ctrl="{_f(stand_front[0])} {_f(stand_front[1])} {_f(stand_front[0])} {_f(stand_front[1])} {_f(stand_rear[0])} {_f(stand_rear[1])} {_f(stand_rear[0])} {_f(stand_rear[1])}"/>{park_key}
            <key name="compact"
                 qpos="0 0 {_f(p.compact_root_height)} 1 0 0 0 {_f(compact[0])} {_f(compact[1])} {_f(compact[0])} {_f(compact[1])} {_f(compact[0])} {_f(compact[1])} {_f(compact[0])} {_f(compact[1])}"
                 ctrl="{_f(compact[0])} {_f(compact[1])} {_f(compact[0])} {_f(compact[1])} {_f(compact[0])} {_f(compact[1])} {_f(compact[0])} {_f(compact[1])}"/>
          </keyframe>
        </mujoco>
        """
    )


def write_mjcf_3d(
    path: Path,
    parameters: FixedParameters = FIXED_PARAMETERS,
    *,
    detailed_structure: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_mjcf_3d(parameters, detailed_structure=detailed_structure),
        encoding="utf-8",
    )
    return path
