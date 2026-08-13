"""MuJoCo generator for the original-dimension Pupper shell concept."""

from __future__ import annotations

import math
from pathlib import Path

from .pupper_shell_geometry import (
    PupperShellDesign,
    solve_compact_geometry,
)


def _f(value: float) -> str:
    return f"{value:.10g}"


def _rotate_to_local(
    point: tuple[float, float],
    origin: tuple[float, float],
    body_angle: float,
) -> tuple[float, float]:
    """Map world compact x/z coordinates into one body's local x/z frame."""

    dx, dz = point[0] - origin[0], point[1] - origin[1]
    cosine, sine = math.cos(body_angle), math.sin(body_angle)
    return cosine * dx - sine * dz, sine * dx + cosine * dz


def _angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _shell_geoms(design: PupperShellDesign):
    """Split one compact circular centerline among the five moving bodies."""

    p = design
    s = solve_compact_geometry(p)
    center = (0.0, -s.shell_center_below_hip)
    right_hip = (p.hip_half_distance, 0.0)
    right_knee = (s.knee_x, -s.knee_below_hip)
    right_foot = (s.foot_x, -s.foot_below_hip)
    left_hip = (-right_hip[0], right_hip[1])
    left_knee = (-right_knee[0], right_knee[1])
    left_foot = (-right_foot[0], right_foot[1])

    def polar(point):
        return math.atan2(point[1] - center[1], point[0] - center[0])

    anchors = {
        "torso": math.pi / 2.0,
        "front_thigh": polar(((right_hip[0] + right_knee[0]) / 2.0,
                              (right_hip[1] + right_knee[1]) / 2.0)),
        "front_shank": polar(((right_knee[0] + right_foot[0]) / 2.0,
                              (right_knee[1] + right_foot[1]) / 2.0)),
        "rear_shank": polar(((left_knee[0] + left_foot[0]) / 2.0,
                             (left_knee[1] + left_foot[1]) / 2.0)),
        "rear_thigh": polar(((left_hip[0] + left_knee[0]) / 2.0,
                             (left_hip[1] + left_knee[1]) / 2.0)),
    }
    frames = {
        "torso": ((0.0, 0.0), 0.0),
        "front_thigh": (right_hip, -s.hip_angle),
        "front_shank": (right_knee, s.knee_angle - s.hip_angle),
        "rear_thigh": (left_hip, s.hip_angle),
        "rear_shank": (left_knee, s.hip_angle - s.knee_angle),
    }
    grouped = {name: [] for name in anchors}
    radius = p.shell_centerline_radius
    for index in range(p.shell_segments):
        angle_a = 2.0 * math.pi * index / p.shell_segments
        angle_b = 2.0 * math.pi * (index + 1) / p.shell_segments
        middle = 0.5 * (angle_a + angle_b)
        body = min(anchors, key=lambda name: _angular_distance(middle, anchors[name]))
        point_a = (
            center[0] + radius * math.cos(angle_a),
            center[1] + radius * math.sin(angle_a),
        )
        point_b = (
            center[0] + radius * math.cos(angle_b),
            center[1] + radius * math.sin(angle_b),
        )
        origin, angle = frames[body]
        local_a = _rotate_to_local(point_a, origin, angle)
        local_b = _rotate_to_local(point_b, origin, angle)
        grouped[body].append(
            f'<geom name="{body}_shell_{index:02d}" class="rolling_shell" '
            f'fromto="{_f(local_a[0])} 0 {_f(local_a[1])} '
            f'{_f(local_b[0])} 0 {_f(local_b[1])}"/>'
        )
    return {name: "\n".join(lines) for name, lines in grouped.items()}


def build_pupper_shell_mjcf(
    design: PupperShellDesign = PupperShellDesign(),
) -> str:
    p = design
    s = solve_compact_geometry(p)
    shell = _shell_geoms(p)
    # The design center lies below the hip line.  Put the outer circle's
    # lowest point exactly on the ground.
    root_height = p.shell_outer_radius + s.shell_center_below_hip
    thigh_mass = 0.732
    shank_mass = 0.100
    torso_mass = 1.506
    upper_com = 0.38204 * p.upper_leg_length
    lower_com = 0.59864 * p.lower_leg_length
    hip_limit = "-1.12 2.41"
    knee_limit = "-0.61 2.69"
    return f'''<mujoco model="pupper_original_shell_2d">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"
          cone="elliptic" iterations="30" ls_iterations="10"/>
  <visual><global azimuth="90" elevation="0" offwidth="960" offheight="720"/>
    <rgba contactpoint="1 0.1 0.1 1" contactforce="1 0.7 0.1 1"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1=".18 .23 .30" rgb2=".02 .03 .05" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".18 .22 .27" rgb2=".10 .13 .17"/>
    <material name="ground" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".05"/></asset>
  <default>
    <joint limited="true" damping="0.02" armature="0.0032"/>
    <geom condim="3" friction="0.8 0.02 0.01" solref="0.003 1" solimp="0.95 0.99 0.001"/>
    <default class="structure"><geom contype="2" conaffinity="7"/></default>
    <default class="rolling_shell"><geom type="capsule" size="{_f(p.shell_capsule_radius)}" contype="4" conaffinity="3" group="1" rgba="0.55 0.78 0.95 .88"/></default>
    <default class="servo"><general forcelimited="true" forcerange="-6 6" gainprm="10 0 0" biastype="affine" biasprm="0 -10 -.2"/></default>
  </default>
  <worldbody>
    <light name="key" pos="-.4 -1 1.2" dir=".2 .8 -1" diffuse=".9 .9 .9"/>
    <camera name="side" pos="0 -.65 .14" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" size="2 2 .05" contype="1" conaffinity="7" material="ground"/>
    <body name="torso">
      <joint name="root_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="root_z" type="slide" axis="0 0 1" limited="false"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" limited="false"/>
      <inertial pos=".025 0 .015" mass="{_f(torso_mass)}" diaginertia=".0085 .0085 .0024"/>
      <geom name="torso_proxy" type="box" class="structure" pos="0 0 .015" size=".060 .012 .045" rgba=".12 .48 .88 1"/>
      {shell['torso']}
      <body name="front_thigh" pos="{_f(p.hip_half_distance)} 0 0">
        <joint name="front_hip" type="hinge" axis="0 -1 0" range="{hip_limit}"/>
        <inertial pos="0 0 -{_f(upper_com)}" mass="{_f(thigh_mass)}" diaginertia=".002673 .002673 .0002"/>
        <geom name="front_hip_motor" type="cylinder" class="structure" size="{_f(p.motor_envelope_radius)} .0165" euler="1.570796327 0 0" rgba=".30 .32 .36 1"/>
        <geom name="front_thigh_proxy" type="capsule" class="structure" fromto="0 0 0 0 0 -{_f(p.upper_leg_length)}" size=".012" rgba=".95 .45 .12 1"/>
        {shell['front_thigh']}
        <body name="front_shank" pos="0 0 -{_f(p.upper_leg_length)}">
          <joint name="front_knee" type="hinge" axis="0 1 0" range="{knee_limit}"/>
          <inertial pos="0 0 -{_f(lower_com)}" mass="{_f(shank_mass)}" diaginertia=".000155075 .000155075 .00002"/>
          <geom name="front_knee_motor" type="cylinder" class="structure" size="{_f(p.motor_envelope_radius)} .0165" euler="1.570796327 0 0" rgba=".30 .32 .36 1"/>
          <geom name="front_shank_proxy" type="capsule" class="structure" fromto="0 0 0 0 0 -{_f(p.lower_leg_length)}" size=".010" rgba=".98 .70 .18 1"/>
          {shell['front_shank']}
          <geom name="front_foot_proxy" type="sphere" class="structure" pos="0 0 -{_f(p.lower_leg_length)}" size="{_f(p.foot_radius)}" rgba=".85 .85 .88 1"/>
        </body>
      </body>
      <body name="rear_thigh" pos="-{_f(p.hip_half_distance)} 0 0">
        <joint name="rear_hip" type="hinge" axis="0 1 0" range="{hip_limit}"/>
        <inertial pos="0 0 -{_f(upper_com)}" mass="{_f(thigh_mass)}" diaginertia=".002673 .002673 .0002"/>
        <geom name="rear_hip_motor" type="cylinder" class="structure" size="{_f(p.motor_envelope_radius)} .0165" euler="1.570796327 0 0" rgba=".30 .32 .36 1"/>
        <geom name="rear_thigh_proxy" type="capsule" class="structure" fromto="0 0 0 0 0 -{_f(p.upper_leg_length)}" size=".012" rgba=".84 .24 .18 1"/>
        {shell['rear_thigh']}
        <body name="rear_shank" pos="0 0 -{_f(p.upper_leg_length)}">
          <joint name="rear_knee" type="hinge" axis="0 -1 0" range="{knee_limit}"/>
          <inertial pos="0 0 -{_f(lower_com)}" mass="{_f(shank_mass)}" diaginertia=".000155075 .000155075 .00002"/>
          <geom name="rear_knee_motor" type="cylinder" class="structure" size="{_f(p.motor_envelope_radius)} .0165" euler="1.570796327 0 0" rgba=".30 .32 .36 1"/>
          <geom name="rear_shank_proxy" type="capsule" class="structure" fromto="0 0 0 0 0 -{_f(p.lower_leg_length)}" size=".010" rgba=".94 .55 .16 1"/>
          {shell['rear_shank']}
          <geom name="rear_foot_proxy" type="sphere" class="structure" pos="0 0 -{_f(p.lower_leg_length)}" size="{_f(p.foot_radius)}" rgba=".85 .85 .88 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <contact><pair name="front_rear_foot_contact" geom1="front_foot_proxy" geom2="rear_foot_proxy" condim="3" friction=".8 .02 .01" solref=".002 1" solimp=".97 .995 .001"/></contact>
  <actuator>
    <general name="front_hip_servo" joint="front_hip" class="servo" ctrlrange="{hip_limit}"/>
    <general name="front_knee_servo" joint="front_knee" class="servo" ctrlrange="{knee_limit}"/>
    <general name="rear_hip_servo" joint="rear_hip" class="servo" ctrlrange="{hip_limit}"/>
    <general name="rear_knee_servo" joint="rear_knee" class="servo" ctrlrange="{knee_limit}"/>
  </actuator>
  <keyframe><key name="compact" qpos="0 {_f(root_height)} 0 {_f(s.hip_angle)} {_f(s.knee_angle)} {_f(s.hip_angle)} {_f(s.knee_angle)}" ctrl="{_f(s.hip_angle)} {_f(s.knee_angle)} {_f(s.hip_angle)} {_f(s.knee_angle)}"/></keyframe>
</mujoco>'''


def write_pupper_shell_mjcf(
    path: Path,
    design: PupperShellDesign = PupperShellDesign(),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_pupper_shell_mjcf(design), encoding="utf-8")
    return path
