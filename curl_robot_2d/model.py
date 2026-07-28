"""Generate the first MuJoCo model for the planar curl study."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import dedent

from .parameters import FIXED_PARAMETERS, FixedParameters


def _f(value: float) -> str:
    return f"{value:.10g}"


def _arc_shell_geoms(
    prefix: str,
    start: tuple[float, float],
    end: tuple[float, float],
    outward: tuple[float, float],
    parameters: FixedParameters,
    *,
    indent: int = 14,
) -> str:
    """Return capsule geoms for one shell arc in a body's local x-z plane.

    ``start`` and ``end`` are the link's two joint-center points. ``outward``
    is the unit normal pointing away from the compact pentagon interior.
    """

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
    half_angle = math.pi / 5.0
    trimmed_half_angle = half_angle - p.shell_arc_trim_angle
    points: list[tuple[float, float]] = []
    for index in range(p.shell_segments_per_edge + 1):
        fraction = index / p.shell_segments_per_edge
        angle = -trimmed_half_angle + 2.0 * trimmed_half_angle * fraction
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
            f'{pad}      fromto="{_f(point_a[0])} 0 {_f(point_a[1])} '
            f'{_f(point_b[0])} 0 {_f(point_b[1])}"/>'
        )
    return "\n".join(lines)


def build_mjcf(parameters: FixedParameters = FIXED_PARAMETERS) -> str:
    p = parameters
    torso_half_length = p.torso_length / 2

    # Only the out-of-plane inertia is dynamically active.  The remaining
    # diagonal terms are positive proxy values satisfying MuJoCo's 3-D inertia
    # requirements.
    torso_inertia = (
        p.torso_planar_inertia,
        p.torso_planar_inertia,
        0.0024,
    )
    thigh_inertia = (
        p.thigh.planar_inertia,
        p.thigh.planar_inertia,
        0.0002,
    )
    shank_inertia = (
        p.shank.planar_inertia,
        p.shank.planar_inertia,
        0.00002,
    )

    return dedent(
        f"""\
        <mujoco model="curl_robot_2d">
          <!--
            自动生成文件。请修改 curl_robot_2d/parameters.py 或
            curl_robot_2d/model.py，然后运行：
              python -m scripts.generate_model

            坐标约定：
              x = 前进方向，z = 竖直向上，y = 垂直侧视平面。

            根节点具有 x/z 平移和绕 y 的俯仰；四个内部关节也只绕 y 转动。
            因此这是在 MuJoCo 三维动力学引擎中实现的严格 7-DOF 平面模型。

            等边基线：
              Torso 髋间边 = 前/后大腿 = 前/后小腿 = {_f(p.edge_length)} m
            compact 关键帧保持五条中心线边等长，并令两个有限尺寸足端球
            表面接触而非中心重合。视觉和碰撞几何有意使用简单 primitive。
          -->

          <!-- 使用弧度和 SI 单位，重力沿 -z。 -->
          <compiler angle="radian" autolimits="true"/>
          <option timestep="{_f(p.timestep)}" gravity="0 0 -9.81"
                  integrator="implicitfast" cone="elliptic"
                  iterations="20" ls_iterations="10"/>

          <!-- 以下设置只影响渲染，不影响动力学。 -->
          <visual>
            <global azimuth="90" elevation="0" offwidth="960" offheight="720"/>
            <rgba com="0.1 1 0.1 1" contactpoint="1 0.2 0.1 1"
                  contactforce="1 0.7 0.1 1"/>
          </visual>

          <!-- 程序生成的天空和地面纹理；项目不依赖外部 mesh。 -->
          <asset>
            <texture type="skybox" builtin="gradient" rgb1=".18 .23 .30"
                     rgb2=".02 .03 .05" width="512" height="512"/>
            <texture name="grid" type="2d" builtin="checker" width="512"
                     height="512" rgb1=".18 .22 .27" rgb2=".10 .13 .17"/>
            <material name="ground" texture="grid" texrepeat="1 1"
                      texuniform="true" reflectance="0.05"/>
          </asset>

          <default>
            <!--
              bit 1 = 地面，bit 2 = 结构代理，bit 4 = 弧壳。
              结构和弧壳都与地面以及非相邻机器人刚体碰撞；MuJoCo 默认
              过滤同一 body 和父子 body，避免关节两侧的预期邻接误碰。
            -->
            <joint limited="true"/>
            <geom contype="0" conaffinity="1" condim="3"
                  friction="{_f(p.nominal_ground_friction)} 0.02 0.01"
                  solref="0.01 1" solimp="0.9 0.95 0.001"/>
            <default class="hip_joint">
              <!--
                左右两个同步髋电机合并为一个二维自由度。弧壳端部已经缩短，
                collision-compatible range 恢复为源模型建议安全范围。
              -->
              <joint damping="{_f(p.hip.damping)}"
                     armature="{_f(p.hip.armature)}"
                     range="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            </default>
            <default class="knee_joint">
              <!-- 等效膝关节正角度表示向内蜷缩。 -->
              <joint damping="{_f(p.knee.damping)}"
                     armature="{_f(p.knee.armature)}"
                     range="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
            </default>
            <default class="paired_servo">
              <!--
                左右电机对的等效伺服：最大力矩 +/-6 N m，
                PD 增益取两个源电机之和。
              -->
              <general forcelimited="true" forcerange="-{_f(p.hip.force_limit)} {_f(p.hip.force_limit)}"
                       ctrllimited="true"
                       gainprm="{_f(p.hip.kp)} 0 0"
                       biastype="affine"
                       biasprm="0 -{_f(p.hip.kp)} -{_f(p.hip.kd)}"/>
            </default>
            <default class="rolling_shell">
              <!--
                第一版弧壳由短 capsule 拼接。body 已显式给出 inertial，
                因而这些 geom 目前只改变外形和接触，不增加质量或惯量。
              -->
              <geom type="capsule" size="{_f(p.shell_capsule_radius)}"
                    contype="4" conaffinity="7"
                    solref="0.003 1" solimp="0.95 0.99 0.001"
                    group="1" rgba="0.55 0.78 0.95 0.88"/>
            </default>
            <default class="structure_collision">
              <!-- 有限厚度结构代理；参与地面与选择性机器人自碰撞。 -->
              <geom contype="2" conaffinity="7"
                    solref="0.003 1" solimp="0.95 0.99 0.001"/>
            </default>
          </default>

          <worldbody>
            <!-- 固定侧视相机沿 +y 观察 x-z 平面。 -->
            <light name="key" pos="-0.4 -1.0 1.2" dir="0.2 0.8 -1"
                   diffuse="0.9 0.9 0.9"/>
            <camera name="side" pos="0 -0.75 0.18"
                    xyaxes="1 0 0 0 0 1"/>

            <!-- 地面是当前唯一 contype=1 的碰撞物体。 -->
            <geom name="floor" type="plane" size="2 2 0.05"
                  contype="1" conaffinity="1" material="ground"/>

            <!--
              Torso 是五连杆树的根。简化 capsule 的中心线正好连接前后髋
              锚点，因此 Torso 边与四条腿边等长。
            -->
            <body name="torso" pos="0 0 0">
              <!-- 平面复合根：x slide + z slide + 绕 y 的 hinge。 -->
              <joint name="root_x" type="slide" axis="1 0 0" limited="false"
                     damping="0.01"/>
              <joint name="root_z" type="slide" axis="0 0 1" limited="false"
                     damping="0.01"/>
              <joint name="root_pitch" type="hinge" axis="0 1 0" limited="false"
                     damping="0.01"/>
              <inertial pos="{_f(p.torso_com_x)} 0 {_f(p.torso_com_z)}"
                        mass="{_f(p.torso_mass)}"
                        diaginertia="{' '.join(_f(x) for x in torso_inertia)}"/>
              <geom name="torso_proxy" type="capsule" class="structure_collision"
                    fromto="-{_f(torso_half_length)} 0 0 {_f(torso_half_length)} 0 0"
                    size="{_f(p.upper_proxy_radius)}"
                    rgba="0.12 0.48 0.88 1"/>
{_arc_shell_geoms("torso", (-torso_half_length, 0.0), (torso_half_length, 0.0), (0.0, 1.0), p)}
              <site name="torso_com" type="sphere"
                    pos="{_f(p.torso_com_x)} 0 {_f(p.torso_com_z)}"
                    size="0.006" rgba="0.1 1 0.1 1"/>

              <!--
                前腿链：髋关节正转使膝向外（+x），膝关节正转使足端向内。
              -->
              <body name="front_thigh" pos="{_f(p.hip_half_span)} 0 0">
                <joint name="front_hip" type="hinge" axis="0 -1 0"
                       class="hip_joint"/>
                <inertial pos="0 0 -{_f(p.thigh.com_along_link)}"
                          mass="{_f(p.thigh.mass)}"
                          diaginertia="{' '.join(_f(x) for x in thigh_inertia)}"/>
                <geom name="front_thigh_proxy" type="capsule" class="structure_collision"
                      fromto="0 0 0 0 0 -{_f(p.upper_length)}"
                      size="{_f(p.upper_proxy_radius)}"
                      rgba="0.95 0.45 0.12 1"/>
{_arc_shell_geoms("front_thigh", (0.0, 0.0), (0.0, -p.upper_length), (1.0, 0.0), p, indent=16)}
                <site name="front_hip_site" type="sphere" size="0.006"
                      rgba="1 0.85 0.1 1"/>
                <site name="front_thigh_com" type="sphere"
                      pos="0 0 -{_f(p.thigh.com_along_link)}"
                      size="0.005" rgba="0.1 1 0.1 1"/>

                <body name="front_shank" pos="0 0 -{_f(p.upper_length)}">
                  <joint name="front_knee" type="hinge" axis="0 1 0"
                         class="knee_joint"/>
                  <inertial pos="0 0 -{_f(p.shank.com_along_link)}"
                            mass="{_f(p.shank.mass)}"
                            diaginertia="{' '.join(_f(x) for x in shank_inertia)}"/>
                  <geom name="front_shank_proxy" type="capsule" class="structure_collision"
                        fromto="0 0 0 0 0 -{_f(p.lower_length)}"
                        size="{_f(p.lower_proxy_radius)}"
                        rgba="0.98 0.70 0.18 1"/>
{_arc_shell_geoms("front_shank", (0.0, 0.0), (0.0, -p.lower_length), (1.0, 0.0), p, indent=18)}
                  <geom name="front_foot_proxy" type="sphere" class="structure_collision"
                        pos="0 0 -{_f(p.lower_length)}"
                        size="{_f(p.foot_radius)}"
                        rgba="0.85 0.85 0.88 1"/>
                  <site name="front_knee_site" type="sphere" size="0.006"
                        rgba="1 0.85 0.1 1"/>
                  <site name="front_shank_com" type="sphere"
                        pos="0 0 -{_f(p.shank.com_along_link)}"
                        size="0.005" rgba="0.1 1 0.1 1"/>
                  <site name="front_foot_site"
                        pos="0 0 -{_f(p.lower_length)}" size="0.004"/>
                </body>
              </body>

              <!--
                后腿链通过反转两个 hinge 轴镜像前腿，因此前后腿可以共用
                相同的等效关节角约定。
              -->
              <body name="rear_thigh" pos="-{_f(p.hip_half_span)} 0 0">
                <joint name="rear_hip" type="hinge" axis="0 1 0"
                       class="hip_joint"/>
                <inertial pos="0 0 -{_f(p.thigh.com_along_link)}"
                          mass="{_f(p.thigh.mass)}"
                          diaginertia="{' '.join(_f(x) for x in thigh_inertia)}"/>
                <geom name="rear_thigh_proxy" type="capsule" class="structure_collision"
                      fromto="0 0 0 0 0 -{_f(p.upper_length)}"
                      size="{_f(p.upper_proxy_radius)}"
                      rgba="0.84 0.24 0.18 1"/>
{_arc_shell_geoms("rear_thigh", (0.0, 0.0), (0.0, -p.upper_length), (-1.0, 0.0), p, indent=16)}
                <site name="rear_hip_site" type="sphere" size="0.006"
                      rgba="1 0.85 0.1 1"/>
                <site name="rear_thigh_com" type="sphere"
                      pos="0 0 -{_f(p.thigh.com_along_link)}"
                      size="0.005" rgba="0.1 1 0.1 1"/>

                <body name="rear_shank" pos="0 0 -{_f(p.upper_length)}">
                  <joint name="rear_knee" type="hinge" axis="0 -1 0"
                         class="knee_joint"/>
                  <inertial pos="0 0 -{_f(p.shank.com_along_link)}"
                            mass="{_f(p.shank.mass)}"
                            diaginertia="{' '.join(_f(x) for x in shank_inertia)}"/>
                  <geom name="rear_shank_proxy" type="capsule" class="structure_collision"
                        fromto="0 0 0 0 0 -{_f(p.lower_length)}"
                        size="{_f(p.lower_proxy_radius)}"
                        rgba="0.94 0.55 0.16 1"/>
{_arc_shell_geoms("rear_shank", (0.0, 0.0), (0.0, -p.lower_length), (-1.0, 0.0), p, indent=18)}
                  <geom name="rear_foot_proxy" type="sphere" class="structure_collision"
                        pos="0 0 -{_f(p.lower_length)}"
                        size="{_f(p.foot_radius)}"
                        rgba="0.85 0.85 0.88 1"/>
                  <site name="rear_knee_site" type="sphere" size="0.006"
                        rgba="1 0.85 0.1 1"/>
                  <site name="rear_shank_com" type="sphere"
                        pos="0 0 -{_f(p.shank.com_along_link)}"
                        size="0.005" rgba="0.1 1 0.1 1"/>
                  <site name="rear_foot_site"
                        pos="0 0 -{_f(p.lower_length)}" size="0.004"/>
                </body>
              </body>
            </body>
          </worldbody>

          <contact>
            <!--
              compact 中允许两个有限尺寸足端表面接触。显式 pair 使用更硬的
              内部接触参数，减少软接触导致的数值穿透；并不把足端焊接。
            -->
            <pair name="front_rear_foot_contact"
                  geom1="front_foot_proxy" geom2="rear_foot_proxy"
                  condim="3" friction="{_f(p.nominal_ground_friction)} 0.02 0.01"
                  solref="0.002 1" solimp="0.97 0.995 0.001"/>
          </contact>

          <!-- 四个成对位置伺服；ctrl 表示目标关节角。 -->
          <actuator>
            <general name="front_hip_servo" joint="front_hip"
                     class="paired_servo"
                     ctrlrange="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            <general name="front_knee_servo" joint="front_knee"
                     class="paired_servo"
                     ctrlrange="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
            <general name="rear_hip_servo" joint="rear_hip"
                     class="paired_servo"
                     ctrlrange="{_f(p.hip.shell_compatible_range[0])} {_f(p.hip.shell_compatible_range[1])}"/>
            <general name="rear_knee_servo" joint="rear_knee"
                     class="paired_servo"
                     ctrlrange="{_f(p.knee.shell_compatible_range[0])} {_f(p.knee.shell_compatible_range[1])}"/>
          </actuator>

          <!--
            同一模型中的刚性实验模式。四个约束默认关闭，不影响正常 servo
            仿真；被动刚性基准脚本在运行时通过 data.eq_active 开启它们，
            并同时关闭 actuator。joint2 省略时，polycoef[0] 是相对 qpos0
            的固定目标角。
          -->
          <equality>
            <joint name="lock_front_hip_compact" joint1="front_hip"
                   polycoef="{_f(p.compact_hip_angle)} 0 0 0 0"
                   active="false" solref="0.004 1"
                   solimp="0.999 0.9999 0.001"/>
            <joint name="lock_front_knee_compact" joint1="front_knee"
                   polycoef="{_f(p.compact_knee_angle)} 0 0 0 0"
                   active="false" solref="0.004 1"
                   solimp="0.999 0.9999 0.001"/>
            <joint name="lock_rear_hip_compact" joint1="rear_hip"
                   polycoef="{_f(p.compact_hip_angle)} 0 0 0 0"
                   active="false" solref="0.004 1"
                   solimp="0.999 0.9999 0.001"/>
            <joint name="lock_rear_knee_compact" joint1="rear_knee"
                   polycoef="{_f(p.compact_knee_angle)} 0 0 0 0"
                   active="false" solref="0.004 1"
                   solimp="0.999 0.9999 0.001"/>
          </equality>

          <!-- 用于诊断和未来 RL 观测的最小传感器集合。 -->
          <sensor>
            <framepos name="torso_position" objtype="body" objname="torso"/>
            <framelinvel name="torso_velocity" objtype="body" objname="torso"/>
            <frameangvel name="torso_angular_velocity" objtype="body" objname="torso"/>
            <jointpos name="front_hip_position" joint="front_hip"/>
            <jointpos name="front_knee_position" joint="front_knee"/>
            <jointpos name="rear_hip_position" joint="rear_hip"/>
            <jointpos name="rear_knee_position" joint="rear_knee"/>
          </sensor>

          <keyframe>
            <!-- 展开姿态：两条腿竖直，两个足端球与地面相切。 -->
            <key name="open"
                 qpos="0 {_f(p.open_root_height)} 0 0 0 0 0"
                 ctrl="0 0 0 0"/>
            <!--
              二维行走检查姿态：前足支撑，后足摆动。它只用于验证展开后的
              外壳间隙与足端工作区，不代表固定的行走控制轨迹。
            -->
            <key name="walk"
                 qpos="0 {_f(p.walk_root_height)} 0 {_f(p.walk_front_hip_angle)} {_f(p.walk_front_knee_angle)} {_f(p.walk_rear_hip_angle)} {_f(p.walk_rear_knee_angle)}"
                 ctrl="{_f(p.walk_front_hip_angle)} {_f(p.walk_front_knee_angle)} {_f(p.walk_rear_hip_angle)} {_f(p.walk_rear_knee_angle)}"/>
            <!--
              等边 compact 姿态：
                hip 固定为 18 度；
                knee 由有限尺寸足端的表面接触条件反算。
              前后足端属于两条开链，只允许接触，不焊接、不中心重合。
            -->
            <key name="compact"
                 qpos="0 {_f(p.compact_root_height)} 0 {_f(p.compact_hip_angle)} {_f(p.compact_knee_angle)} {_f(p.compact_hip_angle)} {_f(p.compact_knee_angle)}"
                 ctrl="{_f(p.compact_hip_angle)} {_f(p.compact_knee_angle)} {_f(p.compact_hip_angle)} {_f(p.compact_knee_angle)}"/>
          </keyframe>
        </mujoco>
        """
    )


def write_mjcf(path: Path, parameters: FixedParameters = FIXED_PARAMETERS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_mjcf(parameters), encoding="utf-8")
    return path
