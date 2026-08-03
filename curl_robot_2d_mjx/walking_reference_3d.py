"""Morphology-aware walking reference for the 3-D curl robot."""

from __future__ import annotations

import math

from curl_robot_2d_mjx.config_walking_3d import WalkingReference3DConfig


# Joint order is front-left, front-right, rear-left, rear-right.  Rear joint
# axes are mirrored in MJCF, so their effective outward coordinate is -world-x.
LEG_WORLD_X_SIGNS_3D = (1.0, 1.0, -1.0, -1.0)


def leg_inverse_kinematics(xp, outward_x_m, depth_m, upper_m, lower_m):
    """Solve the curl leg in its effective outward/downward coordinates."""

    radius = xp.sqrt(xp.square(outward_x_m) + xp.square(depth_m))
    minimum_radius = abs(upper_m - lower_m) + 1.0e-6
    maximum_radius = upper_m + lower_m - 1.0e-6
    radius = xp.clip(radius, minimum_radius, maximum_radius)
    bend_cosine = (
        xp.square(radius) - upper_m * upper_m - lower_m * lower_m
    ) / (2.0 * upper_m * lower_m)
    knee = xp.arccos(xp.clip(bend_cosine, -1.0, 1.0))
    direction = xp.arctan2(outward_x_m, depth_m)
    hip_offset = xp.arctan2(
        lower_m * xp.sin(knee), upper_m + lower_m * xp.cos(knee)
    )
    return direction + hip_offset, knee


def leg_forward_kinematics(xp, hip, knee, upper_m, lower_m):
    """Return effective outward/downward foot coordinates."""

    outward = upper_m * xp.sin(hip) + lower_m * xp.sin(hip - knee)
    depth = upper_m * xp.cos(hip) + lower_m * xp.cos(knee - hip)
    return outward, depth


def walking_foot_trajectory(xp, phase, config: WalkingReference3DConfig):
    """Return world-x offset, downward depth, stance flag, and foot lift."""

    phase = xp.mod(phase, 1.0)
    stance = phase < config.duty_factor
    stance_phase = phase / config.duty_factor
    swing_phase = (phase - config.duty_factor) / (1.0 - config.duty_factor)
    swing_phase = xp.clip(swing_phase, 0.0, 1.0)
    blend = swing_phase * swing_phase * (3.0 - 2.0 * swing_phase)
    stance_x = config.step_length_m * (0.5 - stance_phase)
    swing_x = config.step_length_m * (blend - 0.5)
    lift = config.foot_lift_m * xp.sin(math.pi * blend)
    lift = xp.where(stance, 0.0, lift)
    world_x = xp.where(stance, stance_x, swing_x)
    return (
        world_x + config.fore_aft_center_m,
        config.body_height_m - lift,
        stance,
        lift,
    )


def walking_reference_3d(
    xp,
    oscillator_phase_rad,
    config: WalkingReference3DConfig,
):
    """Generate eight joint targets and per-foot schedule information."""

    base_phase = oscillator_phase_rad / (2.0 * math.pi)
    joint_pairs = []
    stance = []
    lift = []
    world_x = []
    for phase_offset, world_x_sign in zip(
        config.phase_offsets, LEG_WORLD_X_SIGNS_3D
    ):
        foot_x, depth, foot_stance, foot_lift = walking_foot_trajectory(
            xp, base_phase + phase_offset, config
        )
        hip, knee = leg_inverse_kinematics(
            xp,
            world_x_sign * foot_x,
            depth,
            config.upper_length_m,
            config.lower_length_m,
        )
        joint_pairs.extend((hip, knee))
        stance.append(foot_stance)
        lift.append(foot_lift)
        world_x.append(foot_x)
    targets = xp.stack(joint_pairs)
    joint_low = xp.asarray(
        (config.hip_range[0], config.knee_range[0]) * 4
    )
    joint_high = xp.asarray(
        (config.hip_range[1], config.knee_range[1]) * 4
    )
    return {
        "joint_targets": xp.clip(targets, joint_low, joint_high),
        "stance": xp.stack(stance),
        "foot_lift_m": xp.stack(lift),
        "foot_world_x_m": xp.stack(world_x),
    }
