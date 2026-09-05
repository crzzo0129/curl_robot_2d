"""Low-speed compact targets and contact-point slip, without a folding path."""

from dataclasses import dataclass
import numpy as np

from curl_robot_2d_mjx.autonomous_startup_3d import AutonomousStartupConfig

COMPACT_STARTUP_CONTRACT = "stand_to_low_speed_compact_v3_anti_ballistic"
COMPACT_REACH_CONTRACT = "walking_stand_to_compact_only_v1"


@dataclass(frozen=True)
class CompactStartupConfig(AutonomousStartupConfig):
    discounting: float = .999
    continuation_s: float = 10.0
    minimum_turns: float = 5.0
    # Provisional training gates. Actual teacher continuation is the acceptance.
    joint_position_rad: float = .02
    joint_velocity_rad_s: float = .05
    root_z_m: float = .01
    root_linear_velocity_m_s: float = .02
    root_angular_velocity_rad_s: float = .10
    orientation_rad: float = .05
    rolling_phase_rad: float = .05
    first_command_jump_rad: float = .05
    foot_slip_weight: float = .05
    foot_slip_sigma_m_s: float = .10
    settling_velocity_weight: float = .02
    settling_joint_sigma_rad_s: float = .10
    settling_linear_sigma_m_s: float = .05
    settling_angular_sigma_rad_s: float = .20
    settling_pose_sigma_rad: float = .20
    potential_root_height_sigma_m: float = .03
    potential_orientation_sigma_rad: float = .20
    max_joint_target_step_rad: float = .05
    upward_velocity_weight: float = .05
    upward_velocity_sigma_m_s: float = .15
    excess_height_weight: float = .05
    excess_height_margin_m: float = .02
    excess_height_sigma_m: float = .02
    angular_velocity_weight: float = .02
    angular_velocity_sigma_rad_s: float = .50
    axis_tilt_weight: float = .05
    axis_tilt_sigma_rad: float = .15


@dataclass(frozen=True)
class CompactReachConfig(CompactStartupConfig):
    """Stage one: terminate at a confirmed compact window, without a teacher."""

    confirmation_steps: int = 5

    def episode_steps(self, dt):
        return round(self.startup_budget_s / dt)


def compact_target(model):
    key = model.key("compact")
    if model.nq != 19 or model.nv != 18 or model.nu != 12:
        raise ValueError("compact startup requires the corrected 12-joint model")
    return {"qpos": np.asarray(key.qpos, dtype=np.float32)[None].copy(),
            "qvel": np.zeros((1, model.nv), dtype=np.float32),
            "ctrl": np.asarray(key.ctrl, dtype=np.float32)[None].copy(),
            "time": np.zeros(1, dtype=np.float32),
            "rolling_phase": np.zeros(1, dtype=np.float32),
            "oscillator_phase": np.zeros(1, dtype=np.float32)}


def compact_potential(xp, qpos, qvel, target, cfg):
    # Pose attraction must remain useful while moving: small velocity gates
    # must not flatten the reward far from compact. Slow down near the target.
    target_qpos = target["qpos"][0]
    error = (qpos[7:] - target_qpos[7:]) / cfg.settling_pose_sigma_rad
    joint_cost = xp.mean(xp.square(error))
    height_cost = xp.square((qpos[2] - target_qpos[2])
                            / cfg.potential_root_height_sigma_m)
    quaternion = qpos[3:7] / xp.sqrt(xp.maximum(xp.sum(xp.square(qpos[3:7])), 1e-12))
    target_quaternion = target_qpos[3:7] / xp.sqrt(
        xp.maximum(xp.sum(xp.square(target_qpos[3:7])), 1e-12))
    dot_squared = xp.clip(xp.square(xp.sum(quaternion * target_quaternion)), 0., 1.)
    # 4*(1-dot^2) is a smooth sign-invariant small-angle approximation to
    # squared quaternion distance, with no acos gradient singularity at zero.
    orientation_cost = 4. * (1. - dot_squared) / cfg.potential_orientation_sigma_rad**2
    pose = xp.exp(-.5 * (joint_cost + height_cost + orientation_cost) / 3.)
    speed_cost = (xp.mean(xp.square(qvel[6:] / cfg.settling_joint_sigma_rad_s))
                  + xp.mean(xp.square(qvel[:3] / cfg.settling_linear_sigma_m_s))
                  + xp.mean(xp.square(qvel[3:6] / cfg.settling_angular_sigma_rad_s))) / 3
    return pose, pose * xp.log1p(speed_cost)


def limit_startup_action(xp, requested, previous, action_scales, max_joint_step_rad):
    """Slew-limit physical joint targets while keeping all 8 actions independent."""
    normalized_step = max_joint_step_rad / xp.maximum(xp.abs(action_scales), 1e-8)
    delta = xp.clip(requested - previous, -normalized_step, normalized_step)
    return xp.clip(previous + delta, -1., 1.)


def startup_stability_costs(xp, qpos, qvel, axis_tilt, *, stand_z, compact_z, cfg):
    """Dense anti-ballistic costs; no foot-count or left/right symmetry term."""
    upward = xp.square(xp.maximum(qvel[2], 0.) / cfg.upward_velocity_sigma_m_s)
    height_limit = xp.maximum(stand_z, compact_z) + cfg.excess_height_margin_m
    excess_height = xp.square(xp.maximum(qpos[2] - height_limit, 0.)
                              / cfg.excess_height_sigma_m)
    angular = xp.mean(xp.square(qvel[3:6] / cfg.angular_velocity_sigma_rad_s))
    tilt = xp.square(axis_tilt / cfg.axis_tilt_sigma_rad)
    penalties = (cfg.upward_velocity_weight * upward,
                 cfg.excess_height_weight * excess_height,
                 cfg.angular_velocity_weight * angular,
                 cfg.axis_tilt_weight * tilt)
    return penalties, sum(penalties)


def contact_slip(xp, geom1, geom2, distance, position, normal, *, geom_bodyid,
                 body_rootid, subtree_com, cvel, foot_geom_ids, floor_geom_id):
    """World contact-point tangential velocity relative to the floor.

    cvel is MuJoCo's (angular, linear) spatial velocity centered on the root
    subtree COM, not the individual body's COM. Match support.jac_dot's
    transfer convention. Per-foot contact means avoid mesh vertex-count bias.
    All detected foot-mesh/floor contacts count; airborne/self contacts do not.
    Returns mean foot squared slip, summed foot RMS speed, max point speed.
    """
    g1 = xp.clip(geom1, 0, geom_bodyid.shape[0] - 1)
    g2 = xp.clip(geom2, 0, geom_bodyid.shape[0] - 1)
    def velocity(geom):
        body = geom_bodyid[geom]
        offset = position - subtree_com[body_rootid[body]]
        return cvel[body, 3:] + xp.cross(cvel[body, :3], offset)
    relative = velocity(g1) - velocity(g2)
    tangent = relative - xp.sum(relative * normal, axis=-1, keepdims=True) * normal
    squared = xp.sum(xp.square(tangent), axis=-1)
    valid = (geom1 >= 0) & (geom2 >= 0) & (distance <= 0)
    mask = valid[:, None] & (
        ((geom1[:, None] == foot_geom_ids[None, :]) & (geom2[:, None] == floor_geom_id))
        | ((geom2[:, None] == foot_geom_ids[None, :]) & (geom1[:, None] == floor_geom_id)))
    means = xp.sum(xp.where(mask, squared[:, None], 0.), axis=0) / xp.maximum(xp.sum(mask, axis=0), 1)
    # Concatenating zero also handles zero contact capacity in tiny test models.
    peak = xp.sqrt(xp.max(xp.concatenate((xp.where(xp.any(mask, axis=1), squared, 0.), xp.zeros(1)))))
    return xp.mean(means), xp.sum(xp.sqrt(means)), peak


def data_contact_slip(xp, model, data, foot_geom_ids, floor_geom_id):
    contact = data.contact
    if hasattr(contact, "geom1"):
        geom1, geom2 = contact.geom1, contact.geom2
    else:
        geom1, geom2 = contact.geom[:, 0], contact.geom[:, 1]
    return contact_slip(xp, geom1, geom2, contact.dist, contact.pos,
        contact.frame.reshape((-1, 3, 3))[:, 0, :],
        geom_bodyid=xp.asarray(model.geom_bodyid), body_rootid=xp.asarray(model.body_rootid),
        subtree_com=data.subtree_com, cvel=data.cvel, foot_geom_ids=foot_geom_ids,
        floor_geom_id=floor_geom_id)
