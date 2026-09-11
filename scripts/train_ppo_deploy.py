#!/usr/bin/env python3
"""PPO for rollingquad_2 — trained to be DEPLOYABLE on the Pupper Pi stack.

Same robot and physics as train_ppo_walk3d.py, with deploy gait shaping.
What changes is everything the real controller constrains, and nothing else:

  OBSERVATION.  neural_controller.hpp hardcodes kSingleObservationSize = 36 and
  neural_controller.cpp:127 refuses to load a model whose in_shape disagrees.
  The layout below is copied from the C++ line by line, in its order and with
  its scaling (which is: none).  Two things walk3d fed the policy are simply
  not in it -- base linear velocity, which the robot cannot measure because it
  has no state estimator, and joint velocities, which the hardware has but the
  controller does not pass on.

  HISTORY.  Dropping base linear velocity leaves the policy blind to how fast
  it is travelling, so it has to infer that from how gravity and the joint
  angles evolve.  That is what the observation stack is for, and it is why
  Stanford's own shipped policy uses observation_history = 20.  The buffer is
  ordered newest-first and shifted exactly as std::rotate does in the C++.

  ACTIVATION.  elu.  RTNeural's JSON parser (model_loader.h:561-573) knows
  tanh, relu, sigmoid, softmax and elu; Brax defaults to swish, which cannot
  be written into the file at all.

  NO ACTION FILTER.  walk3d low-passes the action before it becomes a servo
  target.  The controller does not: it applies action * scale + default
  directly.  A filter in simulation and none on the robot is a silent
  sim-to-real gap, so it is off here.

Subcommands
    probe    layout and parity checks, no GPU work
    config   write the JSON metadata block for export_rtneural.py
    video    render from an existing policy
    export   train-free: policy .bin -> RTNeural .json
    dr       (composable) domain randomisation
    terrain  (composable) 40% flat / 60% mild 2-D rough terrain
             --terrain-max-height METERS sets the next terrain difficulty
    <none>   train
"""
import os
import sys as _sys
import time
import functools
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Make ``python -m scripts.train_ppo_deploy`` and direct script execution use
# the same sibling-module resolution on both Windows and Linux.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, SCRIPT_DIR)

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mediapy as media
import flax.linen as linen

from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

# Physics, XML patching and the reward weights all come from the walking task
# so that the two cannot drift apart.  Note that train_ppo_curl3d is NOT
# imported: importing it sets w3.SHELL_CONTACT = True as a side effect, and
# a walking policy must not be trained against colliding shells.
import train_ppo_walk3d as w3
from deploy_gait import init_hip_rom, sample_command, update_hip_rom
from deploy_terrain import (
    RoughTerrainConfig, inject_heightfield, limit_collision_hulls, reference_terrain_data,
    surface_height, terrain_data, write_height_preview,
)
from train_ppo_walk3d import (
    DEFAULT_POSE, CTRL_LO, CTRL_HI, ACTION_SCALE, LEGS,
    CMD_VX, CMD_VY, CMD_WZ, CMD_RESAMPLE, ZERO_CMD_PROB,
    TRACK_LIN_W, TRACK_ANG_W, TRACK_SIGMA, ALIVE_W,
    AIR_TIME_W, AIR_TIME_TARGET, LIN_Z_W, ANG_XY_W, ORIENT_W, HEIGHT_W,
    TORQUE_W, TERM_W,
    Z_MIN, UP_MIN, FOOT_R, Ticker, _hms, _INT,
)

w3.SELF_COLLISION = False
w3.RUN_XML = os.path.expanduser(
    "~/robot/rollingquad_2_deploy_no_self_collision.xml")

SAVE = "rollingquad_2_deploy_fine_lift_policy.bin"
VID_DIR = "rollingquad_2_deploy_fine_lift_videos"
CKPT_DIR = "rollingquad_2_deploy_fine_lift_checkpoints"
JSON_OUT = "rollingquad_2_deploy_fine_lift_policy.json"

TERRAIN = False
TERRAIN_CONFIG = RoughTerrainConfig()

# ==================================================== controller contract
# neural_controller.hpp:67-76.  Do not reorder; the C++ writes these indices
# by hand and there is no name attached to any of them.
#   [ 0: 3] base angular velocity (gyro), body frame, UNSCALED
#   [ 3: 6] projected gravity, body frame
#   [ 6: 9] command vx, vy, yaw rate, UNSCALED
#   [ 9:12] desired world z in body frame  (from /cmd_pose)
#   [12:24] joint position - default_joint_pos
#   [24:36] previous action, raw policy output
SINGLE_OBS = 36
HISTORY = 20                 # matches Stanford's shipped policy
OBS_SIZE = SINGLE_OBS * HISTORY
GRAVITY_Z_IDX = 5            # kGravityZIndx
DESIRED_Z_IDX = 9            # desired-world-z occupies [9:12]

# desired_world_z_in_body_frame_ defaults to (0, 0, 1) in the header and only
# changes when something publishes /cmd_pose.  Held constant here; the three
# inputs exist so a policy CAN be taught to lean, not because this one is.
DESIRED_WORLD_Z = jp.array([0.0, 0.0, 1.0])

ACTIVATION = linen.elu
ACTIVATION_NAME = "elu"

# The controller applies  action * scale + default  with no smoothing, so
# training must not smooth either.
ACTION_FILTER = 0.0

# kp from gainprm="5 0 0", kd from biasprm="0 -5 -0.1", both in the XML.
SERVO_KP = 5.0
SERVO_KD = 0.1

# ======================================================= gait shaping
# Deploy-specific weights: stronger motion and contact regularization.
JVEL_W = 0.0004
RATE_W = 0.08
# Contact slip alone misses a swing foot skimming just above the contact
# threshold, so use both contact slip and a smooth near-ground scuff cost.
SLIP_W = 0.50
SCUFF_W = 0.30
SCUFF_HEIGHT = 0.008          # only suppress motion very close to floor (m)
CLEARANCE_W = 0.04
CLEARANCE_TARGET = 0.040      # full height reward at 4 cm foot-bottom clearance
FOOT_LIFT_W = 0.08
FOOT_LIFT_SIGMA = 0.0075      # decay width ABOVE the target height (m)
FOOT_LIFT_SPEED = 0.20        # full reward above this horizontal speed (m/s)

# Per-joint exponential cost around DEFAULT_POSE, active in every command.
# Scales are curvature scales (radians), not hard limits or dead zones.
# Walking allows normal hip/knee excursion; standing restores the pose sooner.
POSE_STAND_W = 0.08
POSE_WALK_W = 0.02
POSE_STAND_SCALE = jp.array([0.10, 0.20, 0.20] * 4)  # abd, hip, knee
POSE_WALK_SCALE = jp.array([0.20, 0.50, 0.50] * 4)
POSE_MAX_JOINT_PENALTY = 0.50  # bound extreme errors before exponentiation

# Straight-line trot symmetry.  The gate below disables these terms for
# lateral motion and turning so they do not remove steering authority.
# Front and rear hip axes oppose each other. Equal raw diagonal actions can
# suppress useful swing, so disable that cost for the first ablation.
DIAG_ACTION_W = 0.0
DIAG_CONTACT_W = 0.10

# Explicit command buckets: 30% forward, 30% backward, 30% mixed, 10% stand.
# The two straight buckets use the same speed-magnitude distribution.
STRAIGHT_CMD_PROB = 0.60
STRAIGHT_CMD_MIN_SPEED = 0.10

# Actual hip peak-to-peak excursion, shared by all four legs and both signs
# of vx. These are conservative starting targets, NOT measured backward-gait
# statistics; calibrate them from stable rollouts at matching speeds.
HIP_ROM_W = 0.08
HIP_ROM_TARGET_RAD = 0.35      # 20 degrees peak-to-peak at reference speed
HIP_ROM_MIN_RAD = 0.10
HIP_ROM_REFERENCE_SPEED = 0.45
HIP_ROM_WARMUP_S = 0.50
HIP_ROM_MIN_CYCLE_S = 0.20
HIP_ROM_MAX_CYCLE_S = 1.20
HIP_ROM_MIN_SWING_S = 0.06
HIP_ROM_MIN_CLEARANCE = 0.008

# ============================================================ PPO config
NUM_TIMESTEPS = 300_000_000  # more than walk3d: no velocity input, so the
                             # policy has to learn to infer it from history
NUM_ENVS = 4096
BATCH_SIZE = 256
NUM_MINIBATCHES = 32
NUM_UPDATES_PER_BATCH = 4
UNROLL_LENGTH = 20
DISCOUNTING = 0.97
LEARNING_RATE = 3e-4
ENTROPY_COST = 1e-2
EPISODE_LENGTH = 1000
NUM_EVALS = 30
VIDEO_SECONDS = 10.0
POLICY_HIDDEN = (512, 256, 128)
VALUE_HIDDEN = (512, 256, 128)
SEED = 0

# ================================================= deploy-only randomisation
# These ranges model fixed robot-to-robot calibration and manufacturing errors.
# Model parameters are sampled independently for each vectorised MJX system;
# episode parameters are sampled at reset and then held for the whole episode.
DEPLOY_DR = False
FRICTION_RANGE = (0.60, 1.40)
TORSO_MASS_SCALE = (0.85, 1.20)
LEG_MASS_SCALE = (0.90, 1.10)
INERTIA_SCALE = (0.85, 1.15)
TORSO_COM_XY_M = 0.010
TORSO_COM_Z_M = 0.005
MOTOR_KP_SCALE = (0.85, 1.15)
MOTOR_KD_SCALE = (0.80, 1.20)
MOTOR_TORQUE_SCALE = (0.85, 1.15)

# No low-pass filter.  Latency is an integer action queue and control-frequency
# jitter is represented by an occasional missed 50 Hz command deadline.  A
# missed deadline holds the last command; physics still advances at 500 Hz.
ACTION_LATENCY_PROBS = jp.array([0.60, 0.30, 0.10])  # 0/20/40 ms at 50 Hz
ACTION_QUEUE_LEN = 3
CONTROL_DEADLINE_MISS_PROB = 0.05
MOTOR_ZERO_BIAS_RAD = 0.020
ENCODER_FIXED_BIAS_RAD = 0.010

# Per-frame observation noise, tiled across the history.
FRAME_SIGMA = jp.concatenate([
    jp.full(3, 0.20),       # gyro
    jp.full(3, 0.05),       # projected gravity
    jp.zeros(3),            # command is known exactly
    jp.zeros(3),            # desired world z is a constant
    jp.full(12, 0.01),      # joint positions
    jp.zeros(12),           # last action is known exactly
])
NOISE_SIGMA = jp.tile(FRAME_SIGMA, HISTORY)


def deploy_domain_randomize(sys, rng):
    """Batched deploy-only model DR with independent motor/body variation."""
    @jax.vmap
    def randomize_one(key):
        (k_fric, k_torso_mass, k_leg_mass, k_inertia, k_com,
         k_kp, k_kd, k_torque) = jax.random.split(key, 8)

        friction_value = jax.random.uniform(
            k_fric, (), minval=FRICTION_RANGE[0], maxval=FRICTION_RANGE[1])
        friction = sys.geom_friction.at[:, 0].set(friction_value)

        torso_mass = jax.random.uniform(
            k_torso_mass, (), minval=TORSO_MASS_SCALE[0],
            maxval=TORSO_MASS_SCALE[1])
        leg_mass = jax.random.uniform(
            k_leg_mass, (sys.nbody,), minval=LEG_MASS_SCALE[0],
            maxval=LEG_MASS_SCALE[1])
        mass_scale = (jp.ones(sys.nbody)
                      .at[1].set(torso_mass)
                      .at[2:].set(leg_mass[2:]))
        body_mass = sys.body_mass * mass_scale

        inertia_uncertainty = jax.random.uniform(
            k_inertia, (sys.nbody,), minval=INERTIA_SCALE[0],
            maxval=INERTIA_SCALE[1])
        body_inertia = (sys.body_inertia * mass_scale[:, None]
                        * inertia_uncertainty[:, None])

        com_unit = jax.random.uniform(k_com, (3,), minval=-1.0, maxval=1.0)
        com_offset = com_unit * jp.array(
            [TORSO_COM_XY_M, TORSO_COM_XY_M, TORSO_COM_Z_M])
        body_ipos = sys.body_ipos.at[1].add(com_offset)

        kp_scale = jax.random.uniform(
            k_kp, (sys.nu,), minval=MOTOR_KP_SCALE[0],
            maxval=MOTOR_KP_SCALE[1])
        kd_scale = jax.random.uniform(
            k_kd, (sys.nu,), minval=MOTOR_KD_SCALE[0],
            maxval=MOTOR_KD_SCALE[1])
        kp = sys.actuator_gainprm[:, 0] * kp_scale
        gain = sys.actuator_gainprm.at[:, 0].set(kp)
        bias = (sys.actuator_biasprm
                .at[:, 1].set(-kp)
                .at[:, 2].set(sys.actuator_biasprm[:, 2] * kd_scale))

        torque_scale = jax.random.uniform(
            k_torque, (sys.nu,), minval=MOTOR_TORQUE_SCALE[0],
            maxval=MOTOR_TORQUE_SCALE[1])
        force = sys.actuator_forcerange * torque_scale[:, None]
        return (friction, body_mass, body_inertia, body_ipos,
                gain, bias, force)

    values = randomize_one(rng)
    names = (
        "geom_friction", "body_mass", "body_inertia", "body_ipos",
        "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
    )
    replacements = dict(zip(names, values))
    in_axes = jax.tree_util.tree_map(lambda _: None, sys).tree_replace(
        {name: 0 for name in names})
    return sys.tree_replace(replacements), in_axes


def enable_deploy_dr():
    """Enable deploy DR without inheriting walk3d's random shove/latency."""
    global DEPLOY_DR, SAVE, VID_DIR, CKPT_DIR, JSON_OUT
    DEPLOY_DR = True
    w3.DOMAIN_RANDOMIZE = True
    w3.OBS_NOISE = 1.0
    w3.PUSH_EVERY = 0.0
    w3.LATENCY_PROB = 0.0
    SAVE = "rollingquad_2_deploy_robust_dr_policy.bin"
    VID_DIR = "rollingquad_2_deploy_robust_dr_videos"
    CKPT_DIR = "rollingquad_2_deploy_robust_dr_checkpoints"
    JSON_OUT = "rollingquad_2_deploy_robust_dr_policy.json"


def enable_deploy_terrain(max_height=None):
    """Run after enable_deploy_dr so terrain outputs cannot overwrite flat runs."""
    global TERRAIN, TERRAIN_CONFIG, SAVE, VID_DIR, CKPT_DIR, JSON_OUT
    global NUM_ENVS, BATCH_SIZE
    TERRAIN = True
    NUM_ENVS, BATCH_SIZE = 1024, 64
    if max_height is not None:
        TERRAIN_CONFIG = replace(TERRAIN_CONFIG, max_height_m=max_height)
    TERRAIN_CONFIG.validate()
    prefix = "rollingquad_2_deploy_terrain" + ("_dr" if DEPLOY_DR else "")
    SAVE = prefix + "_policy.bin"
    VID_DIR = prefix + "_videos"
    CKPT_DIR = prefix + "_checkpoints"
    JSON_OUT = prefix + "_policy.json"


def set_run_name(name):
    """Use independent outputs and generated XML for a named experiment."""
    global SAVE, VID_DIR, CKPT_DIR, JSON_OUT
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                       for c in name):
        raise ValueError("--run-name must contain only letters, digits, '_' or '-'")
    prefix = "rollingquad_2_deploy_" + name
    SAVE = prefix + "_policy.bin"
    VID_DIR = prefix + "_videos"
    CKPT_DIR = prefix + "_checkpoints"
    JSON_OUT = prefix + "_policy.json"
    w3.RUN_XML = str(Path(w3.RUN_XML).with_name(prefix + "_no_self_collision.xml"))


def resolve_training_resume(resume_path=None, fresh=False):
    """Fresh runs never load a policy or overwrite an existing experiment."""
    if fresh:
        if resume_path is not None:
            raise ValueError("--fresh cannot be combined with --resume")
        existing = [p for p in (SAVE, VID_DIR, CKPT_DIR, JSON_OUT) if Path(p).exists()]
        if existing:
            raise FileExistsError(
                "--fresh requires unused outputs; choose a new --run-name. Existing: "
                + ", ".join(existing))
        return None
    return resume_path or (SAVE if os.path.exists(SAVE) else None)


def randomize_deploy_system(sys, rng):
    """Compose optional motor/body DR with per-environment terrain sampling."""
    if DEPLOY_DR:
        randomized, in_axes = deploy_domain_randomize(sys, rng)
    else:
        randomized = sys
        in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    if TERRAIN and sys.nhfield:
        if sys.nhfield != 1:
            raise ValueError("deploy terrain expects exactly one heightfield")

        @jax.vmap
        def sample_field(key):
            key = jax.random.fold_in(key, TERRAIN_CONFIG.seed)
            kn, kh, kf = jax.random.split(key, 3)
            noise = jax.random.uniform(
                kn, (TERRAIN_CONFIG.grid_size, TERRAIN_CONFIG.grid_size))
            height = jax.random.uniform(
                kh, (), minval=TERRAIN_CONFIG.min_height_m,
                maxval=TERRAIN_CONFIG.max_height_m)
            flat = jax.random.uniform(kf) < TERRAIN_CONFIG.flat_probability
            return terrain_data(jp, noise, jp.where(flat, 0.0, height),
                                TERRAIN_CONFIG)

        randomized = randomized.tree_replace({"hfield_data": sample_field(rng)})
        in_axes = in_axes.tree_replace({"hfield_data": 0})
    return randomized, in_axes


# ================================================================== env
def pose_deviation_penalty(joint_error, moving):
    """Sum exponential costs of actual joint errors, without cancellation.

    Equivalent to sum(min(w * expm1((error / scale)**2), cap)).
    Clamp the normalized error before squaring/exponentiation so large finite
    joint errors cannot overflow. The moving flag follows the command gate.
    """
    weight = jp.where(moving, POSE_WALK_W, POSE_STAND_W)
    scale = jp.where(moving, POSE_WALK_SCALE, POSE_STAND_SCALE)
    max_normalized = jp.sqrt(jp.log1p(POSE_MAX_JOINT_PENALTY / weight))
    normalized = jp.minimum(jp.abs(joint_error) / scale, max_normalized)
    per_joint = jp.minimum(weight * jp.expm1(jp.square(normalized)),
                           POSE_MAX_JOINT_PENALTY)
    return jp.sum(per_joint)


class DeployEnv(PipelineEnv):
    """Walking, observed exactly the way the real controller observes."""

    def __init__(self, terrain=None):
        self._terrain = TERRAIN if terrain is None else terrain
        if self._terrain or TERRAIN:
            TERRAIN_CONFIG.validate()
            base_path = Path(w3.patch_xml())
            # The paired flat video uses the same approximate collision hulls.
            xml = limit_collision_hulls(base_path.read_text(encoding="utf-8"),
                                       TERRAIN_CONFIG.collision_hull_vertices)
            if self._terrain:
                xml = inject_heightfield(xml, TERRAIN_CONFIG)
            suffix = "_terrain.xml" if self._terrain else "_terrain_flat.xml"
            terrain_path = base_path.with_name(base_path.stem + suffix)
            terrain_path.write_text(xml, encoding="utf-8")
            mj = mujoco.MjModel.from_xml_path(str(terrain_path))
            hull_max = max((int(mj.mesh_graph[a]) for a in mj.mesh_graphadr
                            if a >= 0), default=0)
            if hull_max > TERRAIN_CONFIG.collision_hull_vertices:
                raise RuntimeError(f"collision hull limit not applied: {hull_max} vertices")
            if self._terrain:
                mj.hfield_data[:] = reference_terrain_data(TERRAIN_CONFIG)
            w3.validate_model_contract(mj)
        else:
            mj = w3.load_mj()
        self._mj = mj
        self._nom_h = w3.NOMINAL_H
        self._init_z = w3.NOMINAL_H + 0.0005
        joint_qpos, joint_dof = w3.joint_state_indices(mj)
        self._joint_qpos = jp.asarray(joint_qpos)
        self._joint_dof = jp.asarray(joint_dof)
        self._stand_qpos = jp.asarray(w3.stand_key_qpos(mj))

        self._foot_site = np.array([
            mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            for leg in LEGS])
        self._shank_body = np.array([
            mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_shank")
            for leg in LEGS])
        if (self._foot_site < 0).any() or (self._shank_body < 0).any():
            raise RuntimeError("foot sites or shank bodies not found")

        sys = mjcf.load_model(mj)
        super().__init__(sys=sys, backend="mjx", n_frames=w3.N_FRAMES)
        self._cmd_steps = max(int(round(CMD_RESAMPLE / float(self.dt))), 1)

    # ------------------------------------------------------------ helpers
    def _ground_height(self, xy):
        if not self._terrain:
            return jp.zeros(xy.shape[:-1])
        # The DR wrapper replaces self.sys per environment. Looking up the
        # native preview model here would silently use the wrong terrain.
        return surface_height(jp, xy, self.sys.hfield_data, TERRAIN_CONFIG)

    def _sample_command(self, rng):
        return sample_command(
            rng, CMD_VX, CMD_VY, CMD_WZ, STRAIGHT_CMD_PROB,
            ZERO_CMD_PROB, STRAIGHT_CMD_MIN_SPEED)

    def _feet(self, ps):
        from brax import base
        pos = ps.site_xpos[self._foot_site]
        offset = base.Transform.create(pos=pos - ps.xpos[self._shank_body])
        vel = offset.vmap().do(ps.xd.take(self._shank_body - 1)).vel
        return pos, vel

    def _frame(self, ps, info):
        """One 36-value observation, in the controller's own order."""
        inv_rot = math.quat_inv(ps.x.rot[0])
        return jp.concatenate([
            math.rotate(ps.xd.ang[0], inv_rot),                 # gyro
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv_rot),   # proj. gravity
            info["command"],                                     # unscaled
            DESIRED_WORLD_Z,
            ps.q[self._joint_qpos] + info["encoder_bias"] - DEFAULT_POSE,
            info["last_act"],
        ])

    def _push(self, hist, frame):
        """Newest first, exactly as std::rotate leaves the C++ buffer."""
        return jp.concatenate([frame, hist[:-SINGLE_OBS]])

    def _noise(self, obs, rng):
        if not w3.OBS_NOISE:
            return obs
        return obs + w3.OBS_NOISE * NOISE_SIGMA * jax.random.normal(
            rng, obs.shape)

    # -------------------------------------------------------------- reset
    def reset(self, rng):
        rng, k_cmd, k_obs, k_latency, k_motor_zero, k_encoder = (
            jax.random.split(rng, 6))

        # A deterministic, contact-consistent reset avoids teaching the policy
        # to compensate for a 5--25 mm drop and independently perturbed legs.
        quat = jp.array([1.0, 0.0, 0.0, 0.0])
        z = self._init_z + self._ground_height(jp.zeros(2))
        joints = DEFAULT_POSE
        qpos = (self._stand_qpos
                .at[:3].set(jp.array([0.0, 0.0, z]))
                .at[3:7].set(quat)
                .at[self._joint_qpos].set(joints))
        ps = self.pipeline_init(qpos, jp.zeros(self.sys.nv))

        # Match neural_controller::on_activate: stationary gravity is -z and
        # the resting desired-world-z command is +z in every stale frame.
        idx = SINGLE_OBS * jp.arange(HISTORY)
        hist = (jp.zeros(OBS_SIZE)
                .at[GRAVITY_Z_IDX + idx].set(-1.0)
                .at[DESIRED_Z_IDX + 2 + idx].set(1.0))

        info = {
            "rng": rng,
            "command": self._sample_command(k_cmd),
            "last_act": jp.zeros(12),
            "action_queue": jp.zeros((ACTION_QUEUE_LEN, 12)),
            "applied_action": jp.zeros(12),
            "latency_steps": jp.where(
                DEPLOY_DR,
                jax.random.choice(
                    k_latency, ACTION_QUEUE_LEN, p=ACTION_LATENCY_PROBS),
                jp.int32(0)),
            "motor_zero_bias": jp.where(
                DEPLOY_DR,
                jax.random.uniform(
                    k_motor_zero, (12,), minval=-MOTOR_ZERO_BIAS_RAD,
                    maxval=MOTOR_ZERO_BIAS_RAD),
                jp.zeros(12)),
            "encoder_bias": jp.where(
                DEPLOY_DR,
                jax.random.uniform(
                    k_encoder, (12,), minval=-ENCODER_FIXED_BIAS_RAD,
                    maxval=ENCODER_FIXED_BIAS_RAD),
                jp.zeros(12)),
            "air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "step": jp.int32(0),
            "hist": hist,
            "terrain_span": (jp.max(self.sys.hfield_data)
                             * TERRAIN_CONFIG.max_height_m
                             if self._terrain else jp.zeros(())),
        }
        info.update(init_hip_rom(
            ps.q[self._joint_qpos].reshape((4, 3))[:, 1], info["command"]))
        info["hist"] = self._push(hist, self._frame(ps, info))
        metrics = {k: jp.zeros(()) for k in
                   ("track_lin", "track_ang", "air", "slip", "scuff",
                    "clearance", "lift", "diag_action", "diag_contact",
                    "hip_rom_penalty", "hip_rom_target", "hip_rom_front",
                    "hip_rom_rear", "hip_rom_valid_fraction", "hip_rom_cycles",
                    "height_error", "height_penalty",
                    "pose_penalty", "pose_stand_penalty", "pose_walk_penalty",
                    "base_clearance", "ground_height", "terrain_span",
                    "terrain_rough", "terrain_boundary",
                    "hip_fl", "hip_fr", "hip_rl", "hip_rr",
                    "vx", "vy", "wz", "height", "cmd_vx", "cmd_wz")}
        return State(ps, self._noise(info["hist"], k_obs), jp.zeros(()),
                     jp.zeros(()), metrics, info)

    # --------------------------------------------------------------- step
    def step(self, state, action):
        info = dict(state.info)
        rng, k_deadline, k_cmd, k_obs = jax.random.split(info["rng"], 4)

        # Identical to neural_controller.cpp:605 -- no filter, no rate limit.
        action = jp.clip(action, -1.0, 1.0)
        action_queue = jp.concatenate(
            [action[None, :], info["action_queue"][:-1]], axis=0)
        delayed_action = action_queue[info["latency_steps"]]
        deadline_missed = (DEPLOY_DR & (jax.random.uniform(k_deadline)
                                        < CONTROL_DEADLINE_MISS_PROB))
        applied_action = jp.where(
            deadline_missed, info["applied_action"], delayed_action)
        ctrl = jp.clip(
            DEFAULT_POSE + applied_action * ACTION_SCALE
            + info["motor_zero_bias"],
            CTRL_LO, CTRL_HI)

        # No random shove in deploy DR.  Robustness comes from model,
        # calibration, sensing, latency and deadline randomisation instead.
        ps = self.pipeline_step(state.pipeline_state, ctrl)

        inv_rot = math.quat_inv(ps.x.rot[0])
        lin_b = math.rotate(ps.xd.vel[0], inv_rot)
        ang_b = math.rotate(ps.xd.ang[0], inv_rot)
        up = math.rotate(jp.array([0.0, 0.0, 1.0]), ps.x.rot[0])
        cmd = info["command"]
        moving = jp.linalg.norm(cmd) > 0.05

        foot_pos, foot_vel = self._feet(ps)
        foot_ground = self._ground_height(foot_pos[:, :2])
        signed_clearance = foot_pos[:, 2] - FOOT_R - foot_ground
        contact = signed_clearance < 1e-3
        contact_filt = contact | info["last_contact"]
        foot_clearance = jp.maximum(signed_clearance, 0.0)
        ground_height = self._ground_height(ps.q[:2])
        base_clearance = ps.q[2] - ground_height
        foot_vxy2 = jp.sum(jp.square(foot_vel[:, :2]), axis=1)
        swing = (~contact_filt).astype(jp.float32)
        first_contact = (info["air_time"] > 0.0) & contact_filt
        r_air = AIR_TIME_W * jp.sum(
            (info["air_time"] - AIR_TIME_TARGET) * first_contact) * moving
        air_time = (info["air_time"] + self.dt) * ~contact_filt

        # Core locomotion rewards follow walk3d; deploy adds the gait-shaping
        # penalties below to reduce foot drag and improve straight-line trot
        # symmetry without constraining turning commands.
        r_lin = TRACK_LIN_W * jp.exp(
            -jp.sum(jp.square(cmd[:2] - lin_b[:2])) / TRACK_SIGMA)
        r_ang = TRACK_ANG_W * jp.exp(
            -jp.square(cmd[2] - ang_b[2]) / TRACK_SIGMA)

        p_orient = ORIENT_W * jp.sum(jp.square(up[:2]))
        p_linz = LIN_Z_W * jp.square(lin_b[2])
        p_angxy = ANG_XY_W * jp.sum(jp.square(ang_b[:2]))
        p_height = HEIGHT_W * jp.square(base_clearance - self._nom_h)
        p_torque = TORQUE_W * jp.sum(jp.square(ps.qfrc_actuator[6:]))
        p_jvel = JVEL_W * jp.sum(jp.square(ps.qd[self._joint_dof]))
        p_rate = RATE_W * jp.sum(jp.square(action - info["last_act"]))
        p_slip = SLIP_W * jp.sum(foot_vxy2 * contact)

        # Penalise fast swing-foot motion close to the floor, including the
        # visually obvious skimming that lies just outside the contact mask.
        near_ground = jp.exp(-foot_clearance / SCUFF_HEIGHT)
        p_scuff = SCUFF_W * jp.sum(
            foot_vxy2 * swing * near_ground) * moving
        clearance_error = jp.maximum(
            CLEARANCE_TARGET - foot_clearance, 0.0) / CLEARANCE_TARGET
        p_clearance = CLEARANCE_W * jp.sum(
            jp.square(clearance_error) * swing) * moving

        # Below 4 cm, height quality is (clearance / target)^2: 2 cm earns
        # 1/4, 3 cm earns 9/16, and only 4 cm reaches full height credit.
        # Above target, decay smoothly to discourage excessive high-stepping.
        # Cap the sum at two feet so jumping cannot earn extra lift reward.
        lift_height_fraction = jp.clip(
            foot_clearance / CLEARANCE_TARGET, 0.0, 1.0)
        excess_height = jp.maximum(foot_clearance - CLEARANCE_TARGET, 0.0)
        lift_quality = jp.square(lift_height_fraction) * jp.exp(
            -jp.square(excess_height / FOOT_LIFT_SIGMA))
        swing_motion = jp.clip(
            jp.sqrt(foot_vxy2) / FOOT_LIFT_SPEED, 0.0, 1.0)
        r_lift = FOOT_LIFT_W * jp.minimum(
            jp.sum(lift_quality * swing_motion * swing), 2.0) * moving

        # LEGS order is FL, FR, RL, RR.  A trot pairs FL<->RR and FR<->RL.
        # Only impose this symmetry for straight commands.
        leg_action = action.reshape((4, 3))
        straight = ((jp.abs(cmd[1]) < 0.05)
                    & (jp.abs(cmd[2]) < 0.15)
                    & moving).astype(jp.float32)
        p_diag_action = DIAG_ACTION_W * straight * (
            jp.sum(jp.square(leg_action[0] - leg_action[3]))
            + jp.sum(jp.square(leg_action[1] - leg_action[2])))
        contact_f = contact.astype(jp.float32)
        p_diag_contact = DIAG_CONTACT_W * straight * (
            jp.square(contact_f[0] - contact_f[3])
            + jp.square(contact_f[1] - contact_f[2]))
        hip = ps.q[self._joint_qpos].reshape((4, 3))[:, 1]
        hip_state, hip_metrics = update_hip_rom(
            info, hip, cmd, contact_filt, foot_clearance, self.dt,
            weight=HIP_ROM_W, target_rad=HIP_ROM_TARGET_RAD,
            target_min_rad=HIP_ROM_MIN_RAD,
            reference_speed=HIP_ROM_REFERENCE_SPEED,
            warmup_s=HIP_ROM_WARMUP_S, min_cycle_s=HIP_ROM_MIN_CYCLE_S,
            max_cycle_s=HIP_ROM_MAX_CYCLE_S,
            min_swing_s=HIP_ROM_MIN_SWING_S,
            min_clearance=HIP_ROM_MIN_CLEARANCE)
        p_pose = pose_deviation_penalty(
            ps.q[self._joint_qpos] - DEFAULT_POSE, moving)

        bad = jp.isnan(ps.q).any() | jp.isnan(ps.qd).any()
        terrain_boundary = (jp.any(jp.abs(ps.q[:2]) > (
            TERRAIN_CONFIG.half_size_m - TERRAIN_CONFIG.edge_margin_m))
                            if self._terrain else jp.array(False))
        done = ((base_clearance < Z_MIN) | (up[2] < UP_MIN)
                | bad | terrain_boundary).astype(jp.float32)

        reward = (ALIVE_W + r_lin + r_ang + r_air + r_lift
                  - p_orient - p_linz - p_angxy - p_height
                  - p_torque - p_jvel - p_rate
                  - p_slip - p_scuff - p_clearance
                  - p_diag_action - p_diag_contact - p_pose
                  - hip_metrics["hip_rom_penalty"]
                  - TERM_W * done)
        reward = jp.clip(reward, -5.0, 10.0)

        step_i = info["step"] + 1
        resample = (step_i % self._cmd_steps) == 0
        info["command"] = jp.where(resample, self._sample_command(k_cmd), cmd)
        # Reset at command boundaries as well as termination. The helper also
        # detects externally supplied command changes (e.g. video/evaluation).
        hip_reset = init_hip_rom(hip, info["command"])
        reset_hip = (done > 0.0) | jp.any(info["command"] != cmd)
        info.update({k: jp.where(reset_hip, hip_reset[k], v)
                     for k, v in hip_state.items()})
        info["rng"] = rng
        info["last_act"] = action
        info["action_queue"] = action_queue
        info["applied_action"] = applied_action
        info["air_time"] = air_time * (1.0 - done)
        info["last_contact"] = contact
        info["step"] = step_i
        info["hist"] = self._push(info["hist"], self._frame(ps, info))

        metrics = dict(state.metrics)
        metrics.update(hip_metrics)
        metrics.update({
            "track_lin": r_lin, "track_ang": r_ang, "air": r_air,
            "slip": p_slip, "scuff": p_scuff,
            "clearance": p_clearance, "lift": r_lift,
            "diag_action": p_diag_action,
            "diag_contact": p_diag_contact,
            "height_error": jp.abs(base_clearance - self._nom_h),
            "height_penalty": p_height,
            "pose_penalty": p_pose,
            "pose_stand_penalty": jp.where(moving, 0.0, p_pose),
            "pose_walk_penalty": jp.where(moving, p_pose, 0.0),
            "base_clearance": base_clearance, "ground_height": ground_height,
            "terrain_span": info["terrain_span"],
            "terrain_rough": (info["terrain_span"] > 1e-6).astype(jp.float32),
            "terrain_boundary": terrain_boundary.astype(jp.float32),
            "hip_fl": hip[0], "hip_fr": hip[1],
            "hip_rl": hip[2], "hip_rr": hip[3],
            "vx": lin_b[0], "vy": lin_b[1], "wz": ang_b[2],
            "height": ps.q[2], "cmd_vx": cmd[0], "cmd_wz": cmd[2],
        })
        return state.replace(pipeline_state=ps,
                             obs=self._noise(info["hist"], k_obs),
                             reward=reward, done=done, metrics=metrics,
                             info=info)


# ============================================================= rendering
def render_follow(env, roll, width=640, height=480):
    """Free camera aimed at the torso.  Does not use the trackcom camera,
    which can end up inside the robot's own geometry and render black."""
    mj = env._mj
    d = mujoco.MjData(mj)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.1, 120.0, -12.0
    frames = []
    r = mujoco.Renderer(mj, height=height, width=width)
    try:
        for ps in roll:
            q = np.asarray(ps.q, dtype=np.float64)
            if not np.isfinite(q).all():
                continue
            d.qpos[:] = q
            d.qvel[:] = np.asarray(ps.qd, dtype=np.float64)
            mujoco.mj_forward(mj, d)
            cam.lookat[:] = d.qpos[:3]
            r.update_scene(d, camera=cam)
            frames.append(r.render())
    finally:
        r.close()
    return frames


CMD_SCRIPT = (
    ("forward", jp.array([0.45, 0.0, 0.0])),
    ("backward", jp.array([-0.45, 0.0, 0.0])),
    ("turn left", jp.array([0.0, 0.0, 1.0])),
    ("turn right", jp.array([0.20, 0.0, -1.0])),
)


def _yaw(q):
    w, x, y, z = (float(v) for v in q[3:7])
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _scripted_rollout(env, act_fn, seconds, step_fn=None, reset_fn=None):
    step_fn = step_fn or jax.jit(env.step)
    reset_fn = reset_fn or jax.jit(env.reset)
    st = reset_fn(jax.random.PRNGKey(0))
    rng = jax.random.PRNGKey(1)
    per_seg = max(int(seconds / len(CMD_SCRIPT) / float(env.dt)), 1)
    roll, report, terminated = [st.pipeline_state], [], False
    for name, cmd in CMD_SCRIPT:
        st = st.replace(info={**st.info, "command": cmd})
        p0 = np.array(st.pipeline_state.q[:3])
        yaw0 = _yaw(st.pipeline_state.q)
        completed_steps = 0
        for _ in range(per_seg):
            rng, k = jax.random.split(rng)
            st = step_fn(st, act_fn(st.obs, k))
            st = st.replace(info={**st.info, "command": cmd})
            roll.append(st.pipeline_state)
            completed_steps += 1
            if bool(st.done):
                terminated = True
                break
        p1 = np.array(st.pipeline_state.q[:3])
        dt_seg = completed_steps * float(env.dt)
        dyaw = (_yaw(st.pipeline_state.q) - yaw0 + np.pi) % (2 * np.pi) - np.pi
        report.append(
            f"    {name:<11} {np.linalg.norm(p1[:2] - p0[:2]) / dt_seg:5.2f} m/s"
            f"   yaw {dyaw / dt_seg:+5.2f} rad/s   elapsed {dt_seg:.2f}s"
            + ("   [TERMINATED]" if terminated else ""))
        if terminated:
            report.append(
                f"    stopped at {(len(roll)-1)*float(env.dt):.2f}s; "
                f"distance from origin={np.linalg.norm(p1[:2]):.3f}m; "
                f"base clearance={float(st.metrics['base_clearance']):.3f}m "
                f"(minimum {Z_MIN:.3f}m); "
                f"terrain boundary={bool(st.metrics['terrain_boundary'])}")
            break
    return roll, report


def _nets(obs_size, act_size):
    return ppo_networks.make_ppo_networks(
        obs_size, act_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN,
        value_hidden_layer_sizes=VALUE_HIDDEN,
        activation=ACTIVATION)


def make_video(policy_path=None, seconds=None, out=None):
    out = Path(out or os.path.join(VID_DIR, "showcase.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    env = DeployEnv()
    print(f"video terrain={env._terrain}; native heightfields={env._mj.nhfield}",
          flush=True)
    if env._terrain:
        samples = np.asarray(env._mj.hfield_data)
        heights = samples * float(env._mj.hfield_size[0, 2])
        print(f"actual terrain heights={heights.min()*1000:.2f} .. "
              f"{heights.max()*1000:.2f} mm; "
              f"flat spawn radius={TERRAIN_CONFIG.spawn_radius_m:.2f} m; "
              f"full roughness after radius="
              f"{TERRAIN_CONFIG.spawn_radius_m + TERRAIN_CONFIG.transition_m:.2f} m",
              flush=True)
        height_path = write_height_preview(
            samples, TERRAIN_CONFIG, out.with_name(out.stem + "_terrain_height.png"))
        print(f"actual terrain height map: {height_path}", flush=True)
    inf = jax.jit(ppo_networks.make_inference_fn(
        _nets(env.observation_size, env.action_size))(
            model.load_params(policy_path or SAVE), deterministic=True))
    roll, report = _scripted_rollout(
        env, lambda o, k: inf(o, k)[0], seconds or VIDEO_SECONDS * 2)
    print("\n".join(report))
    media.write_video(str(out), render_follow(env, roll), fps=1.0 / float(env.dt))
    print(f"video: {out}")


# =============================================================== config
def write_config(path=None):
    """The metadata block export_rtneural.py embeds and the Pi reads back."""
    import json
    path = path or "rollingquad_2_deploy_fine_lift_config.json"
    cfg = {
        "use_imu": True,
        "control_orientation": False,
        "observation_history": HISTORY,
        "kp": SERVO_KP,
        "kd": SERVO_KD,
        "action_scale": [float(x) for x in ACTION_SCALE],
        "default_joint_pos": [float(x) for x in DEFAULT_POSE],
        "joint_lower_limits": [float(x) for x in CTRL_LO],
        "joint_upper_limits": [float(x) for x in CTRL_HI],
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"wrote {path}")
    print(f"  observation_history {HISTORY} -> in_shape must be "
          f"{OBS_SIZE} = {HISTORY} x {SINGLE_OBS}")
    print(f"  joint order (set config.yaml joint_names to match):")
    for i, leg in enumerate(LEGS):
        print(f"    {3*i:2d}..{3*i+2:2d}  {leg}_hip_abduction, "
              f"{leg}_hip, {leg}_knee")
    return path


def do_export(ckpt=None, out=None):
    import subprocess

    cfg = write_config()
    ckpt = ckpt or SAVE
    out = out or JSON_OUT
    exporter = os.path.join(SCRIPT_DIR, "export_rtneural.py")
    subprocess.run([
        _sys.executable, exporter, ckpt, out,
        "--activation", ACTIVATION_NAME,
        "--config", cfg,
        "--obs-history", str(HISTORY),
    ], check=True)


# ================================================================ probe
def probe():
    print("=" * 72)
    env = DeployEnv()
    print(f"env       obs={env.observation_size} (expected {OBS_SIZE} = "
          f"{HISTORY} x {SINGLE_OBS})   action={env.action_size}   "
          f"dt={float(env.dt):.4f} ({1/float(env.dt):.0f} Hz)")
    if env.observation_size != OBS_SIZE:
        raise RuntimeError("observation size mismatch")
    print(f"          activation {ACTIVATION_NAME}  (RTNeural-expressible)")
    print(f"          action filter {ACTION_FILTER} "
          f"(the controller applies none)")
    print(f"          stance height {env._nom_h:.4f} m")

    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    st2 = jax.jit(env.step)(st, jp.zeros(12))
    print(f"\nreset/step ok   reward {float(st2.reward):+.3f}   "
          f"done {float(st2.done):.0f}")
    if set(st.info) != set(st2.info) or set(st.metrics) != set(st2.metrics):
        raise RuntimeError("info/metrics pytree mismatch")
    print("          info/metrics pytree parity ok")

    f = np.asarray(st2.obs[:SINGLE_OBS])
    names = [("gyro", 0, 3), ("proj gravity", 3, 6), ("command", 6, 9),
             ("desired world z", 9, 12), ("joint pos - default", 12, 24),
             ("last action", 24, 36)]
    print("\nnewest frame of the history, against the C++ layout:")
    for n, a, b in names:
        print(f"    [{a:2d}:{b:2d}] {n:<20} "
              f"{np.array2string(f[a:b][:4], precision=3, floatmode='fixed')}")

    h0 = np.asarray(st.obs)
    stale = h0[SINGLE_OBS:]
    print(f"\nstartup buffer: {int((stale == 0).sum())} zeros, "
          f"gravity-z slots = "
          f"{set(np.round(stale[GRAVITY_Z_IDX::SINGLE_OBS], 3).tolist())}, "
          f"desired-z slots = "
          f"{set(np.round(stale[DESIRED_Z_IDX + 2::SINGLE_OBS], 3).tolist())}")
    print("=" * 72)


# ================================================================ train
def main(resume_path=None, fresh=False):
    restore_from = resolve_training_resume(resume_path, fresh)
    os.makedirs(VID_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    env, eval_env = DeployEnv(), DeployEnv()
    flat_video_env = DeployEnv(terrain=False) if TERRAIN else None

    print("=" * 72)
    print("deployment-compatible walking")
    print(f"  obs {OBS_SIZE} = {HISTORY} x {SINGLE_OBS}, controller layout")
    print(f"  activation {ACTIVATION_NAME}, no action filter")
    print(f"  physics {w3.PHYS_TIMESTEP * 1000:.0f} ms x {w3.N_FRAMES}, "
          f"solver {w3.SOLVER_ITER}/{w3.SOLVER_LS_ITER}, "
          f"self-collision {'ON' if w3.SELF_COLLISION else 'off'}, "
          f"walking proxies {'ON' if w3.WALK_COLLISION_PROXIES else 'off'}")
    print(f"  hidden {POLICY_HIDDEN}")
    if TERRAIN:
        import json
        terrain_metadata = Path(CKPT_DIR) / (
            f"terrain_{TERRAIN_CONFIG.max_height_m * 1000:g}mm_config.json")
        terrain_metadata.write_text(
            json.dumps(asdict(TERRAIN_CONFIG), indent=2), encoding="utf-8")
        print(f"  terrain: flat={TERRAIN_CONFIG.flat_probability:.0%}, "
              f"rough={1-TERRAIN_CONFIG.flat_probability:.0%}, "
              f"height={TERRAIN_CONFIG.min_height_m*1000:g}.."
              f"{TERRAIN_CONFIG.max_height_m*1000:g} mm valley-to-peak, "
              f"grid={TERRAIN_CONFIG.grid_size}x{TERRAIN_CONFIG.grid_size}")
        print("           contact/lift/base height relative to local ground; "
              "checkpoint videos: flat + fixed rough field")
        self_pairs, ground_pairs = w3.candidate_pairs(env._mj)
        print(f"           collision hull vertex limit={TERRAIN_CONFIG.collision_hull_vertices}; "
              f"robot-robot pairs={self_pairs}, robot-ground pairs={ground_pairs}")
    print(f"  commands: forward/backward {STRAIGHT_CMD_PROB/2:.0%} each, "
          f"mixed {1-STRAIGHT_CMD_PROB-ZERO_CMD_PROB:.0%}, "
          f"stand {ZERO_CMD_PROB:.0%}; straight |vx| >= "
          f"{STRAIGHT_CMD_MIN_SPEED:.2f} m/s")
    print(f"  gait shaping jvel={JVEL_W} action_rate={RATE_W} "
          f"height={HEIGHT_W} slip={SLIP_W} scuff={SCUFF_W} "
          f"clearance={CLEARANCE_W}@{CLEARANCE_TARGET:.3f}m "
          f"lift={FOOT_LIFT_W}@{CLEARANCE_TARGET:.3f}m "
          f"diag_action={DIAG_ACTION_W} "
          f"diag_contact={DIAG_CONTACT_W}")
    print(f"  hip ROM: weight={HIP_ROM_W}, target="
          f"clip({HIP_ROM_TARGET_RAD:.2f}*|vx|/{HIP_ROM_REFERENCE_SPEED:.2f}, "
          f"{HIP_ROM_MIN_RAD:.2f}, {HIP_ROM_TARGET_RAD:.2f}) rad; "
          f"cycle={HIP_ROM_MIN_CYCLE_S:.2f}..{HIP_ROM_MAX_CYCLE_S:.2f}s, "
          f"warmup={HIP_ROM_WARMUP_S:.2f}s")
    print(f"  pose exponential cost: stand w={POSE_STAND_W}, "
          f"scales={np.asarray(POSE_STAND_SCALE[:3])}; "
          f"walk w={POSE_WALK_W}, scales={np.asarray(POSE_WALK_SCALE[:3])}; "
          f"per-joint cap={POSE_MAX_JOINT_PENALTY}")
    if DEPLOY_DR:
        print("  deploy DR: latency=0/20/40ms@60/30/10%, "
              f"deadline_miss={CONTROL_DEADLINE_MISS_PROB:.0%}, no shove")
        print(f"             motor_zero=±{MOTOR_ZERO_BIAS_RAD:.3f}rad "
              f"encoder_bias=±{ENCODER_FIXED_BIAS_RAD:.3f}rad "
              f"kp={MOTOR_KP_SCALE} kd={MOTOR_KD_SCALE} "
              f"torque={MOTOR_TORQUE_SCALE}")
        print(f"             torso_mass={TORSO_MASS_SCALE} "
              f"leg_mass={LEG_MASS_SCALE} inertia={INERTIA_SCALE} "
              f"torso_com=±{TORSO_COM_XY_M*1000:.0f}/"
              f"{TORSO_COM_Z_M*1000:.0f}mm")
    print(f"  {NUM_TIMESTEPS:,} steps over {NUM_ENVS} envs, {NUM_EVALS} evals; "
          f"batch_size={BATCH_SIZE}, minibatches={NUM_MINIBATCHES}")
    print(f"  writing to {SAVE}, {CKPT_DIR}/, {VID_DIR}/")
    print("=" * 72, flush=True)

    resume = {}
    if restore_from is not None:
        if not os.path.isfile(restore_from):
            raise FileNotFoundError(f"resume checkpoint not found: {restore_from}")
        resume["restore_params"] = model.load_params(restore_from)
        print(f"RESUMING from {restore_from}\n", flush=True)
    else:
        print("STARTING FROM SCRATCH (no checkpoint loaded)\n", flush=True)

    ticker = Ticker(NUM_EVALS)

    def progress(step, metrics):
        took = ticker.stop()
        g = lambda k: metrics.get(f"eval/episode_{k}", float("nan"))
        # Brax accumulates custom metrics over each episode. Divide angle and
        # validity sums by episode length to print radians/fractions, not sums.
        eval_length = max(float(metrics.get("eval/avg_episode_length", 1.0)), 1.0)
        avg = lambda k: g(k) / eval_length
        print(f"[{ticker.done}/{NUM_EVALS}] step {step:>13,} "
              f"({100.0*step/max(NUM_TIMESTEPS,1):4.1f}%)  "
              f"reward {metrics.get('eval/episode_reward', float('nan'))}  "
              f"ep_len {metrics.get('eval/avg_episode_length', float('nan'))}",
              flush=True)
        print(f"    track_lin {g('track_lin')}  track_ang {g('track_ang')}",
              flush=True)
        print(f"    slip {g('slip')}  scuff {g('scuff')}  "
              f"clearance {g('clearance')}  lift {g('lift')}  "
              f"diag_action {g('diag_action')}  "
              f"diag_contact {g('diag_contact')}", flush=True)
        print(f"    height_error {g('height_error')}  "
              f"height_penalty {g('height_penalty')}", flush=True)
        print(f"    pose_penalty_mean {avg('pose_penalty'):.4f}  "
              f"pose_stand_contribution {avg('pose_stand_penalty'):.4f}  "
              f"pose_walk_contribution {avg('pose_walk_penalty'):.4f}", flush=True)
        print(f"    hip_rom_penalty {g('hip_rom_penalty')}  "
              f"hip_rom_front_mean_rad {avg('hip_rom_front')}  "
              f"hip_rom_rear_mean_rad {avg('hip_rom_rear')}  "
              f"hip_rom_target_mean_rad {avg('hip_rom_target')}  "
              f"hip_rom_valid_fraction {avg('hip_rom_valid_fraction')}  "
              f"hip_rom_cycles {g('hip_rom_cycles')}", flush=True)
        print("    hip_mean_rad FL/FR/RL/RR " + " / ".join(
            f"{avg('hip_' + leg):.3f}" for leg in ("fl", "fr", "rl", "rr")),
              flush=True)
        if TERRAIN:
            print(f"    terrain_rough_time_fraction {avg('terrain_rough'):.3f}  "
                  f"terrain_span_mean_m {avg('terrain_span'):.4f}  "
                  f"base_clearance_mean_m {avg('base_clearance'):.4f}  "
                  f"terrain_boundary {g('terrain_boundary')}", flush=True)
        print(f"    took {_hms(took)}  |  elapsed "
              f"{_hms(time.time() - ticker.run_t0)}  |  ETA {ticker.eta()}",
              flush=True)
        ticker.start()

    jit_cache = {}

    def policy_params_fn(step, make_policy, params):
        model.save_params(os.path.join(CKPT_DIR, f"{step:012d}.bin"), params)
        model.save_params(SAVE, params)
        if _INT["n"]:
            raise KeyboardInterrupt
        try:
            if not jit_cache:
                jit_cache["act"] = jax.jit(
                    lambda p, o, k: make_policy(p, deterministic=True)(o, k)[0])
                jit_cache["step"] = jax.jit(eval_env.step)
                jit_cache["reset"] = jax.jit(eval_env.reset)
                if flat_video_env is not None:
                    jit_cache["flat_step"] = jax.jit(flat_video_env.step)
                    jit_cache["flat_reset"] = jax.jit(flat_video_env.reset)
            video_cases = [(eval_env, "terrain" if TERRAIN else "", "")]
            if flat_video_env is not None:
                video_cases.append((flat_video_env, "flat", "flat_"))
            for video_env, label, cache_prefix in video_cases:
                roll, report = _scripted_rollout(
                    video_env, lambda o, k: jit_cache["act"](params, o, k),
                    VIDEO_SECONDS * 2, step_fn=jit_cache[cache_prefix + "step"],
                    reset_fn=jit_cache[cache_prefix + "reset"])
                suffix = "_" + label if label else ""
                v = os.path.join(VID_DIR, f"deploy_{step:012d}{suffix}.mp4")
                media.write_video(v, render_follow(video_env, roll),
                                  fps=1.0 / float(video_env.dt))
                print("\n".join(report), flush=True)
                print(f"    checkpoint + video: {v}", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    video failed ({e}); checkpoint still saved", flush=True)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=NUM_TIMESTEPS, num_evals=NUM_EVALS,
        episode_length=EPISODE_LENGTH, num_envs=NUM_ENVS,
        batch_size=BATCH_SIZE, num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=NUM_UPDATES_PER_BATCH,
        unroll_length=UNROLL_LENGTH, discounting=DISCOUNTING,
        learning_rate=LEARNING_RATE, entropy_cost=ENTROPY_COST,
        reward_scaling=1.0, normalize_observations=True, action_repeat=1,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=POLICY_HIDDEN,
            value_hidden_layer_sizes=VALUE_HIDDEN,
            activation=ACTIVATION),
        randomization_fn=randomize_deploy_system if (DEPLOY_DR or TERRAIN) else None,
        policy_params_fn=policy_params_fn, seed=SEED, **resume,
    )

    ticker.start()
    try:
        _, params, _ = train_fn(environment=env, progress_fn=progress,
                                eval_env=eval_env)
        model.save_params(SAVE, params)
        print(f"\ndone — policy in {SAVE}", flush=True)
        print("export it with:  python -m scripts.train_ppo_deploy export",
              flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped — newest policy in {SAVE} (and {CKPT_DIR}/)",
              flush=True)


if __name__ == "__main__":
    import traceback
    code = 0
    try:
        argv = _sys.argv[1:]
        fresh = "--fresh" in argv
        if fresh:
            argv.remove("--fresh")
        resume_path = None
        if "--resume" in argv:
            i = argv.index("--resume")
            if i + 1 >= len(argv):
                raise ValueError("--resume requires a checkpoint path")
            resume_path = argv[i + 1]
            del argv[i:i + 2]
        if "dr" in argv:
            enable_deploy_dr()
            argv.remove("dr")
        terrain_max_height = None
        if "--terrain-max-height" in argv:
            i = argv.index("--terrain-max-height")
            if i + 1 >= len(argv):
                raise ValueError("--terrain-max-height requires a height in meters")
            terrain_max_height = float(argv[i + 1])
            del argv[i:i + 2]
            if "terrain" not in argv:
                raise ValueError("--terrain-max-height requires terrain mode")
        if "terrain" in argv:
            enable_deploy_terrain(terrain_max_height)
            argv.remove("terrain")
        if "--run-name" in argv:
            i = argv.index("--run-name")
            if i + 1 >= len(argv):
                raise ValueError("--run-name requires an experiment name")
            set_run_name(argv[i + 1])
            del argv[i:i + 2]
        for flag, setting in (("--num-envs", "NUM_ENVS"), ("--batch-size", "BATCH_SIZE")):
            if flag in argv:
                i = argv.index(flag)
                if i + 1 >= len(argv):
                    raise ValueError(f"{flag} requires a positive integer")
                value = int(argv[i + 1])
                if value <= 0:
                    raise ValueError(f"{flag} requires a positive integer")
                globals()[setting] = value
                del argv[i:i + 2]
        if BATCH_SIZE * NUM_MINIBATCHES % NUM_ENVS:
            raise ValueError("batch_size * num_minibatches must be divisible by num_envs")
        cmd = argv[0] if argv else "train"
        if fresh and cmd != "train":
            raise ValueError("--fresh is only valid for training")
        if cmd == "probe":
            probe()
        elif cmd == "config":
            write_config(argv[1] if len(argv) > 1 else None)
        elif cmd == "video":
            make_video(argv[1] if len(argv) > 1 else None)
        elif cmd == "export":
            do_export(argv[1] if len(argv) > 1 else None,
                      argv[2] if len(argv) > 2 else None)
        else:
            main(resume_path=resume_path, fresh=fresh)
    except BaseException:
        traceback.print_exc()
        code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(code)
