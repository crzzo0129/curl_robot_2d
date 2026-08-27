#!/usr/bin/env python3
"""PPO for curl_robot_3d — trained to be DEPLOYABLE on the Pupper Pi stack.

Same robot, same physics and the same reward stack as train_ppo_walk3d.py.
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
    <none>   train
"""
import os
import sys as _sys
import time
import functools

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
from train_ppo_walk3d import (
    DEFAULT_POSE, CTRL_LO, CTRL_HI, ACTION_SCALE, LEGS,
    CMD_VX, CMD_VY, CMD_WZ, CMD_RESAMPLE, ZERO_CMD_PROB,
    TRACK_LIN_W, TRACK_ANG_W, TRACK_SIGMA, ALIVE_W,
    AIR_TIME_W, AIR_TIME_TARGET, LIN_Z_W, ANG_XY_W, ORIENT_W, HEIGHT_W,
    TORQUE_W, JVEL_W, RATE_W, SLIP_W, STAND_W, TERM_W,
    Z_MIN, UP_MIN, FOOT_R, Ticker, _hms, _INT,
)

w3.RUN_XML = os.path.expanduser("~/robot/curl_robot_3d_deploy.xml")

SAVE = "deploy_policy.bin"
VID_DIR = "deploy_videos"
CKPT_DIR = "deploy_checkpoints"
JSON_OUT = "deploy_policy.json"

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
# Contact slip alone misses a swing foot skimming just above the contact
# threshold, so use both contact slip and a smooth near-ground scuff cost.
SLIP_W = 0.25
SCUFF_W = 0.25
SCUFF_HEIGHT = 0.012          # exponential decay length above the floor (m)
CLEARANCE_W = 0.05
CLEARANCE_TARGET = 0.020      # desired swing-foot bottom clearance (m)

# Straight-line trot symmetry.  The gate below disables these terms for
# lateral motion and turning so they do not remove steering authority.
DIAG_ACTION_W = 0.03
DIAG_CONTACT_W = 0.05

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


# ================================================================== env
class DeployEnv(PipelineEnv):
    """Walking, observed exactly the way the real controller observes."""

    def __init__(self):
        mj = w3.load_mj()
        self._mj = mj
        self._nom_h = w3.stand_height(mj)
        self._init_z = w3.NOMINAL_H + 0.0005

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
    def _sample_command(self, rng):
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        cmd = jp.array([
            jax.random.uniform(k1, (), minval=CMD_VX[0], maxval=CMD_VX[1]),
            jax.random.uniform(k2, (), minval=CMD_VY[0], maxval=CMD_VY[1]),
            jax.random.uniform(k3, (), minval=CMD_WZ[0], maxval=CMD_WZ[1]),
        ])
        stand = jax.random.uniform(k4) < ZERO_CMD_PROB
        return jp.where(stand, jp.zeros(3), cmd)

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
            ps.q[7:] - DEFAULT_POSE,
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
        rng, k_cmd, k_obs = jax.random.split(rng, 3)

        # A deterministic, contact-consistent reset avoids teaching the policy
        # to compensate for a 5--25 mm drop and independently perturbed legs.
        quat = jp.array([1.0, 0.0, 0.0, 0.0])
        z = self._init_z
        joints = DEFAULT_POSE
        qpos = jp.concatenate([jp.array([0.0, 0.0, z]), quat, joints])
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
            "air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "step": jp.int32(0),
            "hist": hist,
        }
        info["hist"] = self._push(hist, self._frame(ps, info))
        metrics = {k: jp.zeros(()) for k in
                   ("track_lin", "track_ang", "air", "slip", "scuff",
                    "clearance", "diag_action", "diag_contact",
                    "vx", "vy", "wz", "height", "cmd_vx", "cmd_wz")}
        return State(ps, self._noise(info["hist"], k_obs), jp.zeros(()),
                     jp.zeros(()), metrics, info)

    # --------------------------------------------------------------- step
    def step(self, state, action):
        info = dict(state.info)
        rng, k_push, k_pv, k_cmd, k_obs = jax.random.split(info["rng"], 5)

        # Identical to neural_controller.cpp:605 -- no filter, no rate limit.
        action = jp.clip(action, -1.0, 1.0)
        ctrl = jp.clip(DEFAULT_POSE + action * ACTION_SCALE, CTRL_LO, CTRL_HI)

        ps_in = state.pipeline_state
        if w3.PUSH_EVERY:
            hit = jax.random.uniform(k_push) < (self.dt / w3.PUSH_EVERY)
            dv = jax.random.normal(k_pv, (2,)) * w3.PUSH_MAG
            qvel = ps_in.qvel.at[:2].add(jp.where(hit, dv, jp.zeros(2)))
            ps_in = ps_in.replace(qvel=qvel, qd=qvel)
        ps = self.pipeline_step(ps_in, ctrl)

        inv_rot = math.quat_inv(ps.x.rot[0])
        lin_b = math.rotate(ps.xd.vel[0], inv_rot)
        ang_b = math.rotate(ps.xd.ang[0], inv_rot)
        up = math.rotate(jp.array([0.0, 0.0, 1.0]), ps.x.rot[0])
        cmd = info["command"]
        moving = jp.linalg.norm(cmd) > 0.05

        foot_pos, foot_vel = self._feet(ps)
        contact = (foot_pos[:, 2] - FOOT_R) < 1e-3
        contact_filt = contact | info["last_contact"]
        foot_clearance = jp.maximum(foot_pos[:, 2] - FOOT_R, 0.0)
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
        p_height = HEIGHT_W * jp.square(ps.q[2] - self._nom_h)
        p_torque = TORQUE_W * jp.sum(jp.square(ps.qfrc_actuator[6:]))
        p_jvel = JVEL_W * jp.sum(jp.square(ps.qd[6:]))
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
        p_stand = STAND_W * (1.0 - moving) * jp.sum(
            jp.abs(ps.q[7:] - DEFAULT_POSE))

        bad = jp.isnan(ps.q).any() | jp.isnan(ps.qd).any()
        done = ((ps.q[2] < Z_MIN) | (up[2] < UP_MIN) | bad).astype(jp.float32)

        reward = (ALIVE_W + r_lin + r_ang + r_air
                  - p_orient - p_linz - p_angxy - p_height
                  - p_torque - p_jvel - p_rate
                  - p_slip - p_scuff - p_clearance
                  - p_diag_action - p_diag_contact - p_stand
                  - TERM_W * done)
        reward = jp.clip(reward, -5.0, 10.0)

        step_i = info["step"] + 1
        resample = (step_i % self._cmd_steps) == 0
        info["command"] = jp.where(resample, self._sample_command(k_cmd), cmd)
        info["rng"] = rng
        info["last_act"] = action
        info["air_time"] = air_time * (1.0 - done)
        info["last_contact"] = contact
        info["step"] = step_i
        info["hist"] = self._push(info["hist"], self._frame(ps, info))

        metrics = dict(state.metrics)
        metrics.update({
            "track_lin": r_lin, "track_ang": r_ang, "air": r_air,
            "slip": p_slip, "scuff": p_scuff,
            "clearance": p_clearance,
            "diag_action": p_diag_action,
            "diag_contact": p_diag_contact,
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
    roll, report, fell = [st.pipeline_state], [], False
    for name, cmd in CMD_SCRIPT:
        st = st.replace(info={**st.info, "command": cmd})
        p0 = np.array(st.pipeline_state.q[:3])
        yaw0 = _yaw(st.pipeline_state.q)
        for _ in range(per_seg):
            rng, k = jax.random.split(rng)
            st = step_fn(st, act_fn(st.obs, k))
            st = st.replace(info={**st.info, "command": cmd})
            roll.append(st.pipeline_state)
            if bool(st.done):
                fell = True
                break
        p1 = np.array(st.pipeline_state.q[:3])
        dt_seg = per_seg * float(env.dt)
        dyaw = (_yaw(st.pipeline_state.q) - yaw0 + np.pi) % (2 * np.pi) - np.pi
        report.append(
            f"    {name:<11} {np.linalg.norm(p1[:2] - p0[:2]) / dt_seg:5.2f} m/s"
            f"   yaw {dyaw / dt_seg:+5.2f} rad/s" + ("   [FELL]" if fell else ""))
        if fell:
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
    os.makedirs(VID_DIR, exist_ok=True)
    env = DeployEnv()
    inf = jax.jit(ppo_networks.make_inference_fn(
        _nets(env.observation_size, env.action_size))(
            model.load_params(policy_path or SAVE), deterministic=True))
    roll, report = _scripted_rollout(
        env, lambda o, k: inf(o, k)[0], seconds or VIDEO_SECONDS * 2)
    print("\n".join(report))
    out = out or os.path.join(VID_DIR, "showcase.mp4")
    media.write_video(out, render_follow(env, roll), fps=1.0 / float(env.dt))
    print(f"video: {out}")


# =============================================================== config
def write_config(path=None):
    """The metadata block export_rtneural.py embeds and the Pi reads back."""
    import json
    path = path or "deploy_config.json"
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
    cfg = write_config()
    ckpt = ckpt or SAVE
    out = out or JSON_OUT
    os.system(f"{_sys.executable} export_rtneural.py {ckpt} {out} "
              f"--activation {ACTIVATION_NAME} --config {cfg} "
              f"--obs-history {HISTORY}")


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
def main():
    os.makedirs(VID_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    env, eval_env = DeployEnv(), DeployEnv()

    print("=" * 72)
    print("deployment-compatible walking")
    print(f"  obs {OBS_SIZE} = {HISTORY} x {SINGLE_OBS}, controller layout")
    print(f"  activation {ACTIVATION_NAME}, no action filter")
    print(f"  hidden {POLICY_HIDDEN}")
    print(f"  gait shaping slip={SLIP_W} scuff={SCUFF_W} "
          f"clearance={CLEARANCE_W} diag_action={DIAG_ACTION_W} "
          f"diag_contact={DIAG_CONTACT_W}")
    print(f"  {NUM_TIMESTEPS:,} steps over {NUM_ENVS} envs, {NUM_EVALS} evals")
    print(f"  writing to {SAVE}, {CKPT_DIR}/, {VID_DIR}/")
    print("=" * 72, flush=True)

    resume = {}
    if os.path.exists(SAVE):
        resume["restore_params"] = model.load_params(SAVE)
        print(f"RESUMING from {SAVE}\n", flush=True)

    ticker = Ticker(NUM_EVALS)

    def progress(step, metrics):
        took = ticker.stop()
        g = lambda k: metrics.get(f"eval/episode_{k}", float("nan"))
        print(f"[{ticker.done}/{NUM_EVALS}] step {step:>13,} "
              f"({100.0*step/max(NUM_TIMESTEPS,1):4.1f}%)  "
              f"reward {metrics.get('eval/episode_reward', float('nan'))}  "
              f"ep_len {metrics.get('eval/avg_episode_length', float('nan'))}",
              flush=True)
        print(f"    track_lin {g('track_lin')}  track_ang {g('track_ang')}",
              flush=True)
        print(f"    slip {g('slip')}  scuff {g('scuff')}  "
              f"clearance {g('clearance')}  "
              f"diag_action {g('diag_action')}  "
              f"diag_contact {g('diag_contact')}", flush=True)
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
            roll, report = _scripted_rollout(
                eval_env, lambda o, k: jit_cache["act"](params, o, k),
                VIDEO_SECONDS * 2, step_fn=jit_cache["step"],
                reset_fn=jit_cache["reset"])
            v = os.path.join(VID_DIR, f"deploy_{step:012d}.mp4")
            media.write_video(v, render_follow(eval_env, roll),
                              fps=1.0 / float(eval_env.dt))
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
        randomization_fn=w3.domain_randomize if w3.DOMAIN_RANDOMIZE else None,
        policy_params_fn=policy_params_fn, seed=SEED, **resume,
    )

    ticker.start()
    try:
        _, params, _ = train_fn(environment=env, progress_fn=progress,
                                eval_env=eval_env)
        model.save_params(SAVE, params)
        print(f"\ndone — policy in {SAVE}", flush=True)
        print(f"export it with:  python3 {os.path.basename(__file__)} export",
              flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped — newest policy in {SAVE} (and {CKPT_DIR}/)",
              flush=True)


if __name__ == "__main__":
    import traceback
    code = 0
    try:
        argv = _sys.argv[1:]
        if "dr" in argv:
            w3.enable_dr()          # sets w3.OBS_NOISE / PUSH_* / DOMAIN_*
            SAVE = "deploy_policy_dr.bin"
            VID_DIR = "deploy_videos_dr"
            CKPT_DIR = "deploy_checkpoints_dr"
            JSON_OUT = "deploy_policy_dr.json"
            argv.remove("dr")
        cmd = argv[0] if argv else "train"
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
            main()
    except BaseException:
        traceback.print_exc()
        code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(code)
