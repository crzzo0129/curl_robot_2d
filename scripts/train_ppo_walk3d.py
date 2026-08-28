#!/usr/bin/env python3
"""PPO for rollingquad_2 — omnidirectional walking (forward, backward, turn).

Structure follows the Stanford Pupper / Brax "joystick" locomotion task:

  * the policy is conditioned on a velocity COMMAND (vx, vy, yaw-rate) that is
    resampled during the episode, so one network walks in every direction
    instead of memorising a single forward gait;
  * actions are position-servo residuals on a fixed stance pose, not torques;
  * the reward stack is legged-gym's: exponential velocity tracking plus a set
    of regularisers, with feet-air-time doing the work of producing a gait.

Subcommands
    probe   geometry / contact / stance sanity check, no GPU work
    tune    sweep solver settings and print the resulting stance height
    video   render from an existing policy without training
    dr      (composable) turn on domain randomisation, e.g.  `dr` or `dr video`
    <none>  train
"""
import os
import sys as _sys
import signal
import threading
import time
import functools
import re
from math import sin

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mediapy as media

from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

# ============================================================ file layout
# Resolve from this file so the same checkout works on Windows and Linux,
# regardless of the directory from which ``python -m`` is launched.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_XML = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "assets",
    "rollingquad_description_2", "mjcf", "rollingquad.xml"))
RUN_XML = os.path.expanduser("~/robot/rollingquad_2_walk3d.xml")
SAVE = "rollingquad_2_walk3d_policy.bin"
VID_DIR = "rollingquad_2_walk3d_videos"
CKPT_DIR = "rollingquad_2_walk3d_checkpoints"

# ============================================================== geometry
# Measurements used only by diagnostics and reward normalization.  Reset
# states and observations are resolved from named MJCF joints/keyframes below;
# they never depend on the exported body-tree order.
L_THIGH = 0.0844547808
L_SHANK = 0.0880136895
FOOT_R = 0.0195             # foot sphere radius
HIP_X = 0.0752              # hip fore/aft offset from torso centre
HIP_Y = 0.065                # exported abduction mount lateral offset
SHELL_R = 0.1275            # outer rolling-shell radius (unused when walking)

# Canonical policy/controller order.  The rollingquad_2 body tree uses a
# different qpos order, so all state access is remapped by these names.
#   [FL_abd FL_hip FL_knee  FR_abd FR_hip FR_knee
#    RL_abd RL_hip RL_knee  RR_abd RR_hip RR_knee]
LEGS = ("front_left", "front_right", "rear_left", "rear_right")
JOINT_NAMES = tuple(
    f"{leg}_{joint}"
    for leg in LEGS
    for joint in ("hip_abduction", "hip", "knee")
)

# The values match the rollingquad_2 `stand` keyframe in canonical policy
# order.  Limits are the common safe intersection of the four URDF legs.
DEFAULT_POSE = jp.array([0.0, 0.6, 1.0] * 4)
CTRL_LO = jp.array([-0.5236, -1.745329252, 0.1745329252] * 4)
CTRL_HI = jp.array([3.66519, 1.902408885, 2.094395102] * 4)

# Per-joint action range, in radians, added on top of DEFAULT_POSE.  Abduction
# deliberately uses only +-0.17 rad of its wider URDF travel; hip and knee get
# a useful swing without letting the policy fling the leg to a limit.
ACTION_SCALE = jp.array([0.17, 0.50, 0.50] * 4)


# Root heights are mesh-ground clearances computed for the corrected CAD
# model, rather than planar-link approximations that ignore its inclined axes.
NOMINAL_H = 0.1712243631                  # rollingquad_2 stand keyframe
FULL_H = 0.1934653619                     # rollingquad_2 open keyframe

# =============================================================== physics
# Stanford's shipped Pupper MJX config, which is the one config in this family
# known to train: a coarse solver is fine because the contacts are soft.
PHYS_TIMESTEP = 0.004
SOLVER_ITER = 4
SOLVER_LS_ITER = 8
IMPRATIO = 10.0
CONE = "pyramidal"
EULERDAMP = False           # disable, as Stanford's model does
SELF_COLLISION = False      # corrected CAD meshes collide with ground only
MOTOR_SELF_COLLIDE = False  # retained for compatibility with the older model
SHELL_CONTACT = False       # the rolling shells are decorative when walking
N_FRAMES = 5                # 0.004 * 5 = 50 Hz control

# ============================================================== commands
# "Both sides" = the fore/aft command is signed, so the same policy walks
# forwards and backwards.  Lateral travel is deliberately small: abduction has
# only +-0.17 rad (+-9.7 deg) of travel, so the robot cannot really crab.
CMD_VX = (-0.6, 0.6)        # m/s
CMD_VY = (-0.15, 0.15)      # m/s
CMD_WZ = (-1.2, 1.2)        # rad/s
CMD_RESAMPLE = 5.0          # s; a new command mid-episode teaches transitions
ZERO_CMD_PROB = 0.10        # fraction of commands that are "stand still"

# =============================================================== rewards
TRACK_LIN_W = 1.5
TRACK_ANG_W = 0.8
TRACK_SIGMA = 0.25
ALIVE_W = 0.5
AIR_TIME_W = 0.20           # the term that actually produces stepping
AIR_TIME_TARGET = 0.15      # s of swing per step before it starts paying

LIN_Z_W = 2.0               # bouncing
ANG_XY_W = 0.05             # roll/pitch rate
ORIENT_W = 5.0              # staying upright
HEIGHT_W = 10.0             # not in barkour; this robot will otherwise crouch
TORQUE_W = 0.0002
JVEL_W = 0.0002            # not in barkour; cheap insurance against shivering
RATE_W = 0.01               # action rate
SLIP_W = 0.10
STAND_W = 0.5               # penalise drifting from the pose on a zero command
TERM_W = 1.0

Z_MIN = 0.10                # torso floor; below this the episode ends
UP_MIN = 0.0                # cos(tilt); 0 == tipped 90 degrees

ACTION_FILTER = 0.25        # low-pass on the action; 0 disables

# ============================================================ PPO config
NUM_TIMESTEPS = 200_000_000
NUM_ENVS = 4096
BATCH_SIZE = 256
NUM_MINIBATCHES = 32
NUM_UPDATES_PER_BATCH = 4
UNROLL_LENGTH = 20
DISCOUNTING = 0.97
LEARNING_RATE = 3e-4
ENTROPY_COST = 1e-2
EPISODE_LENGTH = 1000       # 20 s
NUM_EVALS = 25
VIDEO_SECONDS = 10.0
POLICY_HIDDEN = (128, 128, 128, 128)
VALUE_HIDDEN = (256, 256, 256, 256, 256)
SEED = 0

# ================================================== domain randomisation
DOMAIN_RANDOMIZE = False
OBS_NOISE = 0.0
PUSH_EVERY = 0.0            # s between shoves; 0 disables
PUSH_MAG = 0.4              # m/s of instantaneous base velocity
LATENCY_PROB = 0.0

# One sigma per observation element, scaled by OBS_NOISE.
NOISE_SIGMA = jp.concatenate([
    jp.zeros(3),                 # command is known exactly
    jp.full(3, 0.05),            # projected gravity
    jp.full(3, 0.20),            # gyro
    jp.full(3, 0.10),            # base linear velocity
    jp.full(12, 0.01),           # joint positions
    jp.full(12, 0.50),           # joint velocities
    jp.zeros(12),                # last action is known exactly
])
OBS_SIZE = 48


# ========================================================== model set-up
def patch_xml():
    """Rewrite the source XML into the one MJX actually runs.

    Four changes, all of them things MJX or Brax care about and the authoring
    model does not:
      1. preserve the source directory for relative CAD mesh paths after the
         patched XML is written under ~/robot;
      2. solver settings -> Stanford's proven MJX numbers (the shipped file is
         a 1 ms elliptic-cone CPU config, ~40x more solver work per control
         step than MJX needs);
      3. collision masks -> optionally drop self-collision or the shells;
      4. a com-tracking camera, so the training videos follow the robot.
    The <sensor> block is stripped because nothing here reads it.
    """
    with open(SRC_XML) as f:
        xml = f.read()

    mesh_dir = os.path.dirname(os.path.abspath(SRC_XML)).replace("\\", "/")

    def preserve_mesh_dir(match):
        tag = match.group(0)
        if "meshdir=" in tag:
            return tag
        return tag[:-2] + f' meshdir="{mesh_dir}"/>'

    xml, ncompiler = re.subn(
        r"<compiler\b[^>]*/>", preserve_mesh_dir, xml, count=1
    )
    if ncompiler != 1:
        raise RuntimeError("could not find the <compiler .../> element to patch")

    opt = (f'<option timestep="{PHYS_TIMESTEP}" gravity="0 0 -9.81" '
           f'integrator="implicitfast" cone="{CONE}" impratio="{IMPRATIO}" '
           f'iterations="{SOLVER_ITER}" ls_iterations="{SOLVER_LS_ITER}">'
           + ('<flag eulerdamp="disable"/>' if not EULERDAMP else '')
           + '</option>')
    xml, n = re.subn(r"<option\b.*?/>", opt, xml, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("could not find the <option .../> element to patch")

    # Collision masks.  Collide iff (contype_a & conaffinity_b) or the reverse.
    #   floor is contype=1 conaffinity=1
    #   ground-only  -> contype=0 conaffinity=1   (hits the floor, not itself)
    #   structural   -> contype=2 conaffinity=7   (hits the floor and itself)
    struct = ('contype="2" conaffinity="7"' if SELF_COLLISION
              else 'contype="0" conaffinity="1"')
    xml = xml.replace('<geom contype="2" conaffinity="7"',
                      f'<geom {struct}')
    if SHELL_CONTACT:
        xml = xml.replace(
            '<geom type="capsule" size="0.003"\n                    '
            'contype="0" conaffinity="0"',
            '<geom type="capsule" size="0.003"\n                    '
            'contype="0" conaffinity="1"')

    # The older generated model had 32 named motor collision capsules.  The
    # corrected rollingquad_2 CAD model has none, so zero replacements are a
    # valid model contract rather than an error.
    if not MOTOR_SELF_COLLIDE:
        xml, nmot = re.subn(
            r'(<geom name="[a-z_]+_motor_collision_\d+" type="capsule")',
            r'\1 contype="0" conaffinity="1"', xml)
        if nmot not in (0, 32):
            raise RuntimeError(
                f"expected either 0 or 32 motor collision geoms, patched {nmot}")

    xml = re.sub(r"<sensor\b.*?</sensor>", "", xml, flags=re.S)

    cam = ('<camera name="track" mode="trackcom" pos="0 -1.1 0.45" '
           'xyaxes="1 0 0 0 0.38 0.92"/>')
    xml = xml.replace("<worldbody>", "<worldbody>\n    " + cam, 1)

    os.makedirs(os.path.dirname(RUN_XML), exist_ok=True)
    with open(RUN_XML, "w") as f:
        f.write(xml)
    return RUN_XML


def joint_state_indices(mj):
    """Return canonical policy-order qpos and qvel indices by joint name."""

    qpos = []
    dof = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"joint not found in the XML: {name}")
        qpos.append(int(mj.jnt_qposadr[joint_id]))
        dof.append(int(mj.jnt_dofadr[joint_id]))
    return np.asarray(qpos, dtype=np.int32), np.asarray(dof, dtype=np.int32)


def stand_key_qpos(mj):
    key_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise RuntimeError("stand keyframe not found in the XML")
    return np.asarray(mj.key_qpos[key_id]).copy()


def validate_model_contract(mj):
    """Fail before training if policy and corrected-MJCF channels diverge."""

    if (mj.nq, mj.nv, mj.nu) != (19, 18, 12):
        raise RuntimeError(
            f"expected floating-base 12-DoF model, got "
            f"nq={mj.nq} nv={mj.nv} nu={mj.nu}"
        )
    qpos_indices, _ = joint_state_indices(mj)
    actuator_names = tuple(
        mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(mj.nu)
    )
    expected_actuators = tuple(f"{name}_servo" for name in JOINT_NAMES)
    if actuator_names != expected_actuators:
        raise RuntimeError(
            "actuators are not in canonical FL/FR/RL/RR policy order: "
            f"{actuator_names}"
        )
    for index, joint_name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if int(mj.actuator_trnid[index, 0]) != joint_id:
            raise RuntimeError(
                f"{expected_actuators[index]} is bound to the wrong joint"
            )
    ctrl_lo = np.asarray(CTRL_LO)
    ctrl_hi = np.asarray(CTRL_HI)
    if np.any(ctrl_lo < mj.actuator_ctrlrange[:, 0] - 1e-8) or np.any(
        ctrl_hi > mj.actuator_ctrlrange[:, 1] + 1e-8
    ):
        raise RuntimeError("policy control limits exceed the MJCF actuator limits")
    stand = stand_key_qpos(mj)
    if not np.allclose(stand[qpos_indices], np.asarray(DEFAULT_POSE), atol=1e-6):
        raise RuntimeError("stand keyframe does not match DEFAULT_POSE")
    for leg in LEGS:
        for obj_type, name in (
            (mujoco.mjtObj.mjOBJ_BODY, f"{leg}_shank"),
            (mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site"),
        ):
            if mujoco.mj_name2id(mj, obj_type, name) < 0:
                raise RuntimeError(f"required MJCF object not found: {name}")


def load_mj():
    mj = mujoco.MjModel.from_xml_path(patch_xml())
    validate_model_contract(mj)
    return mj


def candidate_pairs(mj):
    """Count geom pairs MJX will allocate constraint rows for.

    MJX sizes its contact buffers from the static candidate set, not from what
    is actually touching, so this number -- not the number of live contacts --
    is what drives GPU memory.  Mirrors MuJoCo's own filtering rules, including
    the exception that parent/child filtering does NOT apply when the parent is
    the world body.
    """
    floor = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self_pairs = ground_pairs = 0
    excl = {(mj.exclude_signature[i] >> 16, mj.exclude_signature[i] & 0xFFFF)
            for i in range(mj.nexclude)}
    for a in range(mj.ngeom):
        for b in range(a + 1, mj.ngeom):
            ca = (mj.geom_contype[a] & mj.geom_conaffinity[b]) or \
                 (mj.geom_contype[b] & mj.geom_conaffinity[a])
            if not ca:
                continue
            ba, bb = mj.geom_bodyid[a], mj.geom_bodyid[b]
            if ba == bb:
                continue
            if (min(ba, bb), max(ba, bb)) in excl:
                continue
            # parent/child filter, world body exempted
            if mj.body_parentid[bb] == ba and ba != 0:
                continue
            if mj.body_parentid[ba] == bb and bb != 0:
                continue
            if floor in (a, b):
                ground_pairs += 1
            else:
                self_pairs += 1
    return self_pairs, ground_pairs


def settle(mj, seconds=1.5, pose=None):
    """Drop the robot from the nominal stance and let it come to rest on CPU."""
    d = mujoco.MjData(mj)
    q = np.array(DEFAULT_POSE if pose is None else pose)
    qpos_indices, _ = joint_state_indices(mj)
    d.qpos[:] = stand_key_qpos(mj)
    d.qpos[:3] = [0.0, 0.0, NOMINAL_H + 0.005]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[qpos_indices] = q
    d.ctrl[:] = q
    for _ in range(int(seconds / mj.opt.timestep)):
        mujoco.mj_step(mj, d)
    return d


def stand_height(mj):
    """Height the robot actually settles at, which is what the servos give.

    The stand-keyframe height has 1 mm mesh clearance; the servos are soft
    (kp = 5), so the real stance sags.  If the robot collapses entirely we
    fall back to the keyframe value rather than spawning it underground.  A
    collapsed reading used as a spawn height produced invalid rewards before.
    """
    z = float(settle(mj).qpos[2])
    if z < 0.5 * NOMINAL_H:
        print(f"  ! settled to {z:.4f} m, far below the stand-keyframe "
              f"{NOMINAL_H:.4f} m -- the stance is collapsing.  Falling back "
              f"to the keyframe height; run `tune` to find solver settings "
              f"that hold the pose.")
        return NOMINAL_H
    return z


def domain_randomize(sys, rng):
    """Per-environment physics variation, in the shape Brax's PPO expects."""
    @jax.vmap
    def rand(key):
        key, k = jax.random.split(key)
        fric = jax.random.uniform(k, (), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(fric)

        key, k = jax.random.split(key)
        g = jax.random.uniform(k, (), minval=0.75, maxval=1.25)
        gain = sys.actuator_gainprm.at[:, 0].set(sys.actuator_gainprm[:, 0] * g)
        bias = sys.actuator_biasprm.at[:, 1].set(sys.actuator_biasprm[:, 1] * g)

        key, k = jax.random.split(key)
        m = jax.random.uniform(k, (), minval=0.85, maxval=1.20)
        mass = sys.body_mass.at[1].set(sys.body_mass[1] * m)
        return friction, gain, bias, mass

    friction, gain, bias, mass = rand(rng)
    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0, "actuator_gainprm": 0,
        "actuator_biasprm": 0, "body_mass": 0,
    })
    sys = sys.tree_replace({
        "geom_friction": friction, "actuator_gainprm": gain,
        "actuator_biasprm": bias, "body_mass": mass,
    })
    return sys, in_axes


# =========================================================== interrupts
# The first Ctrl+C asks for a clean stop at the next eval; a second quits now.
# Neither can land during an XLA compile: the signal is queued until Python
# regains control, and it also reaches the whole foreground process group, so
# a compile in flight may report `ptxas exited with non-zero error code`.  That
# message is the child being killed, not a corrupted build.
_INT = {"n": 0}


def _install_sigint():
    def handler(sig, frame):
        _INT["n"] += 1
        if _INT["n"] == 1:
            print("\n[Ctrl+C] stop requested — exiting at the next eval, with "
                  "the policy saved.\n          Press Ctrl+C again to quit now "
                  "(the last checkpoint is already on disk).", flush=True)
        else:
            print("\n[Ctrl+C] quitting now.", flush=True)
            _sys.stdout.flush()
            os._exit(130)
    signal.signal(signal.SIGINT, handler)


def _hms(s):
    s = int(max(s, 0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


class Ticker:
    """Live progress bar in the gap between videos.

    Brax exposes no per-step hook, so this is a TIME estimate: it measures how
    long each eval interval takes and predicts the next.  The first interval
    carries the one-off compile and is excluded from the calibration.
    """
    WIDTH = 30

    def __init__(self, total_evals):
        self.total_evals = total_evals
        self.est = None
        self.done = 0
        self.t0 = self.run_t0 = time.time()
        self._stop = self._thread = None
        self.on = _sys.stdout.isatty()

    def start(self):
        if not self.on:
            return
        self.t0 = time.time()
        self._stop = threading.Event()

        def loop():
            while not self._stop.wait(1.0):
                el = time.time() - self.t0
                if self.est:
                    frac = min(el / self.est, 0.999)
                    filled = int(self.WIDTH * frac)
                    bar = "#" * filled + "-" * (self.WIDTH - filled)
                    left = f"next video in ~{_hms(self.est - el)}"
                else:
                    pos = int(el / 2) % self.WIDTH
                    bar = "-" * pos + ">" + "-" * (self.WIDTH - pos - 1)
                    frac, left = 0.0, "first interval (includes compile)"
                _sys.stdout.write(
                    f"\r  [{bar}] {frac * 100:5.1f}%  "
                    f"eval {self.done + 1}/{self.total_evals}  "
                    f"elapsed {_hms(el)}  {left}      ")
                _sys.stdout.flush()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        el = time.time() - self.t0
        if self.on and self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            _sys.stdout.write("\r" + " " * 110 + "\r")
            _sys.stdout.flush()
        self.done += 1
        if self.done > 1:
            self.est = el if self.est is None else 0.7 * self.est + 0.3 * el
        return el

    def eta(self):
        if not self.est:
            return "?"
        return _hms(self.est * max(self.total_evals - self.done, 0))


# ================================================================== env
class Walk3DEnv(PipelineEnv):
    """Velocity-command-conditioned walking for curl_robot_3d."""

    def __init__(self):
        mj = load_mj()
        self._nom_h = stand_height(mj)
        self._init_z = self._nom_h + 0.005          # spawn just clear of the floor
        joint_qpos, joint_dof = joint_state_indices(mj)
        self._joint_qpos = jp.asarray(joint_qpos)
        self._joint_dof = jp.asarray(joint_dof)
        self._stand_qpos = jp.asarray(stand_key_qpos(mj))

        self._foot_site = np.array([
            mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            for leg in LEGS])
        self._shank_body = np.array([
            mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_shank")
            for leg in LEGS])
        if (self._foot_site < 0).any() or (self._shank_body < 0).any():
            raise RuntimeError("foot sites or shank bodies not found in the XML")

        sys = mjcf.load_model(mj)
        super().__init__(sys=sys, backend="mjx", n_frames=N_FRAMES)

        # Resolved here, not in step(): self.dt is a jax Array, so int() on
        # it works outside a trace but raises ConcretizationTypeError inside
        # jit. Anything step() needs as a Python int has to be fixed now.
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
        """World foot positions and velocities.

        Site velocity is not stored, so it is transferred from the shank link's
        spatial velocity through the offset to the foot.  `xd` has no world
        body, hence the -1 on the body index.
        """
        pos = ps.site_xpos[self._foot_site]
        offset = base.Transform.create(pos=pos - ps.xpos[self._shank_body])
        vel = offset.vmap().do(ps.xd.take(self._shank_body - 1)).vel
        return pos, vel

    def _obs(self, ps, info, rng):
        inv_rot = math.quat_inv(ps.x.rot[0])
        obs = jp.concatenate([
            info["command"] * jp.array([2.0, 2.0, 0.25]),
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv_rot),   # projected gravity
            math.rotate(ps.xd.ang[0], inv_rot) * 0.25,          # gyro, body frame
            math.rotate(ps.xd.vel[0], inv_rot) * 2.0,           # linvel, body frame
            ps.q[self._joint_qpos] - DEFAULT_POSE,
            ps.qd[self._joint_dof] * 0.05,
            info["last_act"],
        ])
        if OBS_NOISE:
            obs = obs + OBS_NOISE * NOISE_SIGMA * jax.random.normal(
                rng, obs.shape)
        return obs

    # -------------------------------------------------------------- reset
    def reset(self, rng):
        rng, k_pose, k_yaw, k_z, k_cmd, k_obs = jax.random.split(rng, 6)

        yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
        quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        # one-sided height noise: never spawn below the settled stance
        z = self._init_z + jax.random.uniform(k_z, (), minval=0.0, maxval=0.02)
        joints = jp.clip(
            DEFAULT_POSE + jax.random.uniform(
                k_pose, (12,), minval=-0.05, maxval=0.05),
            CTRL_LO, CTRL_HI)

        qpos = (self._stand_qpos
                .at[:3].set(jp.array([0.0, 0.0, z]))
                .at[3:7].set(quat)
                .at[self._joint_qpos].set(joints))
        ps = self.pipeline_init(qpos, jp.zeros(self.sys.nv))

        info = {
            "rng": rng,
            "command": self._sample_command(k_cmd),
            "last_act": jp.zeros(12),
            "act_filt": jp.zeros(12),
            "last_ctrl": DEFAULT_POSE,
            "air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "step": jp.int32(0),
        }
        metrics = {k: jp.zeros(()) for k in
                   ("track_lin", "track_ang", "air", "vx", "vy", "wz",
                    "height", "cmd_vx", "cmd_wz")}
        obs = self._obs(ps, info, k_obs)
        return State(ps, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # --------------------------------------------------------------- step
    def step(self, state, action):
        info = dict(state.info)
        rng, k_push, k_pv, k_cmd, k_lat, k_obs = jax.random.split(
            info["rng"], 6)

        # --- action -> servo target -----------------------------------
        action = jp.clip(action, -1.0, 1.0)
        act_filt = (ACTION_FILTER * info["act_filt"]
                    + (1.0 - ACTION_FILTER) * action) if ACTION_FILTER else action
        ctrl = jp.clip(DEFAULT_POSE + act_filt * ACTION_SCALE, CTRL_LO, CTRL_HI)
        if LATENCY_PROB:
            late = jax.random.uniform(k_lat) < LATENCY_PROB
            ctrl = jp.where(late, info["last_ctrl"], ctrl)

        # --- optional shove -------------------------------------------
        ps_in = state.pipeline_state
        if PUSH_EVERY:
            hit = jax.random.uniform(k_push) < (self.dt / PUSH_EVERY)
            dv = jax.random.normal(k_pv, (2,)) * PUSH_MAG
            qvel = ps_in.qvel.at[:2].add(jp.where(hit, dv, jp.zeros(2)))
            ps_in = ps_in.replace(qvel=qvel, qd=qvel)

        ps = self.pipeline_step(ps_in, ctrl)

        # --- frames ----------------------------------------------------
        inv_rot = math.quat_inv(ps.x.rot[0])
        lin_b = math.rotate(ps.xd.vel[0], inv_rot)
        ang_b = math.rotate(ps.xd.ang[0], inv_rot)
        up = math.rotate(jp.array([0.0, 0.0, 1.0]), ps.x.rot[0])
        cmd = info["command"]
        moving = jp.linalg.norm(cmd) > 0.05

        # --- feet, contact and air time -------------------------------
        foot_pos, foot_vel = self._feet(ps)
        contact = (foot_pos[:, 2] - FOOT_R) < 1e-3
        contact_filt = contact | info["last_contact"]
        first_contact = (info["air_time"] > 0.0) & contact_filt
        r_air = AIR_TIME_W * jp.sum(
            (info["air_time"] - AIR_TIME_TARGET) * first_contact) * moving
        air_time = (info["air_time"] + self.dt) * ~contact_filt

        # --- reward ----------------------------------------------------
        r_lin = TRACK_LIN_W * jp.exp(
            -jp.sum(jp.square(cmd[:2] - lin_b[:2])) / TRACK_SIGMA)
        r_ang = TRACK_ANG_W * jp.exp(
            -jp.square(cmd[2] - ang_b[2]) / TRACK_SIGMA)

        tau = ps.qfrc_actuator[6:]
        p_orient = ORIENT_W * jp.sum(jp.square(up[:2]))
        p_linz = LIN_Z_W * jp.square(lin_b[2])
        p_angxy = ANG_XY_W * jp.sum(jp.square(ang_b[:2]))
        p_height = HEIGHT_W * jp.square(ps.q[2] - self._nom_h)
        p_torque = TORQUE_W * jp.sum(jp.square(tau))
        p_jvel = JVEL_W * jp.sum(jp.square(ps.qd[6:]))
        p_rate = RATE_W * jp.sum(jp.square(action - info["last_act"]))
        p_slip = SLIP_W * jp.sum(
            jp.sum(jp.square(foot_vel[:, :2]), axis=1) * contact)
        p_stand = STAND_W * (1.0 - moving) * jp.sum(
            jp.abs(ps.q[self._joint_qpos] - DEFAULT_POSE))

        bad = jp.isnan(ps.q).any() | jp.isnan(ps.qd).any()
        done = ((ps.q[2] < Z_MIN) | (up[2] < UP_MIN) | bad).astype(jp.float32)

        reward = (ALIVE_W + r_lin + r_ang + r_air
                  - p_orient - p_linz - p_angxy - p_height
                  - p_torque - p_jvel - p_rate - p_slip - p_stand
                  - TERM_W * done)
        reward = jp.clip(reward, -5.0, 10.0)

        # --- bookkeeping ----------------------------------------------
        step_i = info["step"] + 1
        resample = (step_i % self._cmd_steps) == 0
        info["command"] = jp.where(resample, self._sample_command(k_cmd), cmd)
        info["rng"] = rng
        info["last_act"] = action
        info["act_filt"] = act_filt
        info["last_ctrl"] = ctrl
        info["air_time"] = air_time * (1.0 - done)
        info["last_contact"] = contact
        info["step"] = step_i

        # Start from the incoming metrics rather than a fresh dict: brax's
        # EvalWrapper injects its own 'reward' key at reset, and the
        # action-repeat scan requires the carry's pytree structure to be
        # identical in and out.  Replacing the dict wholesale drops that key.
        metrics = dict(state.metrics)
        metrics.update({
            "track_lin": r_lin, "track_ang": r_ang, "air": r_air,
            "vx": lin_b[0], "vy": lin_b[1], "wz": ang_b[2],
            "height": ps.q[2], "cmd_vx": cmd[0], "cmd_wz": cmd[2],
        })
        obs = self._obs(ps, info, k_obs)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward,
                             done=done, metrics=metrics, info=info)


# ================================================================ video
# Each clip drives the policy through a fixed command script, so one video
# shows every behaviour we asked for instead of just whichever direction the
# random command happened to pick.
CMD_SCRIPT = (
    ("forward",   jp.array([0.45, 0.0, 0.0])),
    ("backward",  jp.array([-0.45, 0.0, 0.0])),
    ("turn left", jp.array([0.0, 0.0, 1.0])),
    ("turn right", jp.array([0.20, 0.0, -1.0])),
)


def _yaw(q):
    """Heading from the free-joint quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q[3:7])
    return float(np.arctan2(2 * (w * z + x * y),
                            1 - 2 * (y * y + z * z)))


def _scripted_rollout(env, act_fn, seconds, step_fn=None, reset_fn=None,
                      rng_seed=1):
    """Roll the policy through CMD_SCRIPT, returning the trace and a report.

    step_fn/reset_fn are accepted pre-jitted so the training loop compiles
    them once instead of on every eval.
    """
    step_fn = step_fn or jax.jit(env.step)
    reset_fn = reset_fn or jax.jit(env.reset)
    st = reset_fn(jax.random.PRNGKey(0))
    rng = jax.random.PRNGKey(rng_seed)
    per_seg = max(int(seconds / len(CMD_SCRIPT) / env.dt), 1)

    roll, report, fell = [st.pipeline_state], [], False
    for name, cmd in CMD_SCRIPT:
        st = st.replace(info={**st.info, "command": cmd})
        p0 = np.array(st.pipeline_state.q[:3])
        yaw0 = _yaw(st.pipeline_state.q)
        for _ in range(per_seg):
            rng, k = jax.random.split(rng)
            act = act_fn(st.obs, k)
            st = step_fn(st, act)
            # keep the scripted command; step() may have resampled it
            st = st.replace(info={**st.info, "command": cmd})
            roll.append(st.pipeline_state)
            if bool(st.done):
                fell = True
                break
        p1 = np.array(st.pipeline_state.q[:3])
        yaw1 = _yaw(st.pipeline_state.q)
        dt_seg = per_seg * env.dt
        dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
        report.append(f"    {name:<11} {np.linalg.norm(p1[:2] - p0[:2]) / dt_seg:5.2f} m/s"
                      f"   yaw {dyaw / dt_seg:+5.2f} rad/s"
                      + ("   [FELL]" if fell else ""))
        if fell:
            break
    return roll, report


def make_video(policy_path=None, seconds=None, out=None):
    """Render a saved policy: python train_ppo_walk3d.py video [policy.bin]"""
    from brax.training.acme import running_statistics
    policy_path = policy_path or SAVE
    os.makedirs(VID_DIR, exist_ok=True)
    env = Walk3DEnv()
    nets = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN,
        value_hidden_layer_sizes=VALUE_HIDDEN)
    inf = jax.jit(ppo_networks.make_inference_fn(nets)(
        model.load_params(policy_path), deterministic=True))

    roll, report = _scripted_rollout(
        env, lambda o, k: inf(o, k)[0], seconds or (VIDEO_SECONDS * 2))
    print("\n".join(report))
    frames = PipelineEnv.render(env, roll, height=480, width=640,
                                camera="track")
    out = out or os.path.join(VID_DIR, "showcase.mp4")
    media.write_video(out, frames, fps=1.0 / env.dt)
    print(f"video: {out}")


# ================================================================ probe
def probe():
    """Everything worth checking before spending GPU hours."""
    mj = load_mj()
    print(f"model    nq={mj.nq}  nv={mj.nv}  nu={mj.nu}  "
          f"nbody={mj.nbody}  ngeom={mj.ngeom}")
    print(f"mass     {mj.body_mass.sum():.4f} kg   "
          f"weight {mj.body_mass.sum() * 9.81:.2f} N")

    print("\nkeyframe check — named joints and CAD foot-site clearance")
    qpos_indices, _ = joint_state_indices(mj)
    foot_site_ids = np.asarray([
        mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
        for leg in LEGS
    ])
    data = mujoco.MjData(mj)
    for i in range(mj.nkey):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_KEY, i)
        mujoco.mj_resetDataKeyframe(mj, data, i)
        mujoco.mj_forward(mj, data)
        joints = data.qpos[qpos_indices]
        foot_bottom = data.site_xpos[foot_site_ids, 2] - FOOT_R
        print(f"  {name:<16} stored z={float(data.qpos[2]):.6f}  "
              f"min foot bottom={float(np.min(foot_bottom)):+.6f}  "
              f"joints finite={bool(np.isfinite(joints).all())}")

    print(f"\nstance    default pose {np.array(DEFAULT_POSE)[:3]} per leg")
    print(f"          stand key height    {NOMINAL_H:.4f} m "
          f"(open key: {FULL_H:.4f} m)")
    t0 = time.time()
    d = settle(mj, 2.0)
    print(f"          settled height        {float(d.qpos[2]):.4f} m "
          f"after 2.0 s  ({time.time() - t0:.1f} s wall)")
    print(f"          sag from stand key   {NOMINAL_H - float(d.qpos[2]):+.4f} m "
          f"(soft servos: kp={mj.actuator_gainprm[0, 0]:.1f})")
    tilt = np.degrees(2 * np.arccos(np.clip(abs(d.qpos[3]), 0, 1)))
    print(f"          tilt after settling   {tilt:.2f} deg")

    self_p, ground_p = candidate_pairs(mj)
    print(f"\ncontact   self-collision={SELF_COLLISION} shells={SHELL_CONTACT} "
          f"motor-self={MOTOR_SELF_COLLIDE}")
    print(f"          -> {self_p} robot-vs-robot pairs, {ground_p} vs floor "
          f"({self_p + ground_p} total)")
    print(f"          MJX sizes its constraint buffers from this count, so it "
          f"drives GPU memory\n          directly -- not the number of live "
          f"contacts.")
    print(f"          solver iterations={SOLVER_ITER} "
          f"ls_iterations={SOLVER_LS_ITER} cone={CONE} impratio={IMPRATIO}")
    print(f"          timestep={PHYS_TIMESTEP * 1000:.0f} ms, "
          f"{N_FRAMES} frames/step -> control at {1 / (PHYS_TIMESTEP * N_FRAMES):.0f} Hz")

    print(f"\ntorque    forcerange {mj.actuator_forcerange[0]} N·m per motor "
          f"(one motor per joint on this model)")
    static = mj.body_mass.sum() * 9.81 / 4.0
    lever = L_THIGH * sin(0.6)
    print(f"          static load per leg {static:.2f} N, hip lever "
          f"{lever * 1000:.1f} mm -> {static * lever:.3f} N·m holding torque")

    print("\njoint sign convention in the world frame at stand:")
    mujoco.mj_resetDataKeyframe(mj, data, mj.key("stand").id)
    mujoco.mj_forward(mj, data)
    for leg in LEGS:
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip")
        axis = data.xaxis[jid]
        fwd = "forward" if axis[1] < 0 else "backward"
        print(f"  {leg:<12} hip axis {axis}  -> +hip swings the foot {fwd}")
    print("  The chains are mirrored front-to-rear, so the robot has no "
          "built-in 'front'.\n  That is exactly why one policy can walk both "
          "directions.")

    env = Walk3DEnv()
    print(f"\nenv       obs={env.observation_size} (expected {OBS_SIZE})  "
          f"action={env.action_size}  dt={env.dt:.4f}")
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    st = jax.jit(env.step)(st, jp.zeros(12))
    print(f"          after one step: reward={float(st.reward):+.3f}  "
          f"done={float(st.done):.0f}  z={float(st.pipeline_state.q[2]):.4f}")
    print(f"          command sampled: {np.array(st.info['command'])}")
    print(f"\ncommands  vx {CMD_VX} m/s   vy {CMD_VY} m/s   "
          f"yaw {CMD_WZ} rad/s")
    print(f"          resampled every {CMD_RESAMPLE:.0f} s, "
          f"{ZERO_CMD_PROB * 100:.0f}% are stand-still")


def tune_table():
    """Sweep solver settings; report settled stance height and CPU cost.

    A pose that collapses here will collapse in MJX too, and a collapsed
    stance is what produced the nonsense rewards in the 2-D run.
    """
    global SOLVER_ITER, SOLVER_LS_ITER, CONE, IMPRATIO
    keep = (SOLVER_ITER, SOLVER_LS_ITER, CONE, IMPRATIO)
    print(f"{'iters':>6} {'ls':>4} {'cone':>10} {'impratio':>9} "
          f"{'height':>8} {'tilt':>7} {'wall':>7}")
    print("-" * 58)
    for cone in ("pyramidal", "elliptic"):
        for it, ls in ((1, 5), (2, 6), (4, 8), (8, 10), (20, 10)):
            for imp in (1.0, 10.0):
                SOLVER_ITER, SOLVER_LS_ITER, CONE, IMPRATIO = it, ls, cone, imp
                try:
                    mj = load_mj()
                    t0 = time.time()
                    d = settle(mj, 1.5)
                    z = float(d.qpos[2])
                    tilt = np.degrees(
                        2 * np.arccos(np.clip(abs(d.qpos[3]), 0, 1)))
                    print(f"{it:>6} {ls:>4} {cone:>10} {imp:>9.1f} "
                          f"{z:>8.4f} {tilt:>6.1f}° {time.time() - t0:>6.2f}s"
                          + ("   <-- collapsed" if z < 0.5 * NOMINAL_H else ""))
                except Exception as e:
                    print(f"{it:>6} {ls:>4} {cone:>10} {imp:>9.1f}  failed: {e}")
    SOLVER_ITER, SOLVER_LS_ITER, CONE, IMPRATIO = keep
    print(f"\nstand-key target height {NOMINAL_H:.4f} m; pick the cheapest row "
          f"that holds it.")


def enable_dr():
    """`dr` mode: physics variation plus noise, latency and shoves.

    Writes to its own files so the flat-ground policy is never overwritten.
    """
    global DOMAIN_RANDOMIZE, OBS_NOISE, PUSH_EVERY, LATENCY_PROB
    global SAVE, VID_DIR, CKPT_DIR, NUM_TIMESTEPS
    DOMAIN_RANDOMIZE = True
    OBS_NOISE = 1.0
    PUSH_EVERY = 2.5
    LATENCY_PROB = 0.15
    SAVE = "rollingquad_2_walk3d_policy_dr.bin"
    VID_DIR = "rollingquad_2_walk3d_videos_dr"
    CKPT_DIR = "rollingquad_2_walk3d_checkpoints_dr"
    NUM_TIMESTEPS = 300_000_000


# =============================================================== training
def main():
    _install_sigint()
    os.makedirs(VID_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    env, eval_env = Walk3DEnv(), Walk3DEnv()

    print("=" * 72)
    print("rollingquad_2 — omnidirectional walking PPO")
    print(f"  obs {env.observation_size}  action {env.action_size}  "
          f"control {1 / env.dt:.0f} Hz  episode {EPISODE_LENGTH * env.dt:.0f} s")
    print(f"  stance height {env._nom_h:.4f} m   "
          f"commands vx{CMD_VX} vy{CMD_VY} yaw{CMD_WZ}")
    print(f"  {NUM_TIMESTEPS:,} steps over {NUM_ENVS} envs, "
          f"{NUM_EVALS} evals -> a video every "
          f"{NUM_TIMESTEPS // NUM_EVALS:,} steps")
    print(f"  domain randomisation: {'ON' if DOMAIN_RANDOMIZE else 'off'}")
    print(f"  writing to {SAVE}, {CKPT_DIR}/, {VID_DIR}/")
    print("  Ctrl+C once = stop cleanly at the next eval. During the first "
          "compile the\n  signal is queued, so it will not appear to respond "
          "for a few minutes.")
    print("=" * 72, flush=True)

    resume = {}
    if os.path.exists(SAVE):
        resume["restore_params"] = model.load_params(SAVE)
        print(f"RESUMING from {SAVE}\n", flush=True)

    ticker = Ticker(NUM_EVALS)

    def progress(step, metrics):
        took = ticker.stop()
        r = metrics.get("eval/episode_reward", float("nan"))
        n = metrics.get("eval/avg_episode_length", float("nan"))
        tl = metrics.get("eval/episode_track_lin", float("nan"))
        ta = metrics.get("eval/episode_track_ang", float("nan"))
        pct = 100.0 * step / max(NUM_TIMESTEPS, 1)
        print(f"[{ticker.done}/{NUM_EVALS}] step {step:>13,} ({pct:4.1f}%)  "
              f"reward {r}  ep_len {n}", flush=True)
        print(f"    track_lin {tl}  track_ang {ta}", flush=True)
        print(f"    took {_hms(took)}  |  elapsed "
              f"{_hms(time.time() - ticker.run_t0)}  |  ETA {ticker.eta()}",
              flush=True)
        ticker.start()

    # Compiled once and reused by every eval.  Building these inside the
    # callback would retrace the whole MJX graph 25 times, and closing over
    # `params` would bake them in as constants, forcing a fresh compile on
    # every call.  Params are passed as an ordinary argument instead.
    jit_cache = {}

    def _tools(make_policy):
        if not jit_cache:
            jit_cache["act"] = jax.jit(
                lambda p, o, k: make_policy(p, deterministic=True)(o, k)[0])
            jit_cache["step"] = jax.jit(eval_env.step)
            jit_cache["reset"] = jax.jit(eval_env.reset)
        return jit_cache["act"], jit_cache["step"], jit_cache["reset"]

    def policy_params_fn(step, make_policy, params):
        model.save_params(os.path.join(CKPT_DIR, f"{step:012d}.bin"), params)
        model.save_params(SAVE, params)
        if _INT["n"]:
            raise KeyboardInterrupt      # weights are already on disk
        try:
            act_fn, step_fn, reset_fn = _tools(make_policy)
            roll, report = _scripted_rollout(
                eval_env, lambda o, k: act_fn(params, o, k),
                VIDEO_SECONDS * 2, step_fn=step_fn, reset_fn=reset_fn)
            frames = PipelineEnv.render(eval_env, roll, height=480, width=640,
                                        camera="track")
            v = os.path.join(VID_DIR, f"walk3d_{step:012d}.mp4")
            media.write_video(v, frames, fps=1.0 / eval_env.dt)
            print("\n".join(report), flush=True)
            print(f"    checkpoint + video: {v}  ({len(roll)} frames)",
                  flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    video failed ({e}); checkpoint still saved", flush=True)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=POLICY_HIDDEN,
        value_hidden_layer_sizes=VALUE_HIDDEN)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=NUM_TIMESTEPS,
        num_evals=NUM_EVALS,
        episode_length=EPISODE_LENGTH,
        num_envs=NUM_ENVS,
        batch_size=BATCH_SIZE,
        num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=NUM_UPDATES_PER_BATCH,
        unroll_length=UNROLL_LENGTH,
        discounting=DISCOUNTING,
        learning_rate=LEARNING_RATE,
        entropy_cost=ENTROPY_COST,
        reward_scaling=1.0,
        normalize_observations=True,
        action_repeat=1,
        network_factory=network_factory,
        randomization_fn=domain_randomize if DOMAIN_RANDOMIZE else None,
        policy_params_fn=policy_params_fn,
        seed=SEED,
        **resume,
    )

    ticker.start()
    try:
        _, params, _ = train_fn(environment=env, progress_fn=progress,
                                eval_env=eval_env)
        model.save_params(SAVE, params)
        print(f"\ndone — policy in {SAVE}", flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped — newest policy in {SAVE} (and {CKPT_DIR}/)",
              flush=True)


if __name__ == "__main__":
    import traceback
    code = 0
    try:
        argv = _sys.argv[1:]
        if "dr" in argv:
            enable_dr()
            argv.remove("dr")
        cmd = argv[0] if argv else "train"
        if cmd == "probe":
            probe()
        elif cmd == "tune":
            tune_table()
        elif cmd == "video":
            make_video(argv[1] if len(argv) > 1 else None)
        else:
            main()
    except BaseException:
        traceback.print_exc()
        code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(code)
