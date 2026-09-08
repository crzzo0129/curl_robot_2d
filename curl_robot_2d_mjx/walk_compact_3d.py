"""Walking-start compact transition: contracts, gates, snapshot IO.

Stage-one skill of the walk -> roll direction: the episode starts from a real
0.4 m/s walking state (collected with the deploy-interface walking policy) and
a 12-DoF actor curls the robot into the compact pose while it decelerates.
The terminal gate is pose-only: joint positions, root height, orientation and
lateral drift.  Velocities are deliberately NOT gated and no rolling teacher is
activated in this stage (see docs/walking_to_compact_stage1_zh.md).

All gate/reward helpers take an ``xp`` argument (numpy or jax.numpy) so the
same formulas run in the CPU contracts tests and inside the MJX environment.

Physics/contact scope (2026-09-08, per design decision):
  * mesh model rollingquad_abd10.xml (compact keyframe: front abd -10 deg,
    rear abd +10 deg), ground contacts enabled, self-collision OFF by the
    source XML (default geom contype=0 conaffinity=1, no pair/exclude);
  * runtime XML = source with <option> replaced (0.002 s implicitfast,
    pyramidal, Newton 20/10, impratio 10) to match the CPU snapshot rolls;
  * observation contract = deploy neural controller: 36 dims x 20 frames,
    12-DoF absolute position targets pose + scale * action.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- contracts

WALK_COMPACT_CONTRACT = "walking_0p4_to_compact_v1_pose_gate_mesh_abd10"
GEOMETRY = "rollingquad_2_abd10"
# Source mesh model with the rolling self-collision whitelist baked in.
ABD10_SOURCE_XML_REL = Path("assets") / "rollingquad_description_2" / "mjcf" / "rollingquad_abd10.xml"
# Dedicated variant for the walk->compact stage: self-collision DISABLED,
# ground contact kept (all robot geoms become contype=0 conaffinity=1).
MESH_XML_REL = Path("assets") / "rollingquad_description_2" / "mjcf" / "rollingquad_abd10_no_self_collision.xml"
COMMAND_M_S = 0.4

# Canonical policy order (== actuator order in the XML, per its own comment):
# legs FL, FR, RL, RR; per leg abduction, hip, knee.
POLICY_LEGS = ("front_left", "front_right", "rear_left", "rear_right")
POLICY_JOINT_SUFFIXES = ("hip_abduction", "hip", "knee")

# Deploy observation contract (neural_controller layout, see train_ppo_deploy).
SINGLE_OBS_SIZE = 36
HISTORY_SIZE = 20
OBSERVATION_SIZE = SINGLE_OBS_SIZE * HISTORY_SIZE
ACTION_SIZE = 12

# Physics parity with the CPU snapshot collector.
PHYSICS_TIMESTEP_S = 0.002
CONTROL_TIMESTEP_S = 0.02
RUNTIME_OPTION = (
    '<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" '
    'cone="pyramidal" iterations="20" ls_iterations="10" impratio="10">'
    '<flag eulerdamp="disable"/></option>'
)

COMPACT_GATE_NAMES = ("joint_position", "roll",
                       "base_velocity", "base_angular_velocity", "root_height")

# Deploy frame layout, indices 0-based.
FRAME_GYRO = slice(0, 3)            # base angular velocity, body frame
FRAME_GRAVITY = slice(3, 6)         # projected gravity, body frame
FRAME_COMMAND = slice(6, 9)         # vx, vy, yaw-rate command
FRAME_DESIRED_Z = slice(9, 12)      # desired world z in body frame
FRAME_JOINTS = slice(12, 24)        # joint position - default_joint_pos
FRAME_LAST_ACTION = slice(24, 36)   # previous raw policy output


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def xml_fingerprint(path: Path):
    raw = Path(path).read_bytes()
    return {"xml_sha256": sha256_bytes(raw),
            "xml_lf_sha256": sha256_bytes(raw.replace(b"\r\n", b"\n"))}


_SELF_COLLISION_GEOM_RE = re.compile(r"<geom\b[^>]*>")


def disable_self_collision_xml(xml_text: str) -> str:
    """Rewrite every named geom except the floor to ground-only contact.

    The rolling mesh model encodes a selective self-collision whitelist with
    contype/conaffinity bitmasks (torso 16/7, front leg 2/29, rear leg 4/27,
    foot 8/15).  Setting each named robot geom to contype=0 conaffinity=1 keeps
    ground contact (the floor is contype=1 conaffinity=0) while disabling all
    robot-robot collisions, which is what the walk->compact stage wants.
    """
    def rewrite(match):
        tag = match.group(0)
        name = re.search(r'name="([^"]*)"', tag)
        if not name or name.group(1) == "floor":
            return tag  # leave the anonymous default geom and the floor
        tag = re.sub(r'contype="[^"]*"', 'contype="0"', tag)
        tag = re.sub(r'conaffinity="[^"]*"', 'conaffinity="1"', tag)
        return tag

    return _SELF_COLLISION_GEOM_RE.sub(rewrite, xml_text)


def write_no_self_collision_variant(source_xml: Path, dst_xml: Path) -> Path:
    """Write the no-self-collision variant of a rolling mesh MJCF."""
    xml = Path(source_xml).read_text(encoding="utf-8")
    dst = Path(dst_xml)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(disable_self_collision_xml(xml), encoding="utf-8")
    return dst


def prepare_runtime_xml(source_xml: Path, dst_xml: Path) -> Path:
    """Copy the mesh XML, replacing <option> and pinning mesh resolution.

    Contacts stay exactly as authored in the source (ground contact on,
    self-collision off).  The source MJCF references meshes with relative
    paths (``../meshes/*.stl``), so a ``meshdir`` pointing back at the source
    MJCF directory is injected into <compiler> -- otherwise the copied XML
    under the output directory cannot find the CAD meshes.  Returns the
    runtime path.
    """
    xml = Path(source_xml).read_text(encoding="utf-8")
    xml, count = re.subn(r"<option\b.*?/>", RUNTIME_OPTION, xml, count=1,
                         flags=re.S)
    if count != 1:
        raise ValueError(f"could not replace <option> in {source_xml}")

    mesh_dir = Path(source_xml).resolve().parent.as_posix()

    def preserve_mesh_dir(match):
        tag = match.group(0)
        if "meshdir=" in tag:
            return tag
        return tag[:-2] + f' meshdir="{mesh_dir}"/>'

    xml, ncompiler = re.subn(r"<compiler\b[^>]*/>", preserve_mesh_dir,
                             xml, count=1)
    if ncompiler != 1:
        raise ValueError(f"could not patch <compiler> meshdir in {source_xml}")

    dst = Path(dst_xml)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(xml, encoding="utf-8")
    return dst


def policy_joint_names() -> tuple[str, ...]:
    """12 joint names in canonical actuator/policy order."""
    return tuple(f"{leg}_{suffix}"
                 for leg in POLICY_LEGS for suffix in POLICY_JOINT_SUFFIXES)


def policy_actuator_names() -> tuple[str, ...]:
    return tuple(f"{name}_servo" for name in policy_joint_names())


# --------------------------------------------------------------- configuration

@dataclass(frozen=True)
class WalkCompactConfig:
    # Time budget from the walking snapshot to a confirmed compact window.
    # budget_s_max > budget_s enables a per-episode uniform random budget
    # (curriculum stage 4), so the policy cannot rely on a fixed 5 s window.
    budget_s: float = 5.0
    budget_s_max: float | None = None
    confirmation_steps: int = 10        # 0.2 s at 50 Hz
    # D_t = mean(((q - q_compact)/sigma)^2) -- the joint distance-to-compact.
    joint_cost_sigma_rad: float = 0.50
    # Terminal (success) gate -- a STATE-space target, not just joint-space.
    joint_position_rad: float = 0.10    # max |q - q_compact|
    roll_rad: float = 0.30              # |sideways lean| only (NOT forward pitch)
    base_linear_velocity_m_s: float = 0.30   # |v_xy|
    base_angular_velocity_rad_s: float = 1.00  # |omega|
    root_z_min_m: float = 0.10          # lower height envelope (block collapse)
    root_z_max_margin_m: float = 0.02   # upper envelope = max(stand,compact)+margin
    # Reward weights (transition-skill recipe).
    progress_reward_weight: float = 2.0
    progress_scale: float = 0.02        # divisor on (D_prev - D_t)
    pose_reward_weight: float = 0.5     # exp(-D_t)
    roll_stability_weight: float = 0.10   # sideways lean (roll) only
    roll_stability_sigma_rad: float = 0.30
    angular_velocity_stability_weight: float = 0.03   # base omega_xy
    angular_velocity_stability_sigma_rad_s: float = 2.0
    height_penalty_weight: float = 0.05
    height_sigma_m: float = 0.02
    smooth_weight: float = 0.05
    torque_cost: float = 0.01
    time_cost: float = 0.003
    success_bonus: float = 8.0
    discounting: float = 0.999

    def validate(self, dt: float) -> None:
        for name, value in asdict(self).items():
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.confirmation_steps, int) or self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be a positive integer")
        if not 0 < self.discounting < 1:
            raise ValueError("discounting must be in (0, 1)")
        if not np.isclose(round(self.budget_s / dt) * dt, self.budget_s, atol=1e-8, rtol=0):
            raise ValueError("budget_s must align with the control timestep")
        if self.budget_s_max is not None:
            if self.budget_s_max < self.budget_s:
                raise ValueError("budget_s_max must be >= budget_s")
            if not np.isclose(round(self.budget_s_max / dt) * dt, self.budget_s_max,
                              atol=1e-8, rtol=0):
                raise ValueError("budget_s_max must align with the control timestep")
        if self.confirmation_steps > round(self.budget_s / dt):
            raise ValueError("confirmation exceeds the episode budget")

    def episode_steps(self, dt: float) -> int:
        return round(self.budget_s / dt)

    def max_episode_steps(self, dt: float) -> int:
        return round((self.budget_s_max or self.budget_s) / dt)


# ------------------------------------------------------------------- gates

def compact_target_from_keyframe(key_qpos: np.ndarray,
                                 joint_qpos_indices: np.ndarray,
                                 *,
                                 root_z: float | None = None):
    """Policy-order compact target from an MjModel compact keyframe qpos.

    key_qpos uses the MuJoCo body-tree qpos order; joint_qpos_indices selects
    the 12 joint entries in policy order (see policy_joint_names).  The target
    carries joints + root height only; body orientation is NOT part of the
    compact definition (a curled ball is valid at any forward-roll phase), so
    sideways lean is gated separately via rolling-axis tilt.
    """
    key_qpos = np.asarray(key_qpos, dtype=np.float32)
    joints = np.asarray(key_qpos[joint_qpos_indices], dtype=np.float32).copy()
    return {"joints": joints,
            "root_z": float(root_z) if root_z is not None else float(key_qpos[2])}


def joint_cost(xp, joints, target, cfg):
    """D_t = mean(((q - q_compact)/sigma)^2) -- the distance-to-compact."""
    return xp.mean(xp.square(
        (joints - target["joints"]) / cfg.joint_cost_sigma_rad))


def progress_reward(xp, previous_cost, cost, cfg):
    """Potential-based PROGRESS reward (telescopes; intentionally UNCLIPPED).

    ``sum_t (D_{t-1} - D_t) = D_0 - D_T`` exactly, so the episode total depends
    only on NET progress toward compact, not on the path.  A ``clip()`` (as in
    the original recipe) breaks this telescoping and lets the policy harvest
    reward by tucking then un-tucking back -- which we observed as reward
    climbing while the terminal joint error stayed at the walking value.
    """
    return cfg.progress_reward_weight * (previous_cost - cost) / cfg.progress_scale


def pose_reward(xp, cost, cfg):
    """Gaussian pose reward exp(-D_t): bounded, far states not over-penalised."""
    return cfg.pose_reward_weight * xp.exp(-cost)


def roll_pitch_from_quat(xp, quat):
    """Roll (about body X) and pitch (about body Y) from a wxyz quaternion.

    Uses the body->world rotation matrix: roll = atan2(R21, R22),
    pitch = atan2(-R20, hypot(R21, R22)).  Both are 0 when the body is upright.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r20 = 2.0 * (x * z - w * y)
    roll = xp.arctan2(r21, r22)
    pitch = xp.arctan2(-r20, xp.sqrt(xp.maximum(r21 * r21 + r22 * r22, 1e-12)))
    return roll, pitch


def stability_cost(xp, roll, angular_xy, cfg):
    """Positive penalty: sideways lean (roll) + base angular velocity in xy.

    Forward pitch is deliberately NOT penalised: tucking onto the shell folds
    the body forward, and a curled ball is valid at any forward-roll phase.
    Mild weight: a transition may legitimately have some body disturbance.
    """
    tilt = xp.square(roll / cfg.roll_stability_sigma_rad)
    ang = xp.mean(xp.square(
        angular_xy / cfg.angular_velocity_stability_sigma_rad_s))
    return (cfg.roll_stability_weight * tilt
            + cfg.angular_velocity_stability_weight * ang)


def height_penalty(xp, root_z, stand_z, compact_z, cfg):
    """Positive penalty for leaving the WIDE [z_min, z_max] envelope.

    Upper bound blocks "jump up then curl"; lower bound blocks "collapse to the
    floor".  Deliberately wide (not height tracking) to leave the policy free.
    """
    z_max = xp.maximum(stand_z, compact_z) + cfg.root_z_max_margin_m
    upper = xp.square(xp.maximum(root_z - z_max, 0.0) / cfg.height_sigma_m)
    lower = xp.square(xp.maximum(cfg.root_z_min_m - root_z, 0.0) / cfg.height_sigma_m)
    return cfg.height_penalty_weight * (upper + lower)


def gate_errors(xp, joints, roll, velocity_norm, angular_norm, root_z,
                target, stand_z, cfg):
    """Normalized success-gate errors (1.0 == bound).  A STATE-space target:
    joint pose AND small sideways lean (roll) AND low base velocity/angular
    velocity AND root height inside the envelope, held for confirmation_steps.
    Forward pitch is NOT gated: the compact transition folds the body forward.
    """
    joint_error = xp.max(xp.abs(joints - target["joints"])) / cfg.joint_position_rad
    roll_error = xp.abs(roll) / cfg.roll_rad
    velocity_error = velocity_norm / cfg.base_linear_velocity_m_s
    angular_error = angular_norm / cfg.base_angular_velocity_rad_s
    z_max = xp.maximum(stand_z, target["root_z"]) + cfg.root_z_max_margin_m
    upper = xp.maximum(root_z - z_max, 0.0) / cfg.height_sigma_m
    lower = xp.maximum(cfg.root_z_min_m - root_z, 0.0) / cfg.height_sigma_m
    height_error = xp.maximum(upper, lower)
    return xp.stack((joint_error, roll_error, velocity_error,
                     angular_error, height_error))


def confirmation_update(xp, previous_id, previous_count, candidate_id, eligible):
    count = xp.where(eligible, xp.where(previous_id == candidate_id,
                                        previous_count + 1, 1), 0)
    return count.astype(xp.int32)


# ------------------------------------------------------------ snapshot format

def snapshot_arrays() -> tuple[str, ...]:
    return ("qpos", "qvel", "ctrl", "hist", "last_action", "time")


def validate_snapshot_bank(npz_path: Path, meta_path: Path | None = None):
    """Load and shape-check a walking snapshot bank; returns (arrays, meta).

    numpy-only so contract tests run without MuJoCo/JAX.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        arrays = {key: np.asarray(data[key], dtype=np.float32).copy()
                  for key in snapshot_arrays()}
    count = arrays["qpos"].shape[0]
    expected = {"qpos": (count, 19), "qvel": (count, 18), "ctrl": (count, 12),
                "hist": (count, OBSERVATION_SIZE),
                "last_action": (count, ACTION_SIZE), "time": (count,)}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"snapshot {key} shape {arrays[key].shape} != {shape}")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("snapshot bank contains non-finite values")
    meta = {}
    if meta_path is not None:
        meta_path = Path(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract") != WALK_COMPACT_CONTRACT:
            raise ValueError(f"snapshot meta contract mismatch: {meta.get('contract')}")
        if meta.get("count") != int(count):
            raise ValueError("snapshot meta count does not match the npz")
    return arrays, meta


def bank_action_arrays(meta: dict):
    """Policy pose/scale/limits (12 floats each) embedded in the snapshot meta."""
    action = meta.get("action")
    if not action or any(key not in action for key in ("default", "scale", "lower", "upper")):
        raise ValueError("snapshot meta action block missing default/scale/lower/upper")
    return {key: np.asarray(action[key], dtype=np.float32).reshape(12).copy()
            for key in ("default", "scale", "lower", "upper")}
