"""Contract tests for the walk-start -> compact stage one skill.

These run with numpy only (no MuJoCo/JAX).  MuJoCo-dependent tests are
skipped when mujoco is unavailable.  Run from curl_robot_2d:

    python -m unittest tests.test_walk_compact_3d -v
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.walk_compact_3d import (
    ACTION_SIZE,
    ABD10_SOURCE_XML_REL,
    COMMAND_M_S,
    CONTROL_TIMESTEP_S,
    GEOMETRY,
    HISTORY_SIZE,
    MESH_XML_REL,
    OBSERVATION_SIZE,
    PROJECT_ROOT,
    SINGLE_OBS_SIZE,
    WALK_COMPACT_CONTRACT,
    WalkCompactConfig,
    bank_action_arrays,
    compact_target_from_keyframe,
    confirmation_update,
    disable_self_collision_xml,
    gate_errors,
    height_penalty,
    joint_cost,
    policy_actuator_names,
    policy_joint_names,
    pose_reward,
    prepare_runtime_xml,
    progress_reward,
    roll_pitch_from_quat,
    stability_cost,
    validate_snapshot_bank,
    write_no_self_collision_variant,
    xml_fingerprint,
)
from scripts.collect_walking_start_snapshots import parse_args as collect_parse
from scripts.train_walk_compact_ppo import parse_args as train_parse, main as train_main


def fake_bank(directory, count=64, contract=WALK_COMPACT_CONTRACT):
    rng = np.random.default_rng(7)
    qpos = rng.uniform(-0.2, 0.3, (count, 19)).astype(np.float32)
    qpos[:, 0] = 0.5
    qpos[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
    arrays = {
        "qpos": qpos,
        "qvel": rng.normal(0.0, 0.1, (count, 18)).astype(np.float32),
        "ctrl": rng.uniform(0.0, 1.5, (count, 12)).astype(np.float32),
        "hist": rng.normal(0.0, 0.1, (count, OBSERVATION_SIZE)).astype(np.float32),
        "last_action": rng.uniform(-1.0, 1.0, (count, 12)).astype(np.float32),
        "time": np.linspace(1.5, 6.5, count).astype(np.float32),
    }
    np.savez_compressed(directory / "walk_start_snapshots.npz", **arrays)
    meta = {
        "contract": contract,
        "count": count,
        "command_m_s": COMMAND_M_S,
        "geometry": GEOMETRY,
        "action": {"default": [0.0, 0.9, 1.15] * 4,
                   "scale": [0.17, 0.5, 0.5] * 4,
                   "lower": [-0.5236, -1.7453, 0.1745] * 4,
                   "upper": [3.66519, 1.90241, 2.0944] * 4},
    }
    meta_path = directory / "walk_start_snapshots_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return arrays, meta


class ConfigTest(unittest.TestCase):
    def test_episode_horizon_and_confirmation(self):
        cfg = WalkCompactConfig()
        cfg.validate(CONTROL_TIMESTEP_S)
        self.assertEqual(cfg.episode_steps(CONTROL_TIMESTEP_S), 250)
        self.assertEqual(cfg.budget_s / CONTROL_TIMESTEP_S, 250)
        self.assertEqual(cfg.confirmation_steps * CONTROL_TIMESTEP_S, 0.20)

    def test_validation_rejects_bad_timing_and_confirmation(self):
        bad = WalkCompactConfig(budget_s=5.01)
        with self.assertRaises(ValueError):
            bad.validate(CONTROL_TIMESTEP_S)
        bad = WalkCompactConfig(confirmation_steps=300)
        with self.assertRaises(ValueError):
            bad.validate(CONTROL_TIMESTEP_S)
        bad = WalkCompactConfig(discounting=1.5)
        with self.assertRaises(ValueError):
            bad.validate(CONTROL_TIMESTEP_S)

    def test_success_is_state_space_target(self):
        cfg = WalkCompactConfig()
        self.assertEqual(cfg.joint_position_rad, 0.10)
        # success also gates orientation, base velocity, base angular velocity,
        # and root height -- not just joint pose
        self.assertEqual(cfg.orientation_rad, 0.30)
        self.assertEqual(cfg.base_linear_velocity_m_s, 0.30)
        self.assertEqual(cfg.base_angular_velocity_rad_s, 1.00)
        self.assertEqual(cfg.root_z_min_m, 0.10)

    def test_reward_weights_match_transition_recipe(self):
        cfg = WalkCompactConfig()
        self.assertEqual(cfg.progress_reward_weight, 2.0)
        self.assertEqual(cfg.pose_reward_weight, 0.5)
        self.assertEqual(cfg.success_bonus, 8.0)
        self.assertEqual(cfg.time_cost, 0.003)

    def test_budget_randomization_validation(self):
        cfg = WalkCompactConfig(budget_s=1.5, budget_s_max=3.0)
        cfg.validate(CONTROL_TIMESTEP_S)
        self.assertEqual(cfg.episode_steps(CONTROL_TIMESTEP_S), 75)
        self.assertEqual(cfg.max_episode_steps(CONTROL_TIMESTEP_S), 150)
        with self.assertRaises(ValueError):
            WalkCompactConfig(budget_s=3.0, budget_s_max=2.0).validate(CONTROL_TIMESTEP_S)
        with self.assertRaises(ValueError):
            WalkCompactConfig(budget_s=3.0, budget_s_max=3.01).validate(CONTROL_TIMESTEP_S)


class ContractTest(unittest.TestCase):
    def test_obs_action_contract_sizes(self):
        self.assertEqual(SINGLE_OBS_SIZE * HISTORY_SIZE, OBSERVATION_SIZE)
        self.assertEqual(SINGLE_OBS_SIZE, 36)
        self.assertEqual(HISTORY_SIZE, 20)
        self.assertEqual(ACTION_SIZE, 12)

    def test_policy_order_is_fl_fr_rl_rr_abd_hip_knee(self):
        names = policy_joint_names()
        self.assertEqual(len(names), 12)
        self.assertEqual(names[:3], ("front_left_hip_abduction", "front_left_hip",
                                     "front_left_knee"))
        self.assertEqual(names[3:6], ("front_right_hip_abduction", "front_right_hip",
                                      "front_right_knee"))
        self.assertEqual(names[6:9], ("rear_left_hip_abduction", "rear_left_hip",
                                      "rear_left_knee"))
        self.assertEqual(names[9:], ("rear_right_hip_abduction", "rear_right_hip",
                                     "rear_right_knee"))
        actuators = policy_actuator_names()
        self.assertEqual(actuators[0], "front_left_hip_abduction_servo")
        self.assertEqual(len(actuators), 12)

    def test_runtime_xml_only_touches_option_and_pins_meshdir(self):
        source = PROJECT_ROOT / MESH_XML_REL
        self.assertTrue(source.is_file())
        with tempfile.TemporaryDirectory() as tmp:
            dst = prepare_runtime_xml(source, Path(tmp) / "runtime.xml")
            text = dst.read_text(encoding="utf-8")
            self.assertEqual(text.count("<option"), 1)
            self.assertIn('timestep="0.002"', text)
            self.assertIn('cone="pyramidal"', text)
            # contacts untouched: ground-contact default is kept
            self.assertIn('<geom contype="0" conaffinity="1"', text)
            self.assertNotIn("contype=\"0\" conaffinity=\"0\"", text)
            # mesh resolution pinned to the source MJCF directory
            mesh_dir = source.resolve().parent.as_posix()
            self.assertIn(f'meshdir="{mesh_dir}"', text)
            self.assertIn("../meshes/Upperleg_with_motor_1.stl", text)
            # no rolling self-collision bitmasks may survive in the variant
            for mask in ('contype="16"', 'contype="2"', 'contype="4"',
                         'contype="8"', 'conaffinity="7"', 'conaffinity="29"',
                         'conaffinity="27"', 'conaffinity="15"'):
                self.assertNotIn(mask, text)

    def test_no_self_collision_variant_is_exact_transform_of_source(self):
        source = PROJECT_ROOT / ABD10_SOURCE_XML_REL
        variant = PROJECT_ROOT / MESH_XML_REL
        self.assertTrue(source.is_file())
        self.assertTrue(variant.is_file())
        source_text = source.read_text(encoding="utf-8")
        self.assertIn('contype="16" conaffinity="7"', source_text)  # rolling whitelist present
        self.assertEqual(disable_self_collision_xml(source_text),
                         variant.read_text(encoding="utf-8"))
        # idempotent
        variant_text = variant.read_text(encoding="utf-8")
        self.assertEqual(disable_self_collision_xml(variant_text), variant_text)

    def test_xml_fingerprint_normalizes_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.xml"
            path.write_bytes(b"<mujoco/>\r\n")
            raw = xml_fingerprint(path)
            path.write_bytes(b"<mujoco/>\n")
            lf = xml_fingerprint(path)
            self.assertNotEqual(raw["xml_sha256"], lf["xml_sha256"])
            self.assertEqual(raw["xml_lf_sha256"], lf["xml_lf_sha256"])


class GateMathTest(unittest.TestCase):
    def setUp(self):
        self.cfg = WalkCompactConfig()
        # compact target in policy order FL,FR,RL,RR x abd,hip,knee:
        self.target = {"joints": np.asarray(
            [-0.1745, 0.1108, 0.9093,
             -0.1745, 0.1108, 0.9093,
             0.1745, 0.1108, 0.9093,
             0.1745, 0.1108, 0.9093], dtype=np.float32),
            "root_z": 0.1663}
        # walking default pose (also the transition action nominal)
        self.walking = np.asarray([0.0, 0.9, 1.15] * 4, dtype=np.float32)

    def _gate(self, joints=None, roll=0.0, pitch=0.0, vel=0.0, ang=0.0, root_z=0.14):
        joints = self.target["joints"] if joints is None else joints
        return gate_errors(np, joints, roll, pitch, vel, ang, root_z,
                           self.target, stand_z=0.158, cfg=self.cfg)

    def test_joint_cost_zero_at_target_and_pose_reward_bounded(self):
        self.assertAlmostEqual(float(joint_cost(np, self.target["joints"],
                                                self.target, self.cfg)), 0.0, places=6)
        self.assertAlmostEqual(float(pose_reward(np, 0.0, self.cfg)), 0.5, places=6)
        walking_cost = joint_cost(np, self.walking, self.target, self.cfg)
        self.assertGreater(float(walking_cost), 0.5)
        self.assertLess(float(pose_reward(np, walking_cost, self.cfg)), 0.5)

    def test_progress_reward_rewards_approach(self):
        # closer (D down) -> positive; farther -> negative; unchanged -> zero
        self.assertAlmostEqual(float(progress_reward(np, 1.0, 0.5, self.cfg)), 2.0,
                               places=6)
        self.assertAlmostEqual(float(progress_reward(np, 0.5, 1.0, self.cfg)), -2.0,
                               places=6)
        self.assertAlmostEqual(float(progress_reward(np, 0.5, 0.5, self.cfg)), 0.0,
                               places=6)
        # small progress stays linear (not clipped)
        self.assertAlmostEqual(float(progress_reward(np, 0.02, 0.0, self.cfg)), 2.0,
                               places=6)

    def test_roll_pitch_from_quat(self):
        import math
        roll, pitch = roll_pitch_from_quat(np, np.asarray([1.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(roll), 0.0, places=6)
        self.assertAlmostEqual(float(pitch), 0.0, places=6)
        # pitch forward about body Y
        q = np.asarray([math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0])
        roll, pitch = roll_pitch_from_quat(np, q)
        self.assertAlmostEqual(float(roll), 0.0, places=6)
        self.assertAlmostEqual(float(pitch), math.pi / 4, places=6)
        # roll sideways about body X
        q = np.asarray([math.cos(math.pi / 12), math.sin(math.pi / 12), 0.0, 0.0])
        roll, pitch = roll_pitch_from_quat(np, q)
        self.assertAlmostEqual(float(roll), math.pi / 6, places=6)
        self.assertAlmostEqual(float(pitch), 0.0, places=6)

    def test_stability_cost_zero_at_rest(self):
        self.assertAlmostEqual(float(stability_cost(np, 0.0, 0.0,
                                                    np.zeros(2), self.cfg)), 0.0,
                               places=6)
        self.assertGreater(float(stability_cost(np, 0.3, 0.0,
                                                np.zeros(2), self.cfg)), 0.0)
        self.assertGreater(float(stability_cost(np, 0.0, 0.0,
                                                np.ones(2) * 2.0, self.cfg)), 0.0)

    def test_height_penalty_envelope(self):
        # inside the wide envelope -> zero
        self.assertAlmostEqual(float(height_penalty(
            np, 0.14, stand_z=0.158, compact_z=0.1663, cfg=self.cfg)), 0.0, places=6)
        # collapse below z_min and jump above z_max -> penalised
        self.assertGreater(float(height_penalty(
            np, 0.08, stand_z=0.158, compact_z=0.1663, cfg=self.cfg)), 0.0)
        self.assertGreater(float(height_penalty(
            np, 0.25, stand_z=0.158, compact_z=0.1663, cfg=self.cfg)), 0.0)

    def test_gate_errors_state_space(self):
        errors = self._gate()
        self.assertLess(float(errors.max()), 1e-5)
        # joint offset 0.05 rad -> 0.5 of the 0.10 rad bound
        joints = self.target["joints"].copy()
        joints[1] += 0.05
        self.assertAlmostEqual(float(self._gate(joints=joints)[0]), 0.5, places=5)
        # base velocity 0.3 m/s -> 1.0 of the 0.30 bound
        self.assertAlmostEqual(float(self._gate(vel=0.3)[3]), 1.0, places=5)
        # roll 0.3 rad -> 1.0 of the 0.30 bound
        self.assertAlmostEqual(float(self._gate(roll=0.3)[1]), 1.0, places=5)

    def test_confirmation_update_contiguity(self):
        count = 0
        for eligible in (True, True, True, False, True, True, True, True, True):
            count = confirmation_update(np, 0, count, 0, eligible)
        self.assertEqual(count, 5)


class SnapshotIOTest(unittest.TestCase):
    def test_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays, meta = fake_bank(Path(tmp))
            loaded, loaded_meta = validate_snapshot_bank(
                Path(tmp) / "walk_start_snapshots.npz",
                Path(tmp) / "walk_start_snapshots_meta.json")
            self.assertEqual(loaded["qpos"].shape, (64, 19))
            self.assertEqual(loaded["hist"].shape, (64, OBSERVATION_SIZE))
            self.assertEqual(loaded_meta["count"], 64)
            self.assertEqual(loaded_meta["contract"], WALK_COMPACT_CONTRACT)
            action = bank_action_arrays(loaded_meta)
            self.assertEqual(action["default"].shape, (12,))
            np.testing.assert_allclose(action["default"],
                                       np.asarray([0.0, 0.9, 1.15] * 4))

    def test_validation_rejects_shape_and_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays, meta = fake_bank(Path(tmp))
            bad = Path(tmp) / "bad.npz"
            np.savez_compressed(bad, qpos=arrays["qpos"][:, :10],
                                qvel=arrays["qvel"], ctrl=arrays["ctrl"],
                                hist=arrays["hist"], last_action=arrays["last_action"],
                                time=arrays["time"])
            with self.assertRaises(ValueError):
                validate_snapshot_bank(bad, Path(tmp) / "walk_start_snapshots_meta.json")
            other = Path(tmp) / "other.json"
            other.write_text(json.dumps({**meta, "contract": "other_contract"}),
                             encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_snapshot_bank(Path(tmp) / "walk_start_snapshots.npz", other)

    def test_validation_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays, meta = fake_bank(Path(tmp))
            bad = Path(tmp) / "bad.npz"
            qpos = arrays["qpos"].copy()
            qpos[0, 5] = np.nan
            np.savez_compressed(bad, qpos=qpos, qvel=arrays["qvel"],
                                ctrl=arrays["ctrl"], hist=arrays["hist"],
                                last_action=arrays["last_action"], time=arrays["time"])
            with self.assertRaises(ValueError):
                validate_snapshot_bank(bad, Path(tmp) / "walk_start_snapshots_meta.json")


class ArgParseTest(unittest.TestCase):
    def test_collector_defaults_and_required_out(self):
        args = collect_parse(["--out", "x"])
        self.assertEqual(args.speed, 0.4)
        self.assertEqual(args.warmup_s, 1.5)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collect_parse([])

    def test_train_requires_snapshot_directory(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_parse(["--out", "x", "--snapshots", "does_not_exist_xyz"])

    def test_train_snapshots_defaults_to_project_root(self):
        args = train_parse(["--out", "x"])
        self.assertEqual(args.snapshots,
                         PROJECT_ROOT / "results" / "walk_start_snapshots_0p4")

    def test_train_relative_paths_resolve_against_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bank(Path(tmp))
            args = train_parse(["--snapshots", "results/walk_start_snapshots_0p4",
                                "--out", "results/some_out", "--dry-run"])
            self.assertEqual(args.snapshots,
                             PROJECT_ROOT / "results" / "walk_start_snapshots_0p4")
            self.assertEqual(args.out, PROJECT_ROOT / "results" / "some_out")

    def test_train_dry_run_end_to_end_without_mujoco(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bank(Path(tmp))
            argv = ["--snapshots", str(Path(tmp)), "--out",
                    str(Path(tmp) / "run"), "--dry-run"]
            args = train_parse(argv)
            payload = train_main(argv)
            self.assertEqual(payload["contract"], WALK_COMPACT_CONTRACT)
            self.assertEqual(payload["snapshot_count"], 64)
            self.assertEqual(payload["observation_size"], OBSERVATION_SIZE)
            self.assertEqual(payload["action_size"], 12)
            self.assertEqual(payload["startup"]["budget_s"], 5.0)
            self.assertFalse(payload["rolling_teacher"])


try:
    import mujoco  # noqa: F401
    HAVE_MUJOCO = True
except ImportError:
    HAVE_MUJOCO = False


@unittest.skipUnless(HAVE_MUJOCO, "requires mujoco")
class MujocoContractTest(unittest.TestCase):
    def test_mesh_abd10_compact_keyframe_is_minus10_plus10(self):
        import mujoco
        xml = PROJECT_ROOT / MESH_XML_REL
        model = mujoco.MjModel.from_xml_path(str(xml))
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "compact")
        self.assertGreaterEqual(key_id, 0)
        ctrl = np.asarray(model.key_ctrl[key_id])
        # actuator order = policy order: FL,FR,RL,RR x abd,hip,knee
        deg = np.degrees(ctrl).reshape(4, 3)
        self.assertAlmostEqual(deg[0, 0], -10.0, places=3)  # front_left abd
        self.assertAlmostEqual(deg[1, 0], -10.0, places=3)  # front_right abd
        self.assertAlmostEqual(deg[2, 0], 10.0, places=3)   # rear_left abd
        self.assertAlmostEqual(deg[3, 0], 10.0, places=3)   # rear_right abd

    def test_policy_order_actuator_names_match_model(self):
        import mujoco
        xml = PROJECT_ROOT / MESH_XML_REL
        model = mujoco.MjModel.from_xml_path(str(xml))
        actual = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                       for i in range(model.nu))
        self.assertEqual(actual, policy_actuator_names())

    def test_compact_target_from_keyframe_joint_selection(self):
        import mujoco
        xml = PROJECT_ROOT / MESH_XML_REL
        model = mujoco.MjModel.from_xml_path(str(xml))
        names = policy_joint_names()
        indices = np.asarray([model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names])
        key = np.asarray(model.key_qpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "compact")])
        target = compact_target_from_keyframe(key, indices)
        deg = np.degrees(target["joints"]).reshape(4, 3)
        self.assertAlmostEqual(deg[0, 0], -10.0, places=3)
        self.assertAlmostEqual(target["root_z"], 0.1663, places=3)

    def test_compact_ctrl_is_reachable_in_transition_action_space(self):
        # Regression: the walking action scales (0.17/0.5/0.5) could not reach
        # the compact hip target; the transition asymmetric scale must.
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / MESH_XML_REL))
        nominal = np.asarray([0.0, 0.9, 1.15] * 4)  # walking default, policy order
        low = np.asarray(model.actuator_ctrlrange[:, 0])
        high = np.asarray(model.actuator_ctrlrange[:, 1])
        scale = np.maximum(high - nominal, nominal - low)
        compact = np.asarray(model.key_ctrl[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "compact")])
        action = (compact - nominal) / scale
        self.assertTrue((np.abs(action) <= 1.0 + 1e-4).all(),
                        msg=f"unreachable action {action}")
        # old walking fixed scale is provably insufficient (documents the bug)
        old_scale = np.asarray([0.17, 0.5, 0.5] * 4)
        self.assertTrue((np.abs((compact - nominal) / old_scale) > 1.0).any())

    def test_variant_has_ground_only_geoms_no_self_collision(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / MESH_XML_REL))
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "(anon)"
            contype = int(model.geom_contype[i])
            conaffinity = int(model.geom_conaffinity[i])
            if name == "floor":
                self.assertEqual((contype, conaffinity), (1, 0))
            else:
                # every robot geom: ground-only, no robot-robot collision
                self.assertEqual((contype, conaffinity), (0, 1), msg=name)


if __name__ == "__main__":
    unittest.main()
