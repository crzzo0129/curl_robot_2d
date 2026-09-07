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
    anti_ballistic_costs,
    bank_action_arrays,
    compact_target_from_keyframe,
    confirmation_update,
    dense_pose_reward,
    disable_self_collision_xml,
    gate_errors,
    policy_actuator_names,
    policy_joint_names,
    pose_cost,
    pose_potential,
    prepare_runtime_xml,
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
        self.assertEqual(cfg.confirmation_steps * CONTROL_TIMESTEP_S, 0.10)

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

    def test_pose_only_gate_ignores_velocity_fields(self):
        cfg = WalkCompactConfig()
        self.assertEqual(cfg.joint_position_rad, 0.10)
        # no velocity tolerance fields exist in the config
        self.assertFalse(hasattr(cfg, "root_linear_velocity_m_s"))


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

    def test_exact_target_has_zero_errors_and_unit_quality(self):
        errors = gate_errors(np, self.target["joints"], 0.0, 0.0,
                             self.target, self.cfg)
        self.assertLess(errors.max(), 1e-5)
        quality = pose_potential(np, self.target["joints"], 0.0,
                                 self.target, self.cfg)
        self.assertAlmostEqual(float(quality), 1.0, places=5)

    def test_joint_offset_scales_with_tolerance(self):
        joints = np.asarray(self.target["joints"], dtype=np.float32).copy()
        joints[1] += 0.05  # hip; tolerance 0.10 rad -> 0.5
        errors = gate_errors(np, joints, 0.0, 0.0, self.target, self.cfg)
        self.assertAlmostEqual(float(errors[0]), 0.5, places=5)
        self.assertAlmostEqual(float(errors[1]), 0.0, places=5)

    def test_lateral_scales_with_tolerance(self):
        errors = gate_errors(np, self.target["joints"], 0.0, 0.05,
                             self.target, self.cfg)
        self.assertAlmostEqual(float(errors[2]), 1.0, places=5)

    def test_axis_tilt_scales_with_tolerance(self):
        errors = gate_errors(np, self.target["joints"], 0.0, 0.0,
                             self.target, self.cfg)
        self.assertAlmostEqual(float(errors[1]), 0.0, places=5)
        errors = gate_errors(np, self.target["joints"], 0.10, 0.0,
                             self.target, self.cfg)
        self.assertAlmostEqual(float(errors[1]), 1.0, places=5)

    @staticmethod
    def _axis_tilt_np(q):
        w, x, y, z = q
        # world_z component of body-Y = R[2][1] = 2(yz + wx) for wxyz quaternion
        body_y_z = 2.0 * (y * z + w * x)
        return np.arcsin(np.clip(np.abs(body_y_z), 0.0, 1.0))

    def test_axis_tilt_invariant_to_forward_roll(self):
        import math
        self.assertAlmostEqual(self._axis_tilt_np([1.0, 0.0, 0.0, 0.0]), 0.0,
                               places=6)
        # 90 deg forward roll about body Y: sideways lean stays zero
        q = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        self.assertAlmostEqual(self._axis_tilt_np(q), 0.0, places=6)
        # 30 deg sideways lean about body X: tilt == 30 deg
        q = [math.cos(math.pi / 12), math.sin(math.pi / 12), 0.0, 0.0]
        self.assertAlmostEqual(self._axis_tilt_np(q), math.pi / 6, places=6)

    def test_pose_potential_is_not_flat_from_walking(self):
        # Regression: the potential must keep gradient from a walking pose,
        # not collapse to ~0 (which starves the dense pose reward).
        quality = pose_potential(np, self.walking, 0.0, self.target, self.cfg)
        self.assertGreater(float(quality), 0.05)
        self.assertLess(float(quality), 0.95)

    def test_pose_cost_is_quadratic_with_linear_gradient(self):
        # Regression: the reward signal must grow quadratically (linear
        # gradient) from a walking pose, not exp-flatten.
        at_target = pose_cost(np, self.target["joints"], 0.0, self.target, self.cfg)
        self.assertAlmostEqual(float(at_target), 0.0, places=6)
        walking = pose_cost(np, self.walking, 0.0, self.target, self.cfg)
        self.assertGreater(float(walking), 0.5)

    def test_dense_pose_reward_negative_and_zero_at_target(self):
        cost = pose_cost(np, self.target["joints"], 0.0, self.target, self.cfg)
        self.assertAlmostEqual(float(dense_pose_reward(np, cost, self.cfg)), 0.0,
                               places=6)
        far = pose_cost(np, self.walking, 0.5, self.target, self.cfg)
        self.assertLess(float(dense_pose_reward(np, far, self.cfg)), 0.0)

    def test_confirmation_update_contiguity(self):
        count = 0
        for eligible in (True, True, True, False, True, True, True, True, True):
            count = confirmation_update(np, 0, count, 0, eligible)
        self.assertEqual(count, 5)

    def test_anti_ballistic_costs_shape_and_zero_at_rest(self):
        parts, total = anti_ballistic_costs(
            np, 0.0, 0.158, np.zeros(3),
            stand_z=0.158, compact_z=0.1663, cfg=self.cfg)
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(float(total), 0.0, places=6)
        parts, total = anti_ballistic_costs(
            np, 0.5, 0.25, np.ones(3) * 1.0,
            stand_z=0.158, compact_z=0.1663, cfg=self.cfg)
        self.assertGreater(float(total), 0.0)


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
