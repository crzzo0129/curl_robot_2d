from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.autonomous_startup_3d import (
    AutonomousStartupConfig, candidate_potential, confirmation_update,
    continuation_score, gate_errors, load_candidate_bank,
    model_fingerprint, sha256, validate_model_fingerprint,
)
from scripts.train_mjx_3d_startup_ppo import DEFAULT_BANK, parse_args


def fixture():
    payload = json.loads(DEFAULT_BANK.read_text(encoding="utf-8"))
    bank = {key: np.asarray([c[key] for c in payload["candidates"]], dtype=np.float32)
            for key in ("qpos", "qvel", "ctrl", "time", "rolling_phase", "oscillator_phase")}
    return bank, payload


class AutonomousStartupContractTest(unittest.TestCase):
    def test_config_budget_confirmation_and_grid(self):
        cfg = AutonomousStartupConfig()
        cfg.validate(.02)
        self.assertEqual(cfg.episode_steps(.02), 300)
        for changes in ({"startup_budget_s": 3.1}, {"continuation_s": 0},
                        {"startup_budget_s": .251}, {"confirmation_steps": 0},
                        {"discounting": 1}, {"gate_scale": float("nan")}):
            with self.assertRaises(ValueError):
                replace(cfg, **changes).validate(.02)

    def test_gate_matches_intact_candidate_and_ignores_x_translation(self):
        bank, _ = fixture()
        q, v, phase = bank["qpos"][0].copy(), bank["qvel"][0], bank["rolling_phase"][0]
        error = gate_errors(np, q, v, phase, bank, AutonomousStartupConfig())[0]
        self.assertLess(float(error.max()), .01)
        q[0] += 10
        moved = gate_errors(np, q, v, phase, bank, AutonomousStartupConfig())[0]
        np.testing.assert_allclose(error, moved)
        q[2] += .2
        self.assertGreater(gate_errors(np, q, v, phase, bank, AutonomousStartupConfig())[0].max(), 1)

    def test_quaternion_sign_equivalence_and_phase_wrap(self):
        bank, _ = fixture()
        q = bank["qpos"][0].copy()
        q[3:7] *= -1
        error = gate_errors(np, q, bank["qvel"][0], bank["rolling_phase"][0] + 2 * np.pi,
                            bank, AutonomousStartupConfig())[0]
        self.assertLess(float(error.max()), .01)

    def test_velocity_mismatch_cannot_pass_as_pose_only(self):
        bank, _ = fixture()
        v = bank["qvel"][0].copy()
        v[4] = 0
        error = gate_errors(np, bank["qpos"][0], v, bank["rolling_phase"][0],
                            bank, AutonomousStartupConfig())[0]
        self.assertGreater(error.max(), 1)

    def test_shaping_is_bounded_and_not_flat_far_from_target(self):
        values = candidate_potential(np, np.asarray([[0.] * 7, [2.] * 7, [20.] * 7]))
        self.assertTrue(np.all(values > 0))
        self.assertEqual(values[0], 1)
        self.assertGreater(values[1], values[2])

    def test_confirmation_requires_same_candidate_and_consecutive_frames(self):
        self.assertEqual(confirmation_update(np, 0, 1, 0, True), 2)
        self.assertEqual(confirmation_update(np, 0, 2, 1, True), 1)
        self.assertEqual(confirmation_update(np, 0, 2, 0, False), 0)

    def test_turns_only_after_handoff_and_forward_signed_rotation(self):
        turns, signed = continuation_score(np, x=4, start_x=4, rotation=100,
            start_rotation=100, phase=50, start_phase=50, radius=.1275)
        self.assertEqual(turns, 0)
        self.assertEqual(signed, 0)
        turns, signed = continuation_score(np, x=5, start_x=4, rotation=110,
            start_rotation=100, phase=49, start_phase=50, radius=.1275)
        self.assertGreater(turns, 0)
        self.assertLess(signed, 0)

    def test_teacher_hash_mismatch_is_rejected(self):
        _, payload = fixture()
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory) / "params"
            teacher.write_bytes(b"not the accepted teacher")
            with self.assertRaisesRegex(ValueError, "different teacher"):
                load_candidate_bank(DEFAULT_BANK, teacher_path=teacher,
                    teacher_payload=payload["teacher_config_payload"], model_path=teacher)

    def test_model_hash_normalizes_only_crlf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            lf = b'<mujoco>\n  <option timestep="0.001"/>\n</mujoco>\n'
            path.write_bytes(lf)
            expected = model_fingerprint(path)
            for raw in (lf, lf.replace(b"\n", b"\r\n"), lf.replace(b"\n", b"\r\n", 1)):
                path.write_bytes(raw)
                validate_model_fingerprint(expected, path)
                self.assertEqual(model_fingerprint(path)["model_lf_sha256"], expected["model_lf_sha256"])
            for raw in (lf.replace(b"0.001", b"0.002"), lf.replace(b"  <", b" <")):
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "CRLF/LF differences are already ignored"):
                    validate_model_fingerprint(expected, path)

    def test_legacy_model_metadata_stays_strict_and_reports_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            path.write_bytes(b"<mujoco/>\r\n")
            expected = {"model_sha256": sha256(path)}
            validate_model_fingerprint(expected, path)
            path.write_bytes(b"<mujoco/>\n")
            with self.assertRaises(ValueError) as caught:
                validate_model_fingerprint(expected, path)
            message = str(caught.exception)
            for value in (str(path.resolve()), expected["model_sha256"], sha256(path), "Legacy metadata"):
                self.assertIn(value, message)

    def test_shipped_bank_loads_same_model_with_lf_crlf_or_mixed_endings(self):
        from curl_robot_2d_mjx.environment_3d import model_path_3d
        _, payload = fixture()
        original = model_path_3d(payload["teacher_config_payload"]["task"]["geometry"]).read_bytes()
        lf = original.replace(b"\r\n", b"\n")
        with tempfile.TemporaryDirectory() as directory:
            teacher, model, bank_path = (Path(directory) / name for name in ("params", "model.xml", "bank.json"))
            teacher.write_bytes(b"test checkpoint")
            payload["teacher_sha256"] = sha256(teacher)
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            # Keep shipped model hashes intact: only substitute the test teacher.
            for raw in (original, lf, lf.replace(b"\n", b"\r\n"), lf.replace(b"\n", b"\r\n", 5)):
                model.write_bytes(raw)
                bank, _ = load_candidate_bank(bank_path, teacher_path=teacher,
                    teacher_payload=payload["teacher_config_payload"], model_path=model)
                self.assertEqual(bank["qpos"].shape, (3, 19))

    def test_model_physics_changes_rejected_even_if_raw_hash_matches(self):
        original = (b'<mujoco>\n<body mass="1"><joint axis="0 1 0" range="-1 1"/>'
                    b'<position kp="5" forcerange="-3 3"/></body>\n</mujoco>\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            path.write_bytes(original)
            expected = model_fingerprint(path)
            for before, after in ((b'mass="1"', b'mass="2"'), (b'axis="0 1 0"', b'axis="1 0 0"'),
                                  (b'range="-1 1"', b'range="-2 2"'), (b'kp="5"', b'kp="10"'),
                                  (b'forcerange="-3 3"', b'forcerange="-6 6"')):
                with self.subTest(change=before):
                    path.write_bytes(original.replace(before, after))
                    with self.assertRaisesRegex(ValueError, "MJCF does not match"):
                        validate_model_fingerprint(expected, path)
                    # A stale/incorrect portable digest cannot fall back to raw.
                    with self.assertRaisesRegex(ValueError, "MJCF does not match"):
                        validate_model_fingerprint({**expected, "model_sha256": sha256(path)}, path)

    def test_export_preserves_portable_provenance_and_rejects_different_models(self):
        from scripts.export_startup_handoff_bank import merge_model_provenance
        legacy = {"model_sha256": "raw-a"}
        portable = {**legacy, "model_lf_sha256": "lf-a"}
        result = dict(legacy)
        merge_model_provenance(result, portable)
        self.assertEqual(result, portable)
        merge_model_provenance(result, {"model_sha256": "raw-b", "model_lf_sha256": "lf-a"})
        for incompatible in ({"model_sha256": "raw-b"},
                             {"model_sha256": "raw-a", "model_lf_sha256": "lf-b"}):
            with self.assertRaisesRegex(ValueError, "different models"):
                merge_model_provenance(result, incompatible)

    def test_cli_requires_real_files_and_new_output(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--teacher", "not-a-checkpoint", "--out", "unused"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "params"
            checkpoint.touch()
            config = Path(directory) / "training_config.json"
            config.write_text("{}", encoding="utf-8")
            args = parse_args(["--teacher", str(checkpoint), "--out", str(Path(directory)/"out")])
            self.assertEqual(args.max_devices, 1)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--teacher", str(checkpoint), "--out", directory])


@unittest.skipUnless(os.environ.get("RUN_STARTUP_MJX_TESTS") == "1", "opt-in MJX integration")
class AutonomousStartupMJXTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jp
        from curl_robot_2d_mjx.config_3d import Rolling3DConfig
        from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig
        from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
        from curl_robot_2d_mjx.environment_autonomous_startup_3d import make_autonomous_startup_env
        bank, payload = fixture()
        teacher = payload["teacher_config_payload"]
        cls.jax, cls.jp = jax, jp
        # Zero residual is deliberately a unit-test double, NOT a teacher probe.
        cls.env = make_autonomous_startup_env(Rolling3DConfig(**teacher["task"]),
            CEMReferenceConfig(**teacher["reference"]), Rolling3DRewardConfig(**teacher["reward"]),
            bank, None, teacher, teacher_policy=lambda obs, key: (jp.zeros(8), {}))
        cls.reset = staticmethod(jax.jit(cls.env.reset))
        cls.step = staticmethod(jax.jit(cls.env.step))

    def test_reset_stand_and_full_independent_direct_targets(self):
        s = self.reset(self.jax.random.PRNGKey(0))
        self.assertEqual(s.obs.shape, (self.env.observation_size,))
        np.testing.assert_allclose(s.pipeline_state.ctrl, self.env.mj_model.key("stand").ctrl, atol=1e-6)
        self.assertEqual(int(s.info["startup_steps"]), 0)
        action = self.jp.asarray([.3, -.2, -.3, .2, .4, -.4, -.4, .4])
        n = self.step(s, action)
        base = self.env.base
        expected = np.clip(np.asarray(base.compact_ctrl[base.actuator_ids])
                           + np.asarray(action) * np.asarray(base.action_scales),
                           base.joint_low, base.joint_high)
        np.testing.assert_allclose(n.pipeline_state.ctrl[base.actuator_ids], expected, atol=1e-6)
        self.assertFalse(bool(n.info["teacher_active"]))
        self.assertAlmostEqual(float(n.pipeline_state.time), .02, places=6)

    def test_preparing_handoff_preserves_all_physics_and_actual_history(self):
        s = self.step(self.reset(self.jax.random.PRNGKey(1)), self.env.stand_action)
        b = self.env._unpack(s)
        prepared = self.jax.jit(self.env.prepare_teacher_context)(b, 0)
        for left, right in zip(self.jax.tree_util.tree_leaves(b.pipeline_state),
                               self.jax.tree_util.tree_leaves(prepared.pipeline_state)):
            np.testing.assert_array_equal(left, right)
        for name in ("last_action", "initial_root_x", "initial_root_y", "rolling_phase",
                     "cumulative_rotation", "step_count", "root_low_step_count", "forbidden_contact_step_count"):
            np.testing.assert_array_equal(prepared.info[name], b.info[name])
        self.assertFalse(bool(prepared.info["direct_action_override"]))
        self.assertAlmostEqual(float(prepared.pipeline_state.time + prepared.info["reference_time_offset"]),
                               float(self.env.bank["time"][0]), places=6)

    def test_terminal_pulse_and_full_state_autoreset(self):
        from curl_robot_2d_mjx.environment_autonomous_startup_3d import wrap_autonomous_startup
        wrapped = wrap_autonomous_startup(self.env, self.env.episode_length)
        reset = self.jax.jit(wrapped.reset)
        step = self.jax.jit(wrapped.step)
        s = reset(self.jax.random.split(self.jax.random.PRNGKey(2), 1))
        # Force budget exhaustion; this is a bookkeeping test, not a rollout result.
        info = dict(s.info)
        info["startup_steps"] = self.jp.asarray([149], dtype=self.jp.int32)
        s = s.replace(info=info)
        n = step(s, self.env.stand_action[None])
        self.assertEqual(float(n.done[0]), 1)
        self.assertEqual(float(n.metrics["startup_timeout"][0]), 1)
        self.assertEqual(int(n.info["startup_steps"][0]), 0)
        self.assertFalse(bool(n.info["terminal"][0]))
        self.assertFalse(bool(n.info["teacher_active"][0]))
        self.assertEqual(float(n.pipeline_state.time[0]), 0)
        after = step(n, self.env.stand_action[None])
        self.assertEqual(float(after.done[0]), 0)
        self.assertEqual(int(after.info["startup_steps"][0]), 1)
        self.assertEqual(int(after.info["base_info"]["step_count"][0]), 1)

    def test_frozen_teacher_tail_ignores_startup_actions(self):
        s = self.reset(self.jax.random.PRNGKey(3))
        b = self.env.prepare_teacher_context(self.env._unpack(s), 0)
        info = self.env._pack_info({**s.info, "teacher_active": self.jp.asarray(True)}, b)
        s = s.replace(info=info)
        a = self.step(s, self.jp.ones(8))
        b = self.step(s, -self.jp.ones(8))
        np.testing.assert_array_equal(a.pipeline_state.qpos, b.pipeline_state.qpos)
        np.testing.assert_array_equal(a.pipeline_state.ctrl, b.pipeline_state.ctrl)

    def test_gate_transition_and_tail_outcome_are_real_state_continuations(self):
        from mujoco import mjx
        jp, env = self.jp, self.env
        s = self.reset(self.jax.random.PRNGKey(4))
        b = env._unpack(s)
        # Inject a fixture ONLY inside this unit test; the training reset is stand.
        d = mjx.forward(env.base.mjx_model, b.pipeline_state.replace(
            qpos=env.bank["qpos"][0], qvel=env.bank["qvel"][0], ctrl=env.bank["ctrl"][0], time=jp.asarray(1.)))
        effective = (d.ctrl[env.base.actuator_ids] - env.base.compact_ctrl[env.base.actuator_ids]) / env.base.action_scales
        bi = {**b.info, "rolling_phase": env.bank["rolling_phase"][0],
              "oscillator_phase": env.bank["oscillator_phase"][0], "last_action": effective}
        b = b.replace(pipeline_state=d, info=bi)
        s = s.replace(pipeline_state=d, info=env._pack_info(s.info, b))
        probe = self.step(s, effective)
        self.assertEqual(float(probe.metrics["gate_eligible"]), 1,
                         f"gate_error={probe.metrics['gate_error']}")
        primed = s.replace(info={**s.info, "confirmation": jp.asarray(2, jp.int32),
                                "candidate_id": probe.info["candidate_id"]})
        handed = self.step(primed, effective)
        self.assertEqual(float(handed.metrics["handoff"]), 1)
        self.assertTrue(bool(handed.info["teacher_active"]))
        for left, right in zip(self.jax.tree_util.tree_leaves(probe.pipeline_state),
                               self.jax.tree_util.tree_leaves(handed.pipeline_state)):
            np.testing.assert_array_equal(left, right)
        # Synthetic accounting fixture verifies the one-shot terminal reward.
        info = {**handed.info, "tail_steps": jp.asarray(149, jp.int32),
                "handoff_x": handed.pipeline_state.qpos[0] - 4 * jp.pi * env.base.rolling_radius,
                "handoff_rotation": handed.info["base_info"]["cumulative_rotation"] - 4 * jp.pi,
                "handoff_phase": handed.info["base_info"]["rolling_phase"] - 4 * jp.pi,
                "tail_turns": jp.asarray(2.)}
        terminal = self.step(handed.replace(info=info), jp.zeros(8))
        self.assertEqual(float(terminal.metrics["startup_success"]), 1)
        self.assertEqual(float(terminal.done), 1)
        repeated = self.step(terminal, jp.zeros(8))
        self.assertEqual(float(repeated.metrics["startup_success"]), 0)
        self.assertEqual(float(repeated.reward), 0)


if __name__ == "__main__":
    unittest.main()
