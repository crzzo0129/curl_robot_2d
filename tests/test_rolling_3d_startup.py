from dataclasses import replace
import importlib.util
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, smoothstep_ramp, validate_3d_config
from curl_robot_2d_mjx.environment_3d import (
    ROLLINGQUAD_2_MODEL_PATH_3D, reference_startup_scale_3d,
)
from curl_robot_2d_mjx.startup_rolling_3d import (
    compose_startup_action_3d, reset_pose_arrays_3d, residual_elapsed_3d,
    rolling_elapsed_3d, stand_startup_action_3d, with_stand_startup,
)
from scripts import evaluate_mjx_3d_policy, train_mjx_3d_residual_ppo, train_mjx_3d_roll_distillation


class RollingStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROLLINGQUAD_2_MODEL_PATH_3D))
        cls.task = Rolling3DConfig(reset_pose="stand", episode_length=560)
        cls.reset_qpos, cls.stand_action = reset_pose_arrays_3d(cls.model, cls.task)

    def test_opt_in_keeps_legacy_reset_and_action_origin(self):
        task = Rolling3DConfig()
        self.assertEqual(task.reset_pose, "compact")
        self.assertEqual(task.rolling_start_time_s, 0)
        qpos, action = reset_pose_arrays_3d(self.model, task)
        np.testing.assert_array_equal(qpos, self.model.key("compact").qpos)
        np.testing.assert_array_equal(action, np.zeros(8))
        for time in (0, 0.1, 2.0):
            self.assertEqual(rolling_elapsed_3d(np, time, task), time)
            self.assertEqual(residual_elapsed_3d(np, time, task), time)
            np.testing.assert_array_equal(stand_startup_action_3d(np, time, action, task), 0)

    def test_stand_mapped_by_name_not_body_order(self):
        ids = [self.model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D]
        addresses = [self.model.joint(name).qposadr[0] for name in JOINT_NAMES_3D]
        expected = self.model.key("compact").ctrl[ids] + self.stand_action * self.task.action_scales
        np.testing.assert_allclose(expected, self.reset_qpos[addresses], atol=1e-9)
        np.testing.assert_array_equal(self.reset_qpos, self.model.key("stand").qpos)
        np.testing.assert_allclose(self.stand_action, (0.986464618625, 0.2006177511666667) * 4)

    def test_hold_fold_then_reference_ramp(self):
        end = self.task.rolling_start_time_s
        for time in (0, self.task.stand_hold_s):
            np.testing.assert_allclose(
                stand_startup_action_3d(np, time, self.stand_action, self.task), self.stand_action)
        np.testing.assert_allclose(stand_startup_action_3d(
            np, self.task.stand_hold_s + 0.5, self.stand_action, self.task), 0.5 * self.stand_action)
        np.testing.assert_array_equal(stand_startup_action_3d(np, end, self.stand_action, self.task), 0)
        for time in (0, 0.5, end):
            self.assertEqual(reference_startup_scale_3d(
                np, rolling_elapsed_3d(np, time, self.task), self.task), 0)
        self.assertAlmostEqual(reference_startup_scale_3d(
            np, rolling_elapsed_3d(np, end + 0.25, self.task), self.task), 1)

    def test_residual_can_correct_fold_and_does_not_restart_at_handoff(self):
        for time, expected in ((0, 0), (0.2, 0), (0.45, 1), (1.199999, 1), (1.200001, 1)):
            self.assertAlmostEqual(smoothstep_ramp(
                np, residual_elapsed_3d(np, time, self.task), self.task.startup_action_ramp_s), expected)

    def test_reference_weight_does_not_scale_stand_and_direct_is_unassisted(self):
        for weight in (0, 0.5, 1):
            result = compose_startup_action_3d(np, self.stand_action, self.stand_action,
                np.zeros(8), reference_weight=weight, residual_gain=0.15, ramp=0, direct=False)
            np.testing.assert_allclose(result, self.stand_action)
        action = np.linspace(-1.5, 1.5, 8)
        for ramp in (0, 0.5, 1):
            result = compose_startup_action_3d(np, self.stand_action, self.stand_action,
                action, reference_weight=1, residual_gain=0.15, ramp=ramp, direct=True)
            np.testing.assert_array_equal(result, np.clip(action, -1, 1))

    def test_startup_and_reference_are_continuous_at_boundaries(self):
        def action(time):
            return (stand_startup_action_3d(np, time, self.stand_action, self.task)
                    + reference_startup_scale_3d(np, rolling_elapsed_3d(np, time, self.task), self.task)
                    * np.ones(8) * 0.5)
        for time in (self.task.stand_hold_s, self.task.rolling_start_time_s):
            np.testing.assert_allclose(action(time - 1e-7), action(time + 1e-7), atol=1e-10)

    def test_invalid_startup_rejected(self):
        for changes in (
            {"reset_pose": "park"}, {"stand_hold_s": -1}, {"stand_hold_s": float("nan")},
            {"stand_to_compact_s": 0}, {"stand_to_compact_s": float("inf")},
            {"reference_ramp_start_scale": 1}, {"reference_ramp_start_scale": None},
            {"episode_length": 50}, {"startup_action_ramp_s": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_3d_config(replace(self.task, **changes))

    def test_ppo_and_evaluation_accept_same_startup_flags(self):
        flags = ["--reset-pose", "stand", "--stand-hold-s", "0.1", "--stand-to-compact-s", "0.9"]
        train_args = train_mjx_3d_residual_ppo.parse_args(flags)
        eval_args = evaluate_mjx_3d_policy.parse_args([
            "--evaluation-mode", "reference", "--out", "unused-startup-test", *flags])
        self.assertEqual(with_stand_startup(Rolling3DConfig(), train_args),
                         with_stand_startup(Rolling3DConfig(), eval_args))

    def test_distillation_propagates_stand_to_teacher_and_direct_task(self):
        # Parsing checks existence, not checkpoint contents. No network loaded here.
        args = train_mjx_3d_roll_distillation.parse_args([
            str(ROLLINGQUAD_2_MODEL_PATH_3D), "--out", "unused-startup-test",
            "--reset-pose", "stand", "--episode-length", "560"])
        for direct in (False, True):
            task = with_stand_startup(train_mjx_3d_roll_distillation._task(
                episode_length=args.episode_length, direct_effective_action=direct), args)
            self.assertEqual(task.reset_pose, "stand")
            self.assertEqual(task.direct_effective_action, direct)
            self.assertTrue(task.explicit_phase_observation)
            np.testing.assert_array_equal(reset_pose_arrays_3d(self.model, task)[0], self.reset_qpos)

    def test_unrepresentable_stand_is_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError, "action range"):
            reset_pose_arrays_3d(self.model, replace(self.task, action_scales=(0.1,) * 8))

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("jax", "brax")),
                         "MJX integration requires JAX and Brax")
    def test_mjx_reset_is_stand_and_first_step_holds_it(self):
        import jax
        import jax.numpy as jp
        from curl_robot_2d_mjx.environment_3d import make_brax_env_3d
        env = make_brax_env_3d(replace(self.task, reset_joint_noise_rad=0, reset_velocity_noise=0))
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        np.testing.assert_allclose(np.asarray(state.pipeline_state.qpos), self.reset_qpos, atol=1e-6)
        result = jax.jit(env.step)(state, jp.zeros(8))
        np.testing.assert_allclose(np.asarray(result.pipeline_state.ctrl), self.model.key("stand").ctrl,
                                   atol=1e-6)
        self.assertEqual(float(result.info["oscillator_phase"]), 0)
        self.assertGreater(float(result.pipeline_state.time), 0)
        self.assertEqual(float(result.metrics["stand_startup_active"]), 1)


if __name__ == "__main__":
    unittest.main()
