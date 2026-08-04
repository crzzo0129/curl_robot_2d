from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d_mjx.config_3d import (
    Rolling3DConfig,
    physics_profile_3d,
    validate_3d_config,
)
from curl_robot_2d_mjx.environment_3d import (
    ACTION_SIZE_3D,
    DEFAULT_3D_CEM_CONTROLLER,
    MODEL_PATH_3D,
    OBSERVATION_SIZE_3D,
    advance_rolling_phase_3d,
    apply_physics_options_3d,
    duplicate_planar_action_3d,
    pair_coupled_residual_action_3d,
)
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
from scripts import mjx_3d_smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MJX3DContractTest(unittest.TestCase):
    def test_3d_task_uses_generated_curl_model(self) -> None:
        self.assertEqual(
            MODEL_PATH_3D, PROJECT_ROOT / "assets" / "curl_robot_3d.xml"
        )
        self.assertTrue(MODEL_PATH_3D.exists())
        self.assertEqual(DEFAULT_3D_CEM_CONTROLLER.name, "best_phase_controller.json")
        self.assertTrue(DEFAULT_3D_CEM_CONTROLLER.exists())
        self.assertEqual(len(JOINT_NAMES_3D), ACTION_SIZE_3D)
        self.assertEqual(OBSERVATION_SIZE_3D, 59)

    def test_3d_config_defaults_are_training_smoke_safe(self) -> None:
        config = Rolling3DConfig()

        self.assertAlmostEqual(config.control_timestep, 0.02)
        self.assertEqual(config.episode_length, 500)
        self.assertEqual(len(config.action_scales), 8)
        self.assertEqual(config.terminate_root_z_min, 0.025)
        self.assertEqual(config.terminate_root_z_low_duration_s, 0.20)
        self.assertEqual(config.terminate_lateral_drift_m, 0.20)
        self.assertEqual(config.terminate_axis_tilt_rad, 0.50)
        self.assertEqual(config.terminate_forbidden_depth_m, 0.004)
        self.assertIsNone(config.residual_pair_differential_scale)
        reward = Rolling3DRewardConfig()
        self.assertEqual(reward.roll_progress, 6.0)
        self.assertEqual(reward.cross_side_foot_contact, 30.0)
        self.assertEqual(reward.termination, 20.0)

    def test_3d_config_validation(self) -> None:
        validate_3d_config(Rolling3DConfig())
        invalid = (
            {"action_scales": (1.0,)},
            {"reference_phase_rate_scale": float("nan")},
            {"residual_pair_differential_scale": 1.1},
            {"terminate_root_z_min": -0.01},
            {"terminate_axis_tilt_duration_s": 0.0},
            {"terminate_forbidden_contact_duration_s": -1.0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_3d_config(Rolling3DConfig(**values))

    def test_3d_physics_profiles_keep_control_rate(self) -> None:
        reference = physics_profile_3d("reference")
        newton4 = physics_profile_3d("newton4")
        cg12 = physics_profile_3d("cg12")

        self.assertAlmostEqual(reference.control_timestep, 0.02)
        self.assertAlmostEqual(newton4.control_timestep, 0.02)
        self.assertAlmostEqual(cg12.control_timestep, 0.02)
        self.assertEqual(cg12.solver_name, "cg")
        self.assertLess(cg12.solver_iterations, reference.solver_iterations)

    def test_duplicate_planar_action_maps_front_rear_to_left_right(self) -> None:
        mapped = duplicate_planar_action_3d(
            np, np.asarray((0.1, 0.2, 0.3, 0.4))
        )
        np.testing.assert_allclose(
            mapped,
            np.asarray((0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4)),
        )

    def test_rolling_phase_integrates_signed_local_y_velocity(self) -> None:
        forward = advance_rolling_phase_3d(np, 0.2, 3.0, 0.01)
        backward = advance_rolling_phase_3d(np, 0.2, -3.0, 0.01)

        self.assertAlmostEqual(float(forward), 0.23)
        self.assertAlmostEqual(float(backward), 0.17)

    def test_pair_coupled_residual_preserves_common_and_limits_difference(
        self,
    ) -> None:
        raw_action = np.asarray(
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
        )

        coupled = pair_coupled_residual_action_3d(
            np, raw_action, differential_scale=0.25
        )

        np.testing.assert_allclose(
            coupled,
            np.asarray((0.225, 0.35, -0.025, 0.05, 0.475, 0.6, 0.125, 0.2)),
        )
        np.testing.assert_allclose(
            pair_coupled_residual_action_3d(
                np, raw_action, differential_scale=None
            ),
            raw_action,
        )

    def test_3d_physics_options_can_disable_freejoint_damping(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH_3D))
        task = Rolling3DConfig(disable_root_damping=True)

        apply_physics_options_3d(model, task)

        root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        dof_id = int(model.jnt_dofadr[root_id])
        np.testing.assert_allclose(model.dof_damping[dof_id : dof_id + 6], 0.0)

    def test_3d_smoke_entry_imports_without_jax(self) -> None:
        args = mjx_3d_smoke.parse_args([])

        self.assertTrue(callable(mjx_3d_smoke.main))
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.steps, 1)

    def test_3d_environment_declares_required_metrics_and_guards(self) -> None:
        source = (
            PROJECT_ROOT / "curl_robot_2d_mjx" / "environment_3d.py"
        ).read_text(encoding="utf-8")
        for token in (
            "failure_lateral_drift",
            "failure_axis_tilt",
            "failure_forbidden_depth",
            "failure_forbidden_contact",
            "same_side_foot_contact_start",
            "cross_side_foot_contact_count",
            "axis_tilt_rad",
            "jax.lax.cond",
            "transition_finite",
            "jp.nan_to_num",
            "advance_oscillator",
            'state.info["rolling_phase"]',
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
