from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d_mjx.config_3d import (
    GEOMETRY_NAMES_3D,
    Rolling3DConfig,
    physics_profile_3d,
    validate_3d_config,
)
from curl_robot_2d_mjx.curriculum_3d import curriculum_stages_3d
from curl_robot_2d_mjx.environment_3d import (
    ACTION_SIZE_3D,
    BASELINE_3D_CEM_CONTROLLER,
    DEFAULT_3D_CEM_CONTROLLER,
    MODEL_PATH_3D,
    PUPPER_OPEN60_MODEL_PATH_3D,
    PUPPER_OPEN60_CEM_CONTROLLER,
    ROLLINGQUAD_2_MODEL_PATH_3D,
    REAL_3D_CEM_CONTROLLER,
    REAL_MODEL_PATH_3D,
    OBSERVATION_SIZE_3D,
    PHASE_FEEDBACK_SIZE_3D,
    advance_rolling_phase_3d,
    apply_physics_options_3d,
    axis_tilted_quaternion_3d,
    duplicate_planar_action_3d,
    pair_coupled_residual_action_3d,
    pair_coupled_reset_noise_3d,
    mirror_rolling_observation_3d,
    phase_feedback_observation_3d,
    reference_startup_scale_3d,
    rolling_axis_heading_3d,
    rolling_target_ctrl_3d,
    geometry_parameters_3d,
    cem_controller_path_3d,
    configure_pupper_shell_collisions_3d,
    model_path_3d,
    validate_rolling_morphology_3d,
)
from curl_robot_2d_mjx.randomization_3d import (
    Rolling3DDomainRandomization,
    validate_domain_randomization_3d,
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
        self.assertTrue(REAL_MODEL_PATH_3D.exists())
        self.assertTrue(PUPPER_OPEN60_MODEL_PATH_3D.exists())
        self.assertEqual(
            GEOMETRY_NAMES_3D,
            ("baseline", "real", "pupper_open60", "rollingquad_2"),
        )
        self.assertEqual(model_path_3d("real"), REAL_MODEL_PATH_3D)
        self.assertEqual(
            model_path_3d("pupper_open60"), PUPPER_OPEN60_MODEL_PATH_3D
        )
        self.assertEqual(
            model_path_3d("rollingquad_2"), ROLLINGQUAD_2_MODEL_PATH_3D
        )
        self.assertAlmostEqual(geometry_parameters_3d("real").edge_length, 0.18)
        self.assertAlmostEqual(geometry_parameters_3d("real").foot_radius, 0.03)
        self.assertEqual(
            DEFAULT_3D_CEM_CONTROLLER, PUPPER_OPEN60_CEM_CONTROLLER
        )
        self.assertEqual(
            cem_controller_path_3d("baseline"),
            BASELINE_3D_CEM_CONTROLLER,
        )
        self.assertEqual(
            cem_controller_path_3d("real"), REAL_3D_CEM_CONTROLLER
        )
        self.assertEqual(
            cem_controller_path_3d("pupper_open60"),
            PUPPER_OPEN60_CEM_CONTROLLER,
        )
        self.assertEqual(
            cem_controller_path_3d("rollingquad_2"),
            PUPPER_OPEN60_CEM_CONTROLLER,
        )
        for controller in (
            BASELINE_3D_CEM_CONTROLLER,
            REAL_3D_CEM_CONTROLLER,
            PUPPER_OPEN60_CEM_CONTROLLER,
        ):
            self.assertTrue(controller.exists())
        self.assertEqual(len(JOINT_NAMES_3D), ACTION_SIZE_3D)
        self.assertEqual(OBSERVATION_SIZE_3D, 61)

    def test_3d_config_defaults_are_training_smoke_safe(self) -> None:
        config = Rolling3DConfig()

        self.assertEqual(config.geometry, "rollingquad_2")
        self.assertAlmostEqual(config.control_timestep, 0.02)
        self.assertEqual(config.episode_length, 500)
        self.assertEqual(len(config.action_scales), 8)
        self.assertEqual(config.terminate_root_z_min, 0.025)
        self.assertEqual(config.terminate_root_z_low_duration_s, 0.20)
        self.assertEqual(config.terminate_lateral_drift_m, 0.20)
        self.assertEqual(config.terminate_axis_tilt_rad, 0.50)
        self.assertEqual(config.terminate_forbidden_depth_m, 0.004)
        self.assertEqual(config.reference_action_scale, 1.0)
        self.assertEqual(config.reference_ramp_start_scale, 0.0)
        self.assertEqual(config.reference_ramp_duration_s, 0.25)
        self.assertEqual(config.reference_startup_boost, 0.0)
        self.assertEqual(config.reference_startup_boost_duration_s, 0.25)
        self.assertIsNone(config.residual_pair_differential_scale)
        self.assertIsNone(config.reset_pair_differential_scale)
        self.assertEqual(config.lateral_reflex_gain, 0.0)
        self.assertEqual(config.lateral_reflex_position_gain, 2.0)
        self.assertEqual(config.lateral_reflex_velocity_gain, 2.0)
        self.assertFalse(config.lateral_command_enabled)
        self.assertEqual(config.lateral_command_max, 0.15)
        self.assertEqual(config.lateral_command_probability, 0.20)
        self.assertEqual(config.lateral_command_error_limit, 0.20)
        self.assertIsNone(config.lateral_command_fixed)
        self.assertEqual(config.reset_axis_tilt_noise_rad, 0.0)
        self.assertEqual(config.geom_friction_scale, 1.0)
        self.assertEqual(config.floor_friction_scale, 1.0)
        self.assertFalse(config.floor_contact_friction_override)
        self.assertEqual(config.body_mass_scale, 1.0)
        self.assertEqual(config.body_mass_left_scale, 1.0)
        self.assertEqual(config.body_mass_right_scale, 1.0)
        self.assertFalse(config.explicit_phase_observation)
        reward = Rolling3DRewardConfig()
        self.assertEqual(reward.roll_progress, 6.0)
        self.assertEqual(reward.cross_side_foot_contact, 30.0)
        self.assertEqual(reward.termination, 20.0)

    def test_3d_config_validation(self) -> None:
        validate_3d_config(Rolling3DConfig())
        invalid = (
            {"geometry": "unknown"},
            {"action_scales": (1.0,)},
            {"reference_phase_rate_scale": float("nan")},
            {"reference_action_scale": 0.0},
            {"reference_ramp_start_scale": -0.1},
            {"reference_ramp_duration_s": 0.0},
            {"reference_startup_boost": -0.1},
            {"reference_startup_boost_duration_s": 0.0},
            {"residual_pair_differential_scale": 1.1},
            {"lateral_reflex_gain": -0.1},
            {"lateral_reflex_position_gain": float("nan")},
            {"lateral_reflex_velocity_gain": float("inf")},
            {"lateral_command_probability": 1.1},
            {"lateral_command_error_limit": 0.0},
            {"lateral_command_fixed": float("nan")},
            {"reset_joint_noise_rad": -0.1},
            {"reset_velocity_noise": float("nan")},
            {"reset_root_velocity_noise": -0.1},
            {"reset_pair_differential_scale": 1.1},
            {"reset_axis_tilt_noise_rad": -0.1},
            {"geom_friction_scale": 0.0},
            {"floor_friction_scale": 0.0},
            {"floor_contact_friction_override": 1},
            {"body_mass_scale": 0.0},
            {"body_mass_left_scale": float("nan")},
            {"body_mass_right_scale": -1.0},
            {"explicit_phase_observation": 1},
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
        newton8 = physics_profile_3d("newton8")
        cg12 = physics_profile_3d("cg12")
        cg20 = physics_profile_3d("cg20")

        self.assertAlmostEqual(reference.control_timestep, 0.02)
        self.assertAlmostEqual(newton4.control_timestep, 0.02)
        self.assertAlmostEqual(newton8.control_timestep, 0.02)
        self.assertEqual(newton8.solver_name, "newton")
        self.assertEqual(newton8.solver_iterations, 8)
        self.assertEqual(newton8.solver_ls_iterations, 8)
        self.assertAlmostEqual(cg12.control_timestep, 0.02)
        self.assertEqual(cg12.solver_name, "cg")
        self.assertLess(cg12.solver_iterations, reference.solver_iterations)
        self.assertEqual(cg20.solver_name, "cg")
        self.assertEqual(cg20.solver_iterations, 20)
        self.assertEqual(cg20.solver_ls_iterations, 10)

    def test_reference_startup_scale_boost_decays_to_nominal(self) -> None:
        config = Rolling3DConfig(
            reference_action_scale=1.0,
            reference_ramp_start_scale=None,
            reference_startup_boost=0.25,
            reference_startup_boost_duration_s=0.5,
        )

        self.assertAlmostEqual(reference_startup_scale_3d(np, 0.0, config), 1.25)
        self.assertAlmostEqual(reference_startup_scale_3d(np, 0.5, config), 1.0)

    def test_reference_startup_scale_can_ramp_from_safe_scale(self) -> None:
        config = Rolling3DConfig(
            reference_action_scale=1.0,
            reference_ramp_start_scale=0.25,
            reference_ramp_duration_s=0.5,
        )

        self.assertAlmostEqual(reference_startup_scale_3d(np, 0.0, config), 0.25)
        self.assertAlmostEqual(reference_startup_scale_3d(np, 0.5, config), 1.0)

    def test_duplicate_planar_action_maps_front_rear_to_left_right(self) -> None:
        mapped = duplicate_planar_action_3d(
            np, np.asarray((0.1, 0.2, 0.3, 0.4))
        )
        np.testing.assert_allclose(
            mapped,
            np.asarray((0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4)),
        )

    def test_eight_rolling_targets_leave_abduction_controls_locked(self) -> None:
        compact = np.asarray(
            (0.0, 0.1, 0.9, 0.0, 0.1, 0.9, 0.0, 0.1, 0.9, 0.0, 0.1, 0.9)
        )
        actuator_ids = np.asarray((1, 2, 4, 5, 7, 8, 10, 11))
        action = np.full((8,), 0.25)
        target = rolling_target_ctrl_3d(
            np,
            compact,
            actuator_ids,
            action,
            np.ones((8,)),
            np.full((8,), -2.0),
            np.full((8,), 2.0),
        )

        np.testing.assert_allclose(target[[0, 3, 6, 9]], 0.0)
        np.testing.assert_allclose(
            target[actuator_ids], compact[actuator_ids] + 0.25
        )

    def test_latest_rolling_model_has_valid_twelve_joint_contract(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(PUPPER_OPEN60_MODEL_PATH_3D))

        validate_rolling_morphology_3d(model, "pupper_open60")
        self.assertEqual(model.nu, 12)

        configure_pupper_shell_collisions_3d(model, enabled=True)
        shell_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if "_shell_" in (model.geom(geom_id).name or "")
        ]
        self.assertGreater(len(shell_ids), 0)
        for geom_id in shell_ids:
            self.assertEqual(int(model.geom_contype[geom_id]), 4)
            self.assertEqual(int(model.geom_conaffinity[geom_id]), 3)

    def test_corrected_rollingquad_is_the_default_rolling_model(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROLLINGQUAD_2_MODEL_PATH_3D))

        validate_rolling_morphology_3d(model, "rollingquad_2")
        self.assertEqual((model.nq, model.nv, model.nu), (19, 18, 12))
        self.assertEqual(Rolling3DConfig().geometry, "rollingquad_2")

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

    def test_pair_coupled_reset_noise_uses_symmetric_joint_order(self) -> None:
        noise = pair_coupled_reset_noise_3d(
            np,
            np.asarray((1.0, 2.0, 3.0, 4.0)),
            np.asarray((0.1, 0.2, 0.3, 0.4)),
            differential_scale=0.5,
        )

        np.testing.assert_allclose(
            noise,
            np.asarray((1.05, 2.10, 0.95, 1.90, 3.15, 4.20, 2.85, 3.80)),
        )

    def test_rolling_observation_mirror_is_involutive(self) -> None:
        observation = np.arange(
            OBSERVATION_SIZE_3D + PHASE_FEEDBACK_SIZE_3D,
            dtype=np.float32,
        )

        mirrored = mirror_rolling_observation_3d(np, observation)
        restored = mirror_rolling_observation_3d(np, mirrored)

        np.testing.assert_allclose(restored, observation)
        self.assertEqual(mirrored[1], -observation[1])
        self.assertEqual(mirrored[2], -observation[2])
        np.testing.assert_allclose(mirrored[15:23], observation[
            [17, 18, 15, 16, 21, 22, 19, 20]
        ])
        self.assertEqual(mirrored[60], -observation[60])
        np.testing.assert_allclose(mirrored[61:65], observation[61:65])

    def test_rolling_observation_mirror_supports_base_observation(self) -> None:
        observation = np.arange(OBSERVATION_SIZE_3D, dtype=np.float32)

        mirrored = mirror_rolling_observation_3d(np, observation)

        self.assertEqual(mirrored.shape, observation.shape)
        np.testing.assert_allclose(
            mirror_rolling_observation_3d(np, mirrored), observation
        )

    def test_axis_tilt_quaternion_is_normalized_and_keeps_zero_exact(self) -> None:
        identity = np.asarray((1.0, 0.0, 0.0, 0.0))

        zero = axis_tilted_quaternion_3d(np, identity, 0.0, 0.0)
        tilted = axis_tilted_quaternion_3d(np, identity, 0.1, -0.2)

        np.testing.assert_allclose(zero, identity)
        self.assertAlmostEqual(float(np.linalg.norm(tilted)), 1.0)

    def test_rolling_axis_heading_is_invariant_to_roll_phase(self) -> None:
        yaw = 0.2
        rotation_z = np.asarray(
            (
                (np.cos(yaw), -np.sin(yaw), 0.0),
                (np.sin(yaw), np.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        for roll_phase in np.linspace(0.0, 2.0 * np.pi, 9):
            rotation_y = np.asarray(
                (
                    (np.cos(roll_phase), 0.0, np.sin(roll_phase)),
                    (0.0, 1.0, 0.0),
                    (-np.sin(roll_phase), 0.0, np.cos(roll_phase)),
                )
            )
            body_y_axis = (rotation_z @ rotation_y)[:, 1]
            self.assertAlmostEqual(
                float(rolling_axis_heading_3d(np, body_y_axis)), yaw
            )

    def test_reset_curriculum_precedes_physics_randomization(self) -> None:
        reset_stages = curriculum_stages_3d("reset_v1")
        robust_stages = curriculum_stages_3d("robustness_v1")

        self.assertEqual(reset_stages[0].name, "symmetric_reset")
        self.assertEqual(reset_stages[-1].name, "differential_025")
        self.assertFalse(
            any(stage.domain_randomization.enabled for stage in reset_stages)
        )
        self.assertEqual(robust_stages[-2].name, "friction")
        self.assertEqual(robust_stages[-1].name, "dynamics")
        self.assertTrue(robust_stages[-1].domain_randomization.enabled)

    def test_reset_v2_resolves_measured_axis_tilt_cliff(self) -> None:
        stages = curriculum_stages_3d("reset_v2")

        self.assertEqual(
            [stage.reset_axis_tilt_noise_rad for stage in stages],
            [0.0, 0.010, 0.015, 0.0175, 0.020, 0.030],
        )
        self.assertTrue(
            all(stage.reset_joint_noise_rad == 0.015 for stage in stages)
        )
        self.assertTrue(
            all(stage.reset_velocity_noise == 0.030 for stage in stages)
        )
        self.assertTrue(
            all(
                stage.reset_pair_differential_scale == 0.25
                for stage in stages
            )
        )
        self.assertAlmostEqual(sum(stage.weight for stage in stages), 1.0)
        self.assertGreaterEqual(
            sum(
                stage.weight
                for stage in stages
                if stage.reset_axis_tilt_noise_rad >= 0.015
            ),
            0.80,
        )

    def test_nominal_reset_v3_ends_with_true_independent_joint_noise(
        self,
    ) -> None:
        stages = curriculum_stages_3d("nominal_reset_v3")

        self.assertEqual(
            [stage.reset_pair_differential_scale for stage in stages[:-1]],
            [0.0, 0.0, 0.0, 0.10, 0.25],
        )
        self.assertTrue(stages[-1].reset_independent)
        self.assertAlmostEqual(sum(stage.weight for stage in stages), 1.0)
        base = Rolling3DConfig(reset_pair_differential_scale=0.0)
        final_task = stages[-1].task_config(base)
        self.assertIsNone(final_task.reset_pair_differential_scale)
        self.assertEqual(final_task.reset_joint_noise_rad, 0.005)
        self.assertEqual(final_task.reset_velocity_noise, 0.005)
        self.assertEqual(final_task.reset_root_velocity_noise, 0.0)
        self.assertEqual(final_task.reset_axis_tilt_noise_rad, 0.0)

    def test_independent_reset_v4_preserves_target_noise_structure(self) -> None:
        stages = curriculum_stages_3d("independent_reset_v4")

        self.assertEqual(
            [stage.reset_joint_noise_rad for stage in stages],
            [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005],
        )
        self.assertEqual(
            [stage.reset_velocity_noise for stage in stages],
            [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005],
        )
        self.assertTrue(all(stage.reset_independent for stage in stages))
        self.assertTrue(
            all(stage.reset_root_velocity_noise == 0.0 for stage in stages)
        )
        self.assertTrue(
            all(stage.reset_axis_tilt_noise_rad == 0.0 for stage in stages)
        )
        self.assertAlmostEqual(sum(stage.weight for stage in stages), 1.0)
        base = Rolling3DConfig(
            reset_root_velocity_noise=0.1,
            reset_pair_differential_scale=0.0,
        )
        self.assertTrue(
            all(
                stage.task_config(base).reset_pair_differential_scale is None
                for stage in stages
            )
        )
        self.assertTrue(
            all(
                stage.task_config(base).reset_root_velocity_noise == 0.0
                for stage in stages
            )
        )

    def test_friction_v1_holds_reset_v2_target_and_only_expands_friction(
        self,
    ) -> None:
        stages = curriculum_stages_3d("friction_v1")

        self.assertEqual(
            [stage.name for stage in stages],
            ["friction_02", "friction_05", "friction_10"],
        )
        self.assertEqual(
            [
                stage.domain_randomization.geom_friction_scale
                for stage in stages
            ],
            [(0.98, 1.02), (0.95, 1.05), (0.90, 1.10)],
        )
        self.assertAlmostEqual(sum(stage.weight for stage in stages), 1.0)
        for stage in stages:
            self.assertEqual(
                stage.domain_randomization.floor_friction_scale,
                (1.0, 1.0),
            )
            self.assertEqual(stage.reset_joint_noise_rad, 0.015)
            self.assertEqual(stage.reset_velocity_noise, 0.030)
            self.assertEqual(stage.reset_pair_differential_scale, 0.25)
            self.assertEqual(stage.reset_axis_tilt_noise_rad, 0.030)
            self.assertEqual(
                stage.domain_randomization.body_mass_scale, (1.0, 1.0)
            )
            self.assertEqual(
                stage.domain_randomization.actuator_gain_scale, (1.0, 1.0)
            )

    def test_floor_friction_v2_keeps_accepted_independent_reset_target(
        self,
    ) -> None:
        stages = curriculum_stages_3d("floor_friction_v2")

        self.assertEqual(
            [stage.name for stage in stages],
            ["floor_friction_02", "floor_friction_05", "floor_friction_10"],
        )
        self.assertEqual([stage.weight for stage in stages], [0.20, 0.30, 0.50])
        self.assertEqual(
            [
                stage.domain_randomization.floor_friction_scale
                for stage in stages
            ],
            [(0.98, 1.02), (0.95, 1.05), (0.90, 1.10)],
        )
        base = Rolling3DConfig(
            reset_root_velocity_noise=0.1,
            reset_pair_differential_scale=0.25,
        )
        for stage in stages:
            task = stage.task_config(base)
            self.assertEqual(stage.reset_joint_noise_rad, 0.005)
            self.assertEqual(stage.reset_velocity_noise, 0.005)
            self.assertTrue(stage.reset_independent)
            self.assertEqual(task.reset_root_velocity_noise, 0.0)
            self.assertIsNone(task.reset_pair_differential_scale)
            self.assertEqual(task.reset_axis_tilt_noise_rad, 0.0)
            self.assertTrue(task.floor_contact_friction_override)
            self.assertEqual(
                stage.domain_randomization.geom_friction_scale,
                (1.0, 1.0),
            )
            self.assertEqual(
                stage.domain_randomization.body_mass_scale,
                (1.0, 1.0),
            )
            self.assertEqual(
                stage.domain_randomization.actuator_gain_scale,
                (1.0, 1.0),
            )

    def test_friction_low_v1_expands_low_friction_only(self) -> None:
        stages = curriculum_stages_3d("friction_low_v1")

        self.assertEqual(
            [stage.name for stage in stages],
            ["friction_low_090", "friction_low_080", "friction_low_070"],
        )
        self.assertEqual(
            [
                stage.domain_randomization.geom_friction_scale
                for stage in stages
            ],
            [(0.90, 1.10), (0.80, 1.10), (0.70, 1.10)],
        )
        self.assertAlmostEqual(sum(stage.weight for stage in stages), 1.0)
        for stage in stages:
            self.assertEqual(stage.reset_joint_noise_rad, 0.015)
            self.assertEqual(stage.reset_velocity_noise, 0.030)
            self.assertEqual(stage.reset_pair_differential_scale, 0.25)
            self.assertEqual(stage.reset_axis_tilt_noise_rad, 0.030)

    def test_floor_mass_v2_keeps_floor_friction_and_independent_reset(
        self,
    ) -> None:
        stages = curriculum_stages_3d("floor_mass_v2")

        self.assertEqual(
            [stage.name for stage in stages],
            ["floor_mass_02", "floor_mass_05"],
        )
        self.assertEqual([stage.weight for stage in stages], [0.30, 0.70])
        self.assertEqual(
            [
                stage.domain_randomization.body_mass_scale
                for stage in stages
            ],
            [(0.98, 1.02), (0.95, 1.05)],
        )
        base = Rolling3DConfig(
            reset_root_velocity_noise=0.1,
            reset_pair_differential_scale=0.25,
        )
        for stage in stages:
            task = stage.task_config(base)
            self.assertEqual(stage.reset_joint_noise_rad, 0.005)
            self.assertEqual(stage.reset_velocity_noise, 0.005)
            self.assertTrue(stage.reset_independent)
            self.assertEqual(task.reset_root_velocity_noise, 0.0)
            self.assertIsNone(task.reset_pair_differential_scale)
            self.assertEqual(task.reset_axis_tilt_noise_rad, 0.0)
            self.assertTrue(task.floor_contact_friction_override)
            self.assertEqual(
                stage.domain_randomization.floor_friction_scale,
                (0.90, 1.10),
            )
            self.assertEqual(
                stage.domain_randomization.geom_friction_scale,
                (1.0, 1.0),
            )
            self.assertEqual(
                stage.domain_randomization.actuator_gain_scale,
                (1.0, 1.0),
            )

    def test_mass_v1_retains_friction_and_only_expands_mass_inertia(
        self,
    ) -> None:
        stages = curriculum_stages_3d("mass_v1")

        self.assertEqual(
            [stage.name for stage in stages], ["mass_02", "mass_05"]
        )
        self.assertEqual(
            [stage.weight for stage in stages], [0.30, 0.70]
        )
        self.assertEqual(
            [
                stage.domain_randomization.body_mass_scale
                for stage in stages
            ],
            [(0.98, 1.02), (0.95, 1.05)],
        )
        for stage in stages:
            self.assertEqual(stage.reset_joint_noise_rad, 0.015)
            self.assertEqual(stage.reset_velocity_noise, 0.030)
            self.assertEqual(stage.reset_pair_differential_scale, 0.25)
            self.assertEqual(stage.reset_axis_tilt_noise_rad, 0.030)
            self.assertEqual(
                stage.domain_randomization.geom_friction_scale,
                (0.90, 1.10),
            )
            self.assertEqual(
                stage.domain_randomization.actuator_gain_scale, (1.0, 1.0)
            )

    def test_domain_randomization_ranges_are_positive_and_ordered(self) -> None:
        validate_domain_randomization_3d(
            Rolling3DDomainRandomization(
                geom_friction_scale=(0.9, 1.1),
                floor_friction_scale=(0.9, 1.1),
                body_mass_scale=(0.95, 1.05),
                actuator_gain_scale=(0.95, 1.05),
            )
        )
        with self.assertRaises(ValueError):
            validate_domain_randomization_3d(
                Rolling3DDomainRandomization(
                    geom_friction_scale=(1.1, 0.9)
                )
            )
        with self.assertRaises(ValueError):
            validate_domain_randomization_3d(
                Rolling3DDomainRandomization(
                    floor_friction_scale=(1.1, 0.9)
                )
            )

    def test_phase_feedback_observation_exposes_actual_and_target_error(
        self,
    ) -> None:
        features = phase_feedback_observation_3d(
            np,
            rolling_phase=0.25,
            oscillator_phase=0.75,
        )

        self.assertEqual(features.shape, (PHASE_FEEDBACK_SIZE_3D,))
        np.testing.assert_allclose(
            features,
            np.asarray(
                (
                    np.sin(0.25),
                    np.cos(0.25),
                    np.sin(-0.5),
                    np.cos(-0.5),
                )
            ),
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

    def test_3d_physics_options_can_scale_all_geom_friction(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH_3D))
        original = model.geom_friction.copy()

        apply_physics_options_3d(
            model, Rolling3DConfig(geom_friction_scale=0.90)
        )

        np.testing.assert_allclose(model.geom_friction, original * 0.90)

    def test_3d_physics_options_scale_only_effective_floor_friction(
        self,
    ) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH_3D))
        original_friction = model.geom_friction.copy()
        floor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        apply_physics_options_3d(
            model,
            Rolling3DConfig(
                floor_friction_scale=0.90,
                floor_contact_friction_override=True,
            ),
        )

        nonfloor = np.arange(model.ngeom) != floor_id
        np.testing.assert_allclose(
            model.geom_friction[nonfloor], original_friction[nonfloor]
        )
        np.testing.assert_allclose(
            model.geom_friction[floor_id],
            original_friction[floor_id] * 0.90,
        )
        self.assertGreater(
            model.geom_priority[floor_id],
            np.max(model.geom_priority[nonfloor]),
        )
        np.testing.assert_allclose(
            model.geom_solref[floor_id], np.asarray((0.0065, 1.0))
        )
        np.testing.assert_allclose(
            model.geom_solimp[floor_id],
            np.asarray((0.925, 0.97, 0.001, 0.5, 2.0)),
        )

    def test_3d_physics_options_scale_mass_and_inertia_by_body_side(
        self,
    ) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH_3D))
        original_mass = model.body_mass.copy()
        original_inertia = model.body_inertia.copy()
        torso_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        left_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "front_left_thigh"
        )
        right_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "front_right_thigh"
        )

        apply_physics_options_3d(
            model,
            Rolling3DConfig(
                body_mass_scale=0.95,
                body_mass_left_scale=1.05,
                body_mass_right_scale=0.95,
            ),
        )

        self.assertAlmostEqual(
            model.body_mass[torso_id], original_mass[torso_id] * 0.95
        )
        self.assertAlmostEqual(
            model.body_mass[left_id],
            original_mass[left_id] * 0.95 * 1.05,
        )
        self.assertAlmostEqual(
            model.body_mass[right_id],
            original_mass[right_id] * 0.95 * 0.95,
        )
        np.testing.assert_allclose(
            model.body_inertia[left_id],
            original_inertia[left_id] * 0.95 * 1.05,
        )
        np.testing.assert_allclose(
            model.body_inertia[right_id],
            original_inertia[right_id] * 0.95 * 0.95,
        )

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
            "failure_lateral_positive",
            "failure_lateral_negative",
            "lateral_drift_abs_m",
            "stability_error_cost",
            'state.info["previous_stability_cost"]',
            "residual_common_rms",
            "residual_differential_rms",
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
            "task.reset_root_velocity_noise",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
