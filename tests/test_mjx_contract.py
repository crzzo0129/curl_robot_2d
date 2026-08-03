from pathlib import Path
import unittest

import numpy as np

from curl_robot_2d_mjx.config import (
    NominalRLConfig,
    physics_profile,
    smoothstep_ramp,
    validate_nominal_rl_config,
)
from curl_robot_2d_mjx.environment import (
    JOINT_NAMES,
    MODEL_PATH,
    resolve_model_path,
)
from curl_robot_2d_mjx.reward import REWARD_TERM_NAMES
from curl_robot_2d_mjx.reward_config import RollingRewardConfig
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    select_mujoco_gl_backend,
)
from scripts import (
    compare_mjx_cem_reference,
    mjx_smoke,
    train_mjx_ppo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MJXContractTest(unittest.TestCase):
    def test_nominal_task_uses_same_generated_model(self) -> None:
        self.assertEqual(
            MODEL_PATH, PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
        )
        self.assertTrue(MODEL_PATH.exists())
        self.assertEqual(len(JOINT_NAMES), 4)

    def test_nominal_task_control_rate_and_action_shape(self) -> None:
        config = NominalRLConfig()
        self.assertAlmostEqual(config.control_timestep, 0.02)
        self.assertIsNone(config.model_xml)
        self.assertEqual(len(config.action_scales), 4)
        self.assertEqual(config.episode_length, 500)
        self.assertEqual(config.startup_action_ramp_s, 0.25)
        self.assertTrue(config.disable_root_damping)
        self.assertEqual(config.disturbance_root_x_velocity_m_s, 0.0)
        self.assertEqual(config.disturbance_root_pitch_velocity_rad_s, 0.0)
        self.assertEqual(config.disturbance_probability, 1.0)
        self.assertEqual(config.disturbance_level_scales, (1.0,))
        self.assertEqual(config.disturbance_level_probabilities, (1.0,))
        self.assertEqual(config.disturbance_backward_probability, 0.5)
        self.assertIsNone(config.terminate_root_z_min)
        self.assertEqual(config.terminate_root_z_low_duration_s, 0.30)
        self.assertIsNone(config.terminate_stuck_root_z_max)
        self.assertEqual(config.terminate_stuck_progress_window_s, 1.0)
        self.assertEqual(config.terminate_stuck_min_progress_rad, 0.20)
        self.assertEqual(config.terminate_stuck_duration_s, 0.75)
        self.assertEqual(config.terminate_stuck_grace_s, 1.50)
        self.assertEqual(config.tail_progress_window_s, 2.0)
        self.assertEqual(config.terminate_root_z_max, 0.70)
        self.assertEqual(config.maximum_foot_center_distance_m, 0.28)
        self.assertTrue(config.terminate_leg_crossing)
        reward = RollingRewardConfig()
        self.assertEqual(reward.allowed_foot_penetration_m, 0.0005)
        self.assertEqual(reward.foot_contact_event, 2.0)
        self.assertEqual(reward.foot_contact_time, 4.0)
        self.assertEqual(reward.termination, 5.0)
        self.assertEqual(reward.root_low_extra_termination, 35.0)
        self.assertEqual(reward.stuck_extra_termination, 35.0)
        self.assertEqual(reward.early_termination_scale, 1.0)

    def test_disturbance_configuration_requires_valid_episode_step(self) -> None:
        validate_nominal_rl_config(
            NominalRLConfig(
                disturbance_root_x_velocity_m_s=0.2,
                disturbance_root_pitch_velocity_rad_s=0.8,
                disturbance_min_step=100,
                disturbance_max_step=400,
            )
        )
        with self.assertRaisesRegex(ValueError, "episode_length"):
            validate_nominal_rl_config(
                NominalRLConfig(
                    episode_length=100,
                    disturbance_root_x_velocity_m_s=0.2,
                    disturbance_min_step=50,
                    disturbance_max_step=100,
                )
            )

    def test_root_low_termination_configuration_is_validated(self) -> None:
        validate_nominal_rl_config(
            NominalRLConfig(
                terminate_root_z_min=0.05,
                terminate_root_z_low_duration_s=0.30,
            )
        )
        invalid = (
            {"terminate_root_z_min": -0.01},
            {"terminate_root_z_min": float("nan")},
            {
                "terminate_root_z_min": 0.05,
                "terminate_root_z_low_duration_s": 0.0,
            },
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_nominal_rl_config(NominalRLConfig(**values))

    def test_hard_termination_configuration_can_be_relaxed(self) -> None:
        validate_nominal_rl_config(
            NominalRLConfig(
                terminate_root_z_max=None,
                maximum_foot_center_distance_m=None,
                terminate_leg_crossing=False,
                terminate_stuck_root_z_max=None,
            )
        )
        invalid = (
            {"terminate_root_z_max": 0.0},
            {"terminate_root_z_max": float("nan")},
            {"maximum_foot_center_distance_m": -0.1},
            {"maximum_foot_center_distance_m": float("nan")},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_nominal_rl_config(NominalRLConfig(**values))

    def test_stuck_termination_configuration_is_validated(self) -> None:
        validate_nominal_rl_config(
            NominalRLConfig(
                terminate_stuck_root_z_max=0.10,
                terminate_stuck_progress_window_s=1.0,
                terminate_stuck_min_progress_rad=0.20,
                terminate_stuck_duration_s=0.75,
                terminate_stuck_grace_s=1.50,
            )
        )
        invalid = (
            {"terminate_stuck_root_z_max": 0.0},
            {
                "terminate_stuck_root_z_max": 0.10,
                "terminate_stuck_progress_window_s": 0.0,
            },
            {
                "terminate_stuck_root_z_max": 0.10,
                "terminate_stuck_min_progress_rad": -0.1,
            },
            {
                "terminate_stuck_root_z_max": 0.10,
                "terminate_stuck_grace_s": 0.5,
            },
            {"tail_progress_window_s": 20.0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_nominal_rl_config(NominalRLConfig(**values))

    def test_task_can_select_explicit_model_xml(self) -> None:
        self.assertEqual(resolve_model_path(NominalRLConfig()), MODEL_PATH)
        self.assertEqual(
            resolve_model_path(
                NominalRLConfig(
                    model_xml="assets/curl_robot_2d_no_self_collision.xml"
                )
            ),
            PROJECT_ROOT / "assets" / "curl_robot_2d_no_self_collision.xml",
        )

    def test_disturbance_mixture_validation(self) -> None:
        validate_nominal_rl_config(
            NominalRLConfig(
                disturbance_root_x_velocity_m_s=1.0,
                disturbance_probability=0.5,
                disturbance_level_scales=(0.5, 1.0, 1.5),
                disturbance_level_probabilities=(0.6, 0.3, 0.1),
                disturbance_backward_probability=0.2,
            )
        )
        invalid = (
            {"disturbance_probability": 1.1},
            {"disturbance_backward_probability": -0.1},
            {
                "disturbance_level_scales": (0.5, 1.0),
                "disturbance_level_probabilities": (1.0,),
            },
            {
                "disturbance_level_scales": (0.5, 1.0),
                "disturbance_level_probabilities": (0.5, 0.4),
            },
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_nominal_rl_config(NominalRLConfig(**values))

    def test_startup_action_ramp_matches_cem_smoothstep(self) -> None:
        elapsed = np.asarray([0.0, 0.125, 0.25, 0.5])
        np.testing.assert_allclose(
            smoothstep_ramp(np, elapsed, 0.25),
            np.asarray([0.0, 0.5, 1.0, 1.0]),
        )

    def test_candidate_physics_keeps_control_rate(self) -> None:
        reference = physics_profile("reference")
        newton4 = physics_profile("newton4")
        cg12 = physics_profile("cg12")
        self.assertAlmostEqual(reference.control_timestep, 0.02)
        self.assertAlmostEqual(newton4.control_timestep, 0.02)
        self.assertAlmostEqual(cg12.control_timestep, 0.02)
        self.assertEqual(reference.action_repeat, 20)
        self.assertEqual(newton4.action_repeat, 20)
        self.assertEqual(cg12.action_repeat, 20)
        self.assertEqual(reference.integrator_name, "implicitfast")
        self.assertEqual(cg12.integrator_name, "implicitfast")
        self.assertEqual(reference.cone_name, "elliptic")
        self.assertEqual(cg12.cone_name, "elliptic")
        self.assertEqual(cg12.solver_name, "cg")
        self.assertLess(
            cg12.solver_iterations, reference.solver_iterations
        )

    def test_cloud_runtime_configuration_is_dependency_light(self) -> None:
        configure_cloud_runtime(memory_fraction=0.85, preallocate=False)
        import os

        self.assertIn(
            "--xla_gpu_triton_gemm_any=True", os.environ["XLA_FLAGS"]
        )
        self.assertIn(
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            os.environ["XLA_FLAGS"],
        )
        self.assertEqual(
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.85"
        )

    def test_cloud_runtime_selects_headless_linux_backend(self) -> None:
        self.assertEqual(
            select_mujoco_gl_backend(
                environ={}, platform_name="linux"
            ),
            "egl",
        )

    def test_mjx_smoke_starts_with_single_environment(self) -> None:
        args = mjx_smoke.parse_args([])
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.steps, 1)

    def test_cem_reference_ablation_is_local_cpu_safe_by_default(self) -> None:
        args = compare_mjx_cem_reference.parse_args([])
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.noise_seeds, 32)
        self.assertEqual(args.mujoco_gl, "disable")
        self.assertEqual(tuple(args.cases), ("A", "B", "C", "D"))
        self.assertEqual(args.disturbance_root_x_velocity, 0.0)
        self.assertEqual(args.disturbance_root_pitch_velocity, 0.0)
        self.assertEqual(args.disturbance_probability, 1.0)
        self.assertEqual(args.disturbance_level_scales, [1.0])
        self.assertEqual(args.disturbance_level_probabilities, [1.0])
        self.assertEqual(args.disturbance_backward_probability, 0.5)
        self.assertIsNone(args.environment_seed)
        self.assertIsNone(args.rollout_seed)

    def test_cem_reference_ablation_accepts_disturbances(self) -> None:
        args = compare_mjx_cem_reference.parse_args(
            [
                "--disturbance-root-x-velocity",
                "0.2",
                "--disturbance-root-pitch-velocity",
                "0.8",
                "--disturbance-probability",
                "0.5",
                "--disturbance-level-scales",
                "0.5",
                "1.0",
                "1.5",
                "--disturbance-level-probabilities",
                "0.6",
                "0.3",
                "0.1",
                "--disturbance-backward-probability",
                "0.2",
                "--environment-seed",
                "20000",
                "--rollout-seed",
                "40000",
                "--cases",
                "D",
            ]
        )

        self.assertEqual(args.disturbance_root_x_velocity, 0.2)
        self.assertEqual(args.disturbance_root_pitch_velocity, 0.8)
        self.assertEqual(args.disturbance_probability, 0.5)
        self.assertEqual(args.disturbance_level_scales, [0.5, 1.0, 1.5])
        self.assertEqual(
            args.disturbance_level_probabilities, [0.6, 0.3, 0.1]
        )
        self.assertEqual(args.disturbance_backward_probability, 0.2)
        self.assertEqual(args.environment_seed, 20000)
        self.assertEqual(args.rollout_seed, 40000)
        self.assertEqual(args.cases, ["D"])

    def test_root_damping_can_match_cpu_cem_runtime(self) -> None:
        import mujoco

        from curl_robot_2d_mjx.environment import apply_physics_options

        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        task = NominalRLConfig(disable_root_damping=True)
        apply_physics_options(model, task)

        for joint_name in ("root_x", "root_z", "root_pitch"):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            dof_id = int(model.jnt_dofadr[joint_id])
            self.assertEqual(float(model.dof_damping[dof_id]), 0.0)

    def test_training_entries_import_without_jax(self) -> None:
        self.assertTrue(callable(mjx_smoke.main))
        self.assertTrue(callable(train_mjx_ppo.main))
        self.assertIn("4090", train_mjx_ppo.PRESETS)
        self.assertIn("h200", train_mjx_ppo.PRESETS)

    def test_restore_checkpoint_selects_latest_numbered_child(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "000000001000").mkdir()
            latest = root / "000000010000"
            latest.mkdir()
            self.assertEqual(
                train_mjx_ppo._resolve_restore_checkpoint(root),
                latest.resolve(),
            )

    def test_environment_declares_collision_and_progress_metrics(self) -> None:
        source = (
            PROJECT_ROOT
            / "curl_robot_2d_mjx"
            / "environment.py"
        ).read_text(encoding="utf-8")
        for metric in (
            "forbidden_contact_count",
            "forbidden_penetration_m",
            "allowed_foot_penetration_m",
            "foot_contact_active",
            "foot_contact_start",
            "leg_crossing",
            "root_low_active",
            "root_low_step_count",
            "stuck_active",
            "stuck_step_count",
            "rolling_window_progress_rad",
            "tail_roll_progress_rad",
            "failure_nonfinite_action",
            "failure_nonfinite_physics",
            "failure_root_low",
            "failure_stuck",
            "failure_root_high",
            "failure_foot_gap",
        ):
            self.assertIn(f'"{metric}"', source)
        self.assertIn("roll_progress", REWARD_TERM_NAMES)
        self.assertIn("collision", REWARD_TERM_NAMES)
        self.assertIn("termination", REWARD_TERM_NAMES)

    def test_environment_quarantines_nonfinite_transitions(self) -> None:
        source = (
            PROJECT_ROOT
            / "curl_robot_2d_mjx"
            / "environment.py"
        ).read_text(encoding="utf-8")
        for guard in (
            "action_finite",
            "physics_finite",
            "transition_finite",
            "jax.lax.cond",
            "jp.nan_to_num",
        ):
            self.assertIn(guard, source)
        step_source = source[
            source.index("        def step(") :
            source.index("        def _contact_arrays(")
        ]
        self.assertLess(
            step_source.index("transition_finite ="),
            step_source.index("contacts = self._contact_metrics(data)"),
        )


if __name__ == "__main__":
    unittest.main()
