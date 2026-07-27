from pathlib import Path
import unittest

from curl_robot_2d_mjx.config import NominalRLConfig, physics_profile
from curl_robot_2d_mjx.environment import JOINT_NAMES, MODEL_PATH
from curl_robot_2d_mjx.reward import REWARD_TERM_NAMES
from curl_robot_2d_mjx.reward_config import RollingRewardConfig
from curl_robot_2d_mjx.runtime import (
    configure_cloud_runtime,
    select_mujoco_gl_backend,
)
from scripts import mjx_smoke, train_mjx_ppo


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
        self.assertEqual(len(config.action_scales), 4)
        self.assertEqual(config.episode_length, 500)
        reward = RollingRewardConfig()
        self.assertEqual(reward.allowed_foot_penetration_m, 0.0005)
        self.assertEqual(reward.termination, 5.0)

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
            "leg_crossing",
            "failure_root_low",
            "failure_root_high",
            "failure_foot_gap",
        ):
            self.assertIn(f'"{metric}"', source)
        self.assertIn("roll_progress", REWARD_TERM_NAMES)
        self.assertIn("collision", REWARD_TERM_NAMES)
        self.assertIn("termination", REWARD_TERM_NAMES)


if __name__ == "__main__":
    unittest.main()
