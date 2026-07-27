from pathlib import Path
import unittest

from curl_robot_2d_mjx.config import NominalRLConfig
from curl_robot_2d_mjx.environment import JOINT_NAMES, MODEL_PATH
from curl_robot_2d_mjx.runtime import configure_cloud_runtime
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
        self.assertEqual(config.allowed_foot_penetration_m, 0.0005)

    def test_cloud_runtime_configuration_is_dependency_light(self) -> None:
        configure_cloud_runtime(memory_fraction=0.85, preallocate=False)
        import os

        self.assertIn(
            "--xla_gpu_triton_gemm_any=true", os.environ["XLA_FLAGS"]
        )
        self.assertEqual(
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.85"
        )

    def test_training_entries_import_without_jax(self) -> None:
        self.assertTrue(callable(mjx_smoke.main))
        self.assertTrue(callable(train_mjx_ppo.main))
        self.assertIn("4090", train_mjx_ppo.PRESETS)
        self.assertIn("h200", train_mjx_ppo.PRESETS)

    def test_environment_declares_collision_and_progress_metrics(self) -> None:
        source = (
            PROJECT_ROOT
            / "curl_robot_2d_mjx"
            / "environment.py"
        ).read_text(encoding="utf-8")
        for metric in (
            "reward_roll_progress",
            "reward_collision",
            "forbidden_contact_count",
            "forbidden_penetration_m",
            "allowed_foot_penetration_m",
            "leg_crossing",
        ):
            self.assertIn(f'"{metric}"', source)


if __name__ == "__main__":
    unittest.main()
