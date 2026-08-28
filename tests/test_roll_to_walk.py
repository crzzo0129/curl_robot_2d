import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from curl_robot_2d.roll_to_walk import (
    RollToWalkConfig,
    TransitionMode,
    load_roll_controller,
    simulate_roll_to_walk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"


class RollToWalkTest(unittest.TestCase):
    def test_controller_has_fallback_and_expected_shape(self) -> None:
        controller = load_roll_controller(PROJECT_ROOT / "missing.json")
        self.assertEqual(controller.coefficients.shape, (8,))
        self.assertTrue(np.isfinite(controller.coefficients).all())

    def test_config_total_duration(self) -> None:
        config = RollToWalkConfig(1.0, 0.5, 0.75, 2.0)
        self.assertAlmostEqual(config.total_duration_s, 4.25)

    def test_complete_transition_runs_all_modes(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        config = RollToWalkConfig(
            roll_duration_s=0.20,
            brake_duration_s=0.20,
            deploy_duration_s=0.30,
            walk_duration_s=0.30,
        )
        result = simulate_roll_to_walk(model, config, detailed=True)
        self.assertEqual(
            result.mode_history,
            tuple(mode.value for mode in (
                TransitionMode.ROLL,
                TransitionMode.BRAKE,
                TransitionMode.DEPLOY,
                TransitionMode.WALK,
                TransitionMode.COMPLETE,
            )),
        )
        self.assertTrue(np.isfinite(result.rows).all())
        self.assertEqual(result.summary["final_mode"], "complete")
        self.assertGreaterEqual(result.summary["mode_count"], 5)

    def test_cli_summary_is_json_serializable(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        result = simulate_roll_to_walk(
            model,
            RollToWalkConfig(
                roll_duration_s=0.05,
                brake_duration_s=0.05,
                deploy_duration_s=0.05,
                walk_duration_s=0.05,
            ),
        )
        json.dumps({**result.summary, "mode_history": result.mode_history})


if __name__ == "__main__":
    unittest.main()
