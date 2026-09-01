from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from curl_robot_2d_mjx.deployment_rolling_3d import (
    ROLLING_EFFECTIVE_ACTION_INDICES_3D,
)
from curl_robot_2d_mjx.rolling_student_dr_ppo_3d import (
    expand_ppo_actor_to_controller_3d,
    initialize_ppo_actor_from_student_3d,
    tanh_normal_scale_logit_3d,
)
from curl_robot_2d_mjx.wrappers_rolling_student_dr_3d import (
    select_reset_lanes_rolling_3d,
)
from scripts import train_mjx_3d_roll_student_dr_ppo
from scripts.export_rtneural import convert as convert_rtneural


def _dense(rng, inputs, outputs):
    return {
        "kernel": rng.normal(size=(inputs, outputs)).astype(np.float32),
        "bias": rng.normal(size=(outputs,)).astype(np.float32),
    }


class RollingStudentDRPPOContractTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.student = {
            "params": {
                "hidden_0": _dense(rng, 5, 4),
                "hidden_1": _dense(rng, 4, 3),
                "location": _dense(rng, 3, 12),
            }
        }
        self.ppo = {
            "params": {
                "hidden_0": _dense(rng, 5, 4),
                "hidden_1": _dense(rng, 4, 3),
                "hidden_2": _dense(rng, 3, 16),
            }
        }

    @staticmethod
    def _elu(value):
        return np.where(value > 0.0, value, np.expm1(value))

    def test_student_initialization_preserves_effective_mean_action(self):
        initialized = initialize_ppo_actor_from_student_3d(
            np,
            self.ppo,
            self.student,
            hidden_layers=(4, 3),
            initial_std=0.05,
        )
        observation = np.linspace(-0.4, 0.4, 5, dtype=np.float32)
        hidden = observation
        for name in ("hidden_0", "hidden_1"):
            layer = self.student["params"][name]
            hidden = self._elu(hidden @ layer["kernel"] + layer["bias"])
        student_head = self.student["params"]["location"]
        student_action = np.tanh(
            hidden @ student_head["kernel"] + student_head["bias"]
        )
        ppo_head = initialized["params"]["hidden_2"]
        ppo_action = np.tanh(
            (hidden @ ppo_head["kernel"] + ppo_head["bias"])[:8]
        )

        np.testing.assert_allclose(
            ppo_action,
            student_action[list(ROLLING_EFFECTIVE_ACTION_INDICES_3D)],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_student_initialization_sets_requested_policy_std(self):
        initialized = initialize_ppo_actor_from_student_3d(
            np,
            self.ppo,
            self.student,
            hidden_layers=(4, 3),
            initial_std=0.05,
        )
        head = initialized["params"]["hidden_2"]
        np.testing.assert_array_equal(head["kernel"][:, 8:], 0.0)
        scale = np.logaddexp(0.0, head["bias"][8:]) + 0.001
        np.testing.assert_allclose(scale, 0.05, rtol=1e-6, atol=1e-6)

    def test_export_expands_to_12_and_locks_abduction(self):
        initialized = initialize_ppo_actor_from_student_3d(
            np,
            self.ppo,
            self.student,
            hidden_layers=(4, 3),
            initial_std=0.05,
        )
        expanded = expand_ppo_actor_to_controller_3d(np, initialized)
        source = initialized["params"]["hidden_2"]
        target = expanded["params"]["hidden_2"]

        self.assertEqual(target["kernel"].shape, (3, 12))
        self.assertEqual(target["bias"].shape, (12,))
        np.testing.assert_array_equal(target["kernel"][:, [0, 3, 6, 9]], 0.0)
        np.testing.assert_array_equal(target["bias"][[0, 3, 6, 9]], 0.0)
        np.testing.assert_array_equal(
            target["kernel"][:, list(ROLLING_EFFECTIVE_ACTION_INDICES_3D)],
            source["kernel"][:, :8],
        )
        np.testing.assert_array_equal(
            target["bias"][list(ROLLING_EFFECTIVE_ACTION_INDICES_3D)],
            source["bias"][:8],
        )

        document = convert_rtneural(
            (
                {"mean": np.zeros(5), "std": np.ones(5)},
                expanded,
                {},
            ),
            {"action_scale": [1.0] * 12},
            activation="elu",
            observation_history=1,
        )
        self.assertEqual(document["in_shape"], [1, 5])
        self.assertEqual(document["out_shape"], [1, 12])

    def test_initial_std_validation(self):
        self.assertAlmostEqual(
            np.logaddexp(0.0, tanh_normal_scale_logit_3d(0.05)) + 0.001,
            0.05,
        )
        with self.assertRaises(ValueError):
            tanh_normal_scale_logit_3d(0.001)

    def test_full_reset_lane_selection_broadcasts_done(self):
        fresh = np.full((3, 2), 9.0)
        current = np.arange(6, dtype=np.float32).reshape(3, 2)
        selected = select_reset_lanes_rolling_3d(
            np, np.asarray((False, True, False)), fresh, current
        )
        np.testing.assert_array_equal(selected[0], current[0])
        np.testing.assert_array_equal(selected[1], fresh[1])
        np.testing.assert_array_equal(selected[2], current[2])

    def test_cli_defaults_to_first_reward_dr_ppo_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            student = root / "student_params"
            controller = root / "controller.json"
            student.write_bytes(b"placeholder")
            controller.write_text("{}", encoding="utf-8")
            args = train_mjx_3d_roll_student_dr_ppo.parse_args(
                [
                    str(student),
                    "--controller",
                    str(controller),
                    "--out",
                    str(root / "output"),
                    "--max-devices",
                    "4",
                ]
            )

        self.assertEqual(args.dr_strength, 0.25)
        self.assertEqual(args.student_anchor_weight, 0.02)
        self.assertEqual(args.minimum_success_turns, 5.0)
        self.assertEqual(args.envs, 64)
        self.assertEqual(args.eval_envs, 8)
        self.assertEqual(args.max_devices, 4)

    def test_cli_rejects_missing_restore_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            student = root / "student_params"
            controller = root / "controller.json"
            student.write_bytes(b"placeholder")
            controller.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                train_mjx_3d_roll_student_dr_ppo.parse_args(
                    [
                        str(student),
                        "--restore-ppo",
                        str(root / "missing_params"),
                        "--controller",
                        str(controller),
                        "--out",
                        str(root / "output"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
