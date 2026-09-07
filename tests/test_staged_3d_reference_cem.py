from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from scripts import run_staged_3d_reference_cem as cem3d


class Staged3DReferenceCEMTest(unittest.TestCase):
    def test_search_contract_keeps_old_controller_and_adds_gap(self) -> None:
        lower, upper = cem3d.parameter_bounds()
        self.assertEqual(lower.shape, (11,))
        self.assertEqual(upper.shape, (11,))
        self.assertEqual(len(cem3d.PARAMETER_NAMES), 11)
        self.assertEqual(cem3d.PARAMETER_NAMES[:8], cem3d.COEFFICIENT_NAMES)
        self.assertEqual(lower[-1], 0.0)
        self.assertEqual(upper[-1], 0.006)

    def test_default_is_full_three_stage_rollingquad_search(self) -> None:
        args = cem3d.parse_args([])
        self.assertEqual(args.preset, "full")
        self.assertEqual(args.physics_profile, "cg20")
        self.assertEqual(args.initial_gap_mm, 2.0)
        self.assertEqual(args.torque_limit, 3.0)
        self.assertEqual(
            [stage.name for stage in cem3d.FULL_STAGES],
            ["01_recover_roll", "02_reduce_contact", "03_strict_10s"],
        )
        self.assertEqual(cem3d.FULL_STAGES[-1].duration_s, 10.0)

    def test_high_speed_objective_is_uncapped_and_rewards_tail_speed(self) -> None:
        args = cem3d.parse_args(["--objective", "high_speed"])
        self.assertEqual(args.objective, "high_speed")
        self.assertEqual(len(cem3d.HIGH_SPEED_STAGES), 3)
        self.assertTrue(all(stage.uncapped_progress for stage in cem3d.HIGH_SPEED_STAGES))
        self.assertTrue(
            all(stage.forward_speed_weight > 0.0 for stage in cem3d.HIGH_SPEED_STAGES)
        )
        self.assertTrue(
            all(stage.tail_speed_weight > 0.0 for stage in cem3d.HIGH_SPEED_STAGES)
        )

    def test_zero_contact_refine_is_local_and_contact_is_infeasible(self) -> None:
        args = cem3d.parse_args(["--objective", "zero_contact_refine"])
        self.assertEqual(args.objective, "zero_contact_refine")
        stage = cem3d.ZERO_CONTACT_REFINE_SMOKE_STAGE
        self.assertTrue(stage.require_zero_contact)
        self.assertLess(stage.search_std_scale, 1.0)
        self.assertEqual(stage.duration_s, 10.0)

    def test_speed_discovery_records_zero_contact_without_hard_rejection(self) -> None:
        stage = cem3d.SPEED_DISCOVERY_ZERO_CONTACT_STAGE
        self.assertFalse(stage.require_zero_contact)
        self.assertTrue(stage.export_best_zero_contact)
        self.assertEqual(stage.name, "01_speed_discovery_zero_contact")

    def test_export_is_compatible_with_shared_reference_loader(self) -> None:
        source = cem3d.PUPPER_OPEN60_CEM_CONTROLLER
        parameters = cem3d.controller_parameters(source, initial_gap_m=0.002)
        rollout = cem3d.RolloutResult(
            score=1.0,
            summary={"score": 1.0, "conservative_rolling_turns": 5.0},
        )
        payload = cem3d.controller_payload(
            parameters,
            rollout,
            stage=cem3d.FULL_STAGES[-1],
            source_controller=source,
            tracking_margin_m=0.004,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_phase_controller.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_cem_reference(path)
        np.testing.assert_allclose(loaded.coefficients, parameters[:8])
        self.assertAlmostEqual(loaded.minimum_foot_surface_gap_m, 0.002)
        self.assertAlmostEqual(loaded.foot_gap_tracking_margin_m, 0.004)

    def test_zero_gap_export_has_no_artificial_knee_bias(self) -> None:
        source = cem3d.PUPPER_OPEN60_CEM_CONTROLLER
        parameters = cem3d.controller_parameters(source, initial_gap_m=0.0)
        payload = cem3d.controller_payload(
            parameters,
            cem3d.RolloutResult(0.0, {}),
            stage=cem3d.SMOKE_STAGES[0],
            source_controller=source,
            tracking_margin_m=0.004,
        )
        self.assertEqual(payload["minimum_foot_surface_gap_m"], 0.0)
        self.assertEqual(payload["nominal_knee_bias_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
