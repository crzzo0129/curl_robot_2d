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
