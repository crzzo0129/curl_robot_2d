import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict

from scripts.train_mjx_3d_transition_ppo import build_task, parse_args
from curl_robot_2d_mjx.reference_bank_contract_3d import validate_reference_split


class PhaseWindowBankTests(unittest.TestCase):
    def test_split_rejects_same_cycles_even_with_different_seeds(self):
        task = build_task(parse_args([
            "--geometry", "rollingquad_2_abd10_no_self_collision",
            "--dynamic-roll-to-stand", "--physics-profile", "accurate"]))
        def report(seed, cycle):
            return dict(source_kind="roll_to_stand_phase_window_reference",
                        status="ok", seed=seed, reference_sha256="same",
                        task=asdict(task), stand_abduction_deg=[0.]*4,
                        pitch_targets_deg=[80., 90., 100.],
                        handoffs=[dict(target_pitch_deg=p, pitch_deg=p-.3,
                                       pitch_rate_rad_s=-7., minimum_turns=cycle)
                                  for p in (80., 90., 100.)])
        with tempfile.TemporaryDirectory() as directory:
            train, evaluation = Path(directory)/"train.npz", Path(directory)/"eval.npz"
            def write(path, value):
                path.with_suffix(".summary.json").write_text(json.dumps(value), encoding="utf-8")
            write(train, report(0, 1))
            write(evaluation, report(1000, 9))
            self.assertTrue(validate_reference_split(train, evaluation, task)["nominal_dynamics_match"])
            write(evaluation, report(1000, 1))
            with self.assertRaisesRegex(ValueError, "cycles overlap"):
                validate_reference_split(train, evaluation, task)
            bad = report(1000, 9)
            bad["handoffs"][0]["pitch_rate_rad_s"] = 7.
            write(evaluation, bad)
            with self.assertRaisesRegex(ValueError, "roll direction"):
                validate_reference_split(train, evaluation, task)


if __name__ == "__main__":
    unittest.main()
