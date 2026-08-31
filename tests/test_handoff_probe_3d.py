from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.cem_reference import load_cem_reference
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.environment_3d import cem_controller_path_3d
from curl_robot_2d_mjx.handoff_probe_3d import (
    FAILURES, HandoffNoise, continuation_rows, perturbation_batch, sampling_steps, summarize_probes,
)
from scripts.probe_3d_roll_handoff import CPUReference, experiment_config, parse_args
from scripts.analyze_3d_roll_handoff import compare_configs, read_trials


class HandoffProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = physics_profile_3d("cg20", Rolling3DConfig(
            reset_joint_noise_rad=0, reset_velocity_noise=0))
        cls.reference = load_cem_reference(cem_controller_path_3d("rollingquad_2"),
                                           reference_weight=1, minimum_residual_gain=0.15)
        cls.runner = CPUReference(cls.task, cls.reference, {}, None)

    def test_grid_includes_zero_and_three_seconds(self):
        self.assertEqual(sampling_steps(3, 0.5, 0.02), (0, 25, 50, 75, 100, 125, 150))
        self.assertEqual(sampling_steps(3, 0.8, 0.02)[-1], 150)
        for values in ((3, 0.25, 0.02), (3, 0, 0.02), (3, float("nan"), 0.02)):
            with self.assertRaises(ValueError):
                sampling_steps(*values)

    def test_default_backend_cannot_silently_run_reference_without_teacher(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--out", "unused-probe-test"])

    def test_reference_backend_rejects_misleading_teacher_argument(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--backend", "cpu-reference", "--teacher", "some_checkpoint", "--out", "unused"])

    def test_teacher_config_required_and_window_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "params_best"
            checkpoint.touch()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--teacher", str(checkpoint), "--out", str(Path(directory) / "out")])
            args = parse_args(["--teacher", str(checkpoint), "--assume-accepted-gain-config",
                               "--out", str(Path(directory) / "out")])
            task, _, payload, _, _ = experiment_config(args)
            self.assertEqual(payload["recipe"], "robust_recovery_v15")
            self.assertIn("ASSUMED", payload["configuration_source"])
            self.assertTrue(task.explicit_phase_observation)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--backend", "cpu-reference", "--window-s", "4", "--out", "unused"])

    def test_horizon_extension_does_not_enable_stand_interpolation(self):
        args = parse_args(["--backend", "cpu-reference", "--out", "unused-probe-test",
                           "--continuation-s", "10"])
        task, _, _, _, _ = experiment_config(args)
        self.assertEqual(task.reset_pose, "compact")
        self.assertGreater(task.episode_length * task.control_timestep, 13)
        self.assertEqual(task.terminate_lateral_drift_m, 0.2)

    def test_noise_axes_and_seed_reproducibility(self):
        noise = HandoffNoise()
        for case, active in (("exact", ()), ("state_noise", ("dq", "dqd", "dv", "daxis")),
                             ("phase_noise", ("dphase",)), ("history_noise", ("dhistory",))):
            a, b = perturbation_batch(case, noise, 12, 4), perturbation_batch(case, noise, 12, 4)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
                if key not in active:
                    np.testing.assert_array_equal(a[key], 0)
                else:
                    self.assertTrue(np.any(a[key] != 0))
        with self.assertRaises(ValueError):
            replace(noise, oscillator_phase_rad=-1).validate()

    def test_cpu_exact_snapshot_replays_without_mutating_source(self):
        state = self.runner.reset(1, 0)
        for _ in range(8):
            state = self.runner.step(state)
        snapshot = self.runner.clone(state)
        saved = self.runner.features(snapshot)
        clone = self.runner.branch(snapshot, [0], perturbation_batch("exact", HandoffNoise(), 0, 1), "exact")
        for _ in range(8):
            state, clone = self.runner.step(state), self.runner.step(clone)
        for name in ("qpos", "qvel", "ctrl", "time", "oscillator_phase", "rolling_phase", "last_action"):
            np.testing.assert_array_equal(self.runner.features(state)[name], self.runner.features(clone)[name])
            np.testing.assert_array_equal(saved[name], self.runner.features(snapshot)[name])

    def test_history_probe_does_not_touch_physics_or_clock(self):
        snapshot = self.runner.reset(1, 0)
        for _ in range(3):
            snapshot = self.runner.step(snapshot)
        original = self.runner.features(snapshot)
        modified = self.runner.branch(snapshot, [0], perturbation_batch("history_noise", HandoffNoise(), 3, 1),
                                      "history_noise")
        features = self.runner.features(modified)
        for key in ("qpos", "qvel", "ctrl", "time", "oscillator_phase"):
            np.testing.assert_array_equal(features[key], original[key])
        self.assertTrue(np.any(features["last_action"] != original["last_action"]))
        # CEM does not read last_action, unlike the residual teacher network.
        np.testing.assert_array_equal(self.runner.features(self.runner.step(modified))["qpos"],
                                      self.runner.features(self.runner.step(self.runner.clone(snapshot)))["qpos"])

    def test_state_probe_preserves_counters_origin_and_normalizes_quaternion(self):
        snapshot = self.runner.step(self.runner.reset(1, 0))
        changed = self.runner.branch(snapshot, [0], perturbation_batch("state_noise", HandoffNoise(), 3, 1),
                                     "state_noise")
        for key in ("step_count", "initial_root_y", "rolling_phase", "oscillator_phase", "cumulative_rotation"):
            self.assertEqual(changed[0]["info"][key], snapshot[0]["info"][key])
        self.assertEqual(changed[0]["data"].time, snapshot[0]["data"].time)
        self.assertAlmostEqual(np.linalg.norm(changed[0]["data"].qpos[3:7]), 1)
        archive = self.runner.snapshot_arrays(snapshot)
        self.assertIn("physics_integration_state", archive)
        self.assertIn("info_last_action", archive)
        self.assertIn("info_oscillator_phase", archive)

    def test_success_requires_new_progress_and_full_horizon(self):
        start = self.runner.features(self.runner.reset(1, 0))
        # Large accumulated turns before takeover must not count as new progress.
        start["absolute_rotation"][:] = 20
        end = {k: v.copy() for k, v in start.items()}
        end["time"][:] = 3
        maxima = {"y": np.zeros(1), "axis_tilt": np.zeros(1), "torque": np.zeros(1),
                  "first_command_jump": np.zeros(1)}
        def rows():
            return continuation_rows(start, end, source_ids=[0], source_success=[True], case="exact",
                sample_step=0, dt=0.02, horizon_s=3, minimum_turn_rate=0.5, maxima=maxima)
        self.assertFalse(rows()[0]["success"])
        slow_summary = summarize_probes(rows())[0]
        self.assertEqual(slow_summary["failure_free_rate"], 1)
        self.assertEqual(slow_summary["slow_but_failure_free_rate"], 1)
        end["absolute_rotation"][:] += 2 * np.pi * 2
        end["rolling_phase"][:] += 2 * np.pi * 2
        end["qpos"][:, 0] += 2 * np.pi * float(start["radius"][0]) * 2
        self.assertTrue(rows()[0]["success"])
        end["failed"][:] = True
        end["failure_lateral_drift"][:] = True
        self.assertFalse(rows()[0]["success"])
        end["failed"][:] = False
        end["time"][:] = 2
        self.assertFalse(rows()[0]["success"])
        summary = summarize_probes(rows())[0]
        self.assertEqual(summary["trials"], 1)
        self.assertEqual(summary["success_rate"], 0)
        self.assertEqual(summary["failure_free_rate"], 0)

    def test_config_audit_checks_physics_but_not_reference_source_path(self):
        args = parse_args(["--backend", "cpu-reference", "--out", "unused-probe-test"])
        args.assume_accepted_gain_config = True
        _, _, recorded, _, _ = experiment_config(args)
        supplied = json.loads(json.dumps(recorded))
        supplied["reference"]["source"] = "/cloud/reference.json"
        # An older config omits stand options, whose defaults remain compact.
        del supplied["task"]["reset_pose"]
        self.assertEqual(compare_configs(recorded, supplied), [])
        supplied["task"]["floor_contact_friction_override"] = True
        diff = compare_configs(recorded, supplied)
        self.assertEqual([d["field"] for d in diff], ["task.floor_contact_friction_override"])

    def test_csv_analysis_does_not_treat_false_strings_as_true(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            path.write_text("case,success,source_id,continued_s,exact_replay_qpos_max_error\n"
                            "state_noise,False,0,3.0,\n", encoding="utf-8")
            row = read_trials(path)[0]
            self.assertIs(row["success"], False)
            self.assertEqual(row["case"], "state_noise")
            self.assertEqual(row["continued_s"], 3)
            self.assertIsNone(row["exact_replay_qpos_max_error"])


if __name__ == "__main__":
    unittest.main()
