"""Synthetic trajectory fixtures test selection, not learned robot capability."""

import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from curl_robot_2d_mjx.config_transition_3d import (
    Transition3DConfig, transition_curriculum_config_3d, validate_transition_config_3d,
)
from curl_robot_2d_mjx.environment_transition_3d import TRANSITION_MODEL_PATH_3D
from curl_robot_2d_mjx.roll_snapshot_curriculum_3d import select_roll_cycle_snapshots
from curl_robot_2d_mjx.training_transition_3d import resolve_transition_checkpoint
from curl_robot_2d_mjx.transition_initialization_3d import (
    collect_roll_snapshots_3d, load_roll_snapshots_3d, save_roll_snapshots_3d,
    validate_roll_snapshots_3d, walking_start_state_3d,
)
from scripts.train_mjx_3d_transition_ppo import build_task, parse_args, main as train_main
from scripts.inspect_transition_roll_snapshots import main as inspect_main


class RollCycleSelectionTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        self.config = Transition3DConfig()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "bank.npz"
        self.turns = np.arange(40) / 8 + .0625
        self.arrays = self.save_bank(self.path, self.turns)

    def save_bank(self, path, turns, *, origin=2.3, episodes=None):
        initial = walking_start_state_3d(self.model, self.config)
        count = len(turns)
        qpos = np.tile(initial["qpos"], (count, 1))
        qpos[:, 3:7] = 0
        qpos[:, 3] = np.cos(np.pi * turns)
        qpos[:, 5] = np.sin(np.pi * turns)
        qvel = np.zeros((count, self.model.nv))
        # Later states are intentionally faster than the retired thresholds.
        qvel[:, 0] = 1 + np.abs(turns)
        qvel[:, 4] = 4 + 5 * np.abs(turns)
        ctrl = np.tile(initial["ctrl"], (count, 1))
        episodes = np.zeros(count) if episodes is None else episodes
        save_roll_snapshots_3d(path, self.model, self.config, qpos=qpos, qvel=qvel,
            ctrl=ctrl, time_s=np.arange(count) * .02, episode_id=episodes,
            source_policy="synthetic-cycle-test-not-a-trained-policy",
            roll_phase_rad=np.asarray(turns) * 2 * np.pi + origin,
            roll_origin_phase_rad=np.broadcast_to(origin, (count,)))
        with np.load(path) as data:
            return dict(data)

    def test_exact_windows_and_high_speeds_are_preserved(self):
        for stage, low, high in (("brake_early", 1, 2), ("brake_later", 2, 4),
                                 ("brake_full", 1, np.inf)):
            with self.subTest(stage=stage):
                task = transition_curriculum_config_3d(stage)
                bank, report = load_roll_snapshots_3d(self.path, self.model, task, return_report=True)
                selected = (self.turns >= low) & (self.turns < high)
                for name in ("qpos", "qvel", "ctrl", "time_s", "episode_id"):
                    np.testing.assert_array_equal(bank[name], self.arrays[name][selected])
                self.assertTrue(report["coverage_complete"])
                self.assertFalse(report["velocities_modified"])
                self.assertGreater(report["linear_speed_m_s"]["min"], .35)
                self.assertGreater(report["angular_speed_rad_s"]["min"], 3.5)

    def test_bounds_are_inclusive_exclusive(self):
        path = Path(self.directory.name) / "boundaries.npz"
        self.save_bank(path, np.arange(1, 4.01, .125), origin=0.)
        bank = load_roll_snapshots_3d(path, self.model, transition_curriculum_config_3d("brake_early"))
        turns = bank["roll_phase_rad"] / (2 * np.pi)
        self.assertEqual(turns.min(), 1.)
        self.assertTrue(np.all(turns < 2.))

    def test_rocking_and_startup_never_count_as_a_completed_cycle(self):
        path = Path(self.directory.name) / "rocking.npz"
        self.save_bank(path, np.tile([0, .2, .45, .2, 0, -.2, 0], 10))
        with self.assertRaisesRegex(ValueError, "no ROLL snapshots"):
            load_roll_snapshots_3d(path, self.model, transition_curriculum_config_3d("brake_early"))

    def test_reverse_direction_and_nonzero_reset_origins(self):
        path = Path(self.directory.name) / "reverse.npz"
        arrays = self.save_bank(path, -self.turns, origin=-17.)
        task = replace(transition_curriculum_config_3d("brake_early"), snapshot_roll_direction=-1)
        bank = load_roll_snapshots_3d(path, self.model, task)
        np.testing.assert_array_equal(bank["qvel"], arrays["qvel"][8:16])
        with self.assertRaisesRegex(ValueError, "no ROLL snapshots"):
            load_roll_snapshots_3d(path, self.model, replace(task, snapshot_roll_direction=1))

    def test_episode_origins_do_not_leak_between_episodes(self):
        path = Path(self.directory.name) / "two_episodes.npz"
        turns = np.tile(self.turns, 2)
        origins = np.repeat([20., -100.], 40)
        self.save_bank(path, turns, origin=origins, episodes=np.repeat([0, 1], 40))
        bank, report = load_roll_snapshots_3d(path, self.model,
            transition_curriculum_config_3d("brake_early"), return_report=True)
        self.assertEqual(report["episodes"], 2)
        self.assertEqual(len(bank["qpos"]), 16)

    def test_sparse_missing_phase_or_missing_later_cycle_is_rejected(self):
        for turns in (np.arange(1, 2, .25) + .01, np.arange(2, 3, .125) + .01):
            stage = "brake_early" if turns[0] < 2 else "brake_later"
            path = Path(self.directory.name) / (stage + ".npz")
            self.save_bank(path, turns)
            with self.assertRaisesRegex(ValueError, "incomplete ROLL cycle/phase"):
                load_roll_snapshots_3d(path, self.model, transition_curriculum_config_3d(stage))
            _, report = load_roll_snapshots_3d(path, self.model,
                transition_curriculum_config_3d(stage), return_report=True, require_coverage=False)
            self.assertFalse(report["coverage_complete"])

    def test_tail_cannot_silently_replace_early_cycles_with_later_ones(self):
        task = replace(transition_curriculum_config_3d("brake_early"), snapshot_tail_fraction=.2)
        with self.assertRaisesRegex(ValueError, "no ROLL snapshots"):
            load_roll_snapshots_3d(self.path, self.model, task)

    def test_cycle_phase_sampling_balances_unequal_time_density(self):
        path = Path(self.directory.name) / "unbalanced.npz"
        turns = np.concatenate(([1.0625] * 20, np.arange(1, 3, .125) + .0625))
        self.save_bank(path, turns)
        bank, _ = load_roll_snapshots_3d(path, self.model,
            transition_curriculum_config_3d("brake_full"), return_report=True)
        prob = np.diff(np.r_[0, bank["sampling_cdf"]])
        cell = np.floor(turns * 8).astype(int)
        for c in np.unique(cell):
            self.assertAlmostEqual(float(prob[cell == c].sum()), 1 / 16, places=6)
        self.assertEqual(bank["sampling_cdf"][-1], 1.)

    def test_full_reports_partial_final_cycle_instead_of_silently_dropping_it(self):
        path = Path(self.directory.name) / "partial_final.npz"
        self.save_bank(path, np.arange(1, 2.5, .125) + .0625)
        bank, report = load_roll_snapshots_3d(path, self.model,
            transition_curriculum_config_3d("brake_full"), return_report=True)
        self.assertEqual(len(bank["qpos"]), 12)
        self.assertEqual(report["incomplete_cycles"], ["2"])

    def test_invalid_progress_and_reset_splice_rejected(self):
        for update in ({"roll_phase_rad": np.full(40, np.nan)},
                       {"roll_origin_phase_rad": np.arange(40)},
                       {"time_s": np.zeros(40)}, {"time_s": np.zeros(1)},
                       {"roll_progress_source": np.asarray("oscillator_phase")},
                       {"schema_version": np.asarray(99)}):
            with self.subTest(update=tuple(update)), self.assertRaises(ValueError):
                validate_roll_snapshots_3d({**self.arrays, **update}, self.model, self.config)

    def test_inspector_and_preflight_do_not_require_jax(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            inspect_main([str(self.path), "--stage", "brake_early"])
        self.assertEqual(json.loads(stream.getvalue())["selected_samples"], 8)
        broken = Path(self.directory.name) / "broken.npz"
        self.save_bank(broken, np.array([.1, .2]))
        out = Path(self.directory.name) / "no_output_on_bad_bank"
        with self.assertRaisesRegex(ValueError, "no ROLL snapshots"):
            train_main(["--stage", "brake_early", "--roll-snapshots", str(broken), "--out", str(out)])
        self.assertFalse(out.exists())

    def test_collector_preserves_reset_origin_despite_warmup_and_decimation(self):
        path = Path(self.directory.name) / "collected.npz"
        initial = walking_start_state_3d(self.model, self.config)
        class Env:
            config = self.config
            mj_model = self.model
            def reset(self, key):
                self.tick = 0
                return self.state()
            def state(self):
                return SimpleNamespace(obs=np.zeros(1), done=False,
                    info={"rolling_phase": 11. + self.tick * np.pi / 2},
                    pipeline_state=SimpleNamespace(**initial, time=self.tick * .02))
            def step(self, state, action):
                self.tick += 1
                return self.state()
        # No JAX execution: fake only the orchestration API to test collection.
        fake_jax = SimpleNamespace(jit=lambda f: f, device_get=lambda x: x,
            random=SimpleNamespace(PRNGKey=lambda k: k, fold_in=lambda k, n: k + n))
        with patch.dict("sys.modules", {"jax": fake_jax}):
            result = collect_roll_snapshots_3d(Env(), lambda obs, key: (np.zeros(12), {}),
                path, source_policy="fake-collector-fixture", episodes=2,
                steps_per_episode=12, warmup_steps=5, sample_every=2)
        with np.load(path) as bank:
            np.testing.assert_array_equal(bank["roll_origin_phase_rad"], np.full(8, 11.))
            turns = (bank["roll_phase_rad"] - bank["roll_origin_phase_rad"]) / (2 * np.pi)
            np.testing.assert_allclose(turns, np.tile([1.25, 1.75, 2.25, 2.75], 2))
            np.testing.assert_array_equal(bank["qvel"], np.tile(initial["qvel"], (8, 1)))
        self.assertEqual(result["schema_version"], 2)


class CycleConfigAndCheckpointTests(unittest.TestCase):
    def test_override_and_reused_config_do_not_keep_old_bounds(self):
        task = build_task(parse_args(["--stage", "brake_later", "--snapshot-min-turns", "3",
            "--snapshot-max-turns", "5", "--snapshot-roll-direction", "-1"]))
        self.assertEqual((task.snapshot_min_turns, task.snapshot_max_turns), (3, 5))
        full = transition_curriculum_config_3d("brake_full", task)
        self.assertEqual((full.snapshot_min_turns, full.snapshot_max_turns), (1, None))
        self.assertEqual(full.snapshot_roll_direction, -1)

    def test_invalid_windows_and_nonbrake_override_rejected(self):
        for options in ({"snapshot_min_turns": 0}, {"snapshot_min_turns": 1.5},
                        {"snapshot_max_turns": 1}, {"snapshot_max_turns": float("nan")},
                        {"snapshot_roll_direction": 0}, {"snapshot_phase_bins": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_transition_config_3d(replace(Transition3DConfig(), **options))
        with self.assertRaisesRegex(ValueError, "only meaningful for BRAKE"):
            build_task(parse_args(["--snapshot-min-turns", "2"]))

    def test_latest_completed_checkpoint_and_direct_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("9", "000000000100", "000000000101", "999.orbax-checkpoint-tmp"):
                step = root / name
                step.mkdir()
                (step / "_METADATA").write_text("{}")
                if name != "000000000101":
                    (step / "ppo_network_config.json").write_text("{}")
            expected = (root / "000000000100").resolve()
            self.assertEqual(resolve_transition_checkpoint(root), expected)
            self.assertEqual(resolve_transition_checkpoint(expected), expected)
            with self.assertRaisesRegex(ValueError, "incomplete checkpoint"):
                resolve_transition_checkpoint(root / "000000000101")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_transition_checkpoint(root / "missing")

    def test_empty_parent_cannot_start_training_from_random_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "no completed numeric"):
                train_main(["--stage", "deploy_near_stand", "--restore-checkpoint", str(root),
                            "--out", str(root / "output")])
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
