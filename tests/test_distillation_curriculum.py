"""Lightweight cloud checks; no JAX, MuJoCo, training, or simulation."""
import copy
import unittest
import tempfile
import numpy as np

from curl_robot_2d_mjx.distillation_curriculum import (
    continuation_decision, low_speed_snapshot_indices,
)
from scripts.run_rolling_low_speed_recovery import distill_command, ppo_command
from pathlib import Path
from curl_robot_2d_mjx.distillation_eval_snapshots import (
    evaluation_environment_seed, save_snapshot_arrays, load_snapshot_arrays,
)


class CurriculumTest(unittest.TestCase):
    def test_sampling_preserves_joint_speed_and_turn_distribution(self):
        forward = np.repeat([.45, .60, .75], 3000)
        yaw = np.tile(np.repeat([0., .05, -.05], 1000), 3)
        indices, summary = low_speed_snapshot_indices(
            forward, yaw, speed_min=.4111, speed_max=.811,
            straight_fraction=.4, seed=27)
        self.assertEqual(len(indices), len(forward))
        self.assertLess(abs(np.mean(forward[indices] < .5444) - .6), .02)
        self.assertLess(abs(np.mean(yaw[indices] == 0) - .4), .02)
        self.assertLess(abs(np.mean((forward[indices] < .5444) & (yaw[indices] < 0)) - .18), .02)
        self.assertAlmostEqual(sum(g['probability'] for g in summary['groups'].values()), 1.)
        repeated, _ = low_speed_snapshot_indices(forward, yaw, speed_min=.4111,
                speed_max=.811, straight_fraction=.4, seed=27)
        np.testing.assert_array_equal(indices, repeated)

    def test_missing_group_refuses_silent_distribution_change(self):
        with self.assertRaisesRegex(ValueError, 'No snapshots'):
            low_speed_snapshot_indices([.45, .6, .75], [0, 0, 0],
                    speed_min=.4111, speed_max=.811, straight_fraction=.4, seed=0)

    def report(self):
        group = {'episodes': 30, 'success_rate': .7, 'full_horizon_rate': .9,
                 'forward_mae_m_s': .16, 'yaw_mae_rad_s': .05}
        return {'overall': dict(group), 'by_speed': {s: dict(group) for s in ('low', 'medium', 'high')},
                'speed_bin_edges_m_s': [.4111, .5444, .6777, .811], 'criteria': {'horizon': 10},
                'per_episode': [{'forward_command_m_s': .45, 'yaw_command_rad_s': 0,
                                 'teacher_warmup_steps': 100}]}

    def test_low_gain_cannot_hide_high_speed_regression(self):
        baseline = self.report()
        candidate = copy.deepcopy(baseline)
        candidate['by_speed']['low'].update(success_rate=.8, forward_mae_m_s=.12)
        self.assertTrue(continuation_decision(baseline, baseline, candidate)['select'])
        candidate['by_speed']['high']['success_rate'] = .6
        self.assertFalse(continuation_decision(baseline, baseline, candidate)['safe'])

    def test_unchanged_policy_is_not_selected_and_nonfinite_is_rejected(self):
        baseline = self.report()
        self.assertFalse(continuation_decision(baseline, baseline, baseline)['select'])
        candidate = copy.deepcopy(baseline)
        candidate['overall']['yaw_mae_rad_s'] = float('nan')
        self.assertFalse(continuation_decision(baseline, baseline, candidate)['safe'])

    def test_changed_command_panel_is_rejected(self):
        baseline = self.report()
        candidate = copy.deepcopy(baseline)
        candidate['per_episode'][0]['yaw_command_rad_s'] = .05
        self.assertFalse(continuation_decision(baseline, baseline, candidate)['safe'])

    def test_same_commands_with_different_physics_are_rejected(self):
        baseline = self.report()
        baseline['initial_state_sha256'] = 'original-state'
        candidate = copy.deepcopy(baseline)
        candidate['initial_state_sha256'] = 'different-state'
        self.assertFalse(continuation_decision(baseline, baseline, candidate)['safe'])

    def test_eval_seed_does_not_depend_on_training_seed(self):
        self.assertEqual(evaluation_environment_seed(1, 123), evaluation_environment_seed(99, 123))
        self.assertEqual(evaluation_environment_seed(99, 123, 7), 7)
        self.assertEqual(evaluation_environment_seed(99, None), 99)

    def test_snapshot_roundtrip_and_contract_rejection(self):
        # A loader must replace template values, preserve exact dtypes/history,
        # and reject a changed simulation contract instead of mixing state pools.
        arrays = [np.array([[1., 2.], [3., 4.]], dtype=np.float32),
                  np.array([111, 299], dtype=np.int32)]
        paths = ['state.qpos', 'state.step_count']
        contract = {'seed': 123, 'envs': 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'states.npz'
            manifest = save_snapshot_arrays(path, arrays, paths, contract)
            loaded, metadata = load_snapshot_arrays(path, [np.zeros_like(a) for a in arrays], paths, contract)
            self.assertEqual(manifest, metadata)
            for a, b in zip(arrays, loaded):
                np.testing.assert_array_equal(a, b)
            with self.assertRaisesRegex(ValueError, 'contract/schema'):
                load_snapshot_arrays(path, arrays, paths, {'seed': 456, 'envs': 2})
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                load_snapshot_arrays(path, [arrays[0].astype(np.float64), arrays[1]], paths, contract)
            with self.assertRaises(FileExistsError):
                save_snapshot_arrays(path, arrays, paths, contract)

    def test_profile_keeps_calibration_and_eval_uniform(self):
        saved = {'hidden_layers': [512, 256, 128], 'steering_calibration': 'calibration.json',
                 'num_devices': 4, 'minimum_closed_loop_turns': 5, 'eval_envs': 256,
                 'eval_seed': 123, 'seed': 45}
        train = distill_command(saved, 'student', Path('train'), seed=1, evaluation=False, chunk_steps=500)
        evaluate = distill_command(saved, 'student', Path('eval'), seed=1, evaluation=True, chunk_steps=500)
        self.assertIn('low_speed_focus', train)
        self.assertNotIn('--dagger-snapshot-sampling', evaluate)
        self.assertIn('calibration.json', evaluate)
        another_seed = distill_command(saved, 'student', Path('other'), seed=999,
                evaluation=False, chunk_steps=500, snapshot_cache=Path('shared.npz'))
        self.assertEqual(train[train.index('--eval-environment-seed') + 1],
                         another_seed[another_seed.index('--eval-environment-seed') + 1])
        self.assertIn('shared.npz', another_seed)
        actor = ppo_command(saved, 'student', Path('run'), 'actor')
        self.assertIn('calibration.json', actor)
        self.assertEqual(actor[actor.index('--restore-ppo') + 1], str(Path('run/critic/params_final')))
        self.assertEqual(actor[actor.index('--max-learning-rate') + 1], '0.000003')


if __name__ == '__main__':
    unittest.main()
