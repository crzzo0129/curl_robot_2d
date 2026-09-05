"""Stage-one contracts: no rolling checkpoint, separate success and horizon."""
import tempfile
import unittest
import io
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

from curl_robot_2d_mjx.compact_startup_3d import (
    CompactReachConfig, CompactStartupConfig, compact_target,
)
from curl_robot_2d_mjx.autonomous_startup_3d import confirmation_update, gate_errors
from scripts.train_mjx_3d_startup_ppo import parse_args, build_inputs, main


class CompactReachTest(unittest.TestCase):
    def test_teacher_modes_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            for extra in ([], ['--compact-only', '--teacher', 'unused.bin'],
                          ['--compact-only', '--teacher-config', 'unused.json']):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    parse_args(['--out', str(Path(tmp) / 'run'), *extra])

    def test_independent_horizon_and_contiguous_confirmation(self):
        cfg = CompactReachConfig()
        cfg.validate(.02)
        self.assertEqual(cfg.episode_steps(.02), 150)
        self.assertEqual(CompactStartupConfig().episode_steps(.02), 650)
        self.assertAlmostEqual(cfg.confirmation_steps * .02, .10)
        count = 0
        for eligible in (True, True, True, True, False, True):
            count = confirmation_update(np, 0, count, 0, eligible)
        self.assertEqual(count, 1)

    def test_stage_one_needs_no_teacher_and_is_not_rolling_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['--compact-only', '--out', str(Path(tmp) / 'run'), '--dry-run']
            args = parse_args(argv)
            self.assertIsNone(args.teacher)
            task, _, _, cfg, bank, _, _ = build_inputs(args)
            self.assertEqual(task.geometry, 'rollingquad_2')
            self.assertEqual(task.reset_velocity_noise, 0)
            self.assertEqual(cfg.confirmation_steps, 5)
            self.assertEqual(bank['qpos'].shape, (1, 19))
            with patch('builtins.print'):
                payload = main(argv)
            self.assertEqual(payload['episode_length'], 150)
            self.assertIsNone(payload['teacher_sha256'])
            self.assertFalse(payload['rolling_continuation_evaluated'])
            self.assertFalse(payload['first_command_jump_gate_applied'])
            self.assertFalse((Path(tmp) / 'run').exists())

    def test_velocity_gate_rejects_pose_only_completion(self):
        import mujoco
        from curl_robot_2d_mjx.environment_3d import model_path_3d
        model = mujoco.MjModel.from_xml_path(str(model_path_3d('rollingquad_2')))
        bank = compact_target(model)
        q, v = bank['qpos'][0], bank['qvel'][0].copy()
        self.assertLess(gate_errors(np, q, v, 0., bank, CompactReachConfig()).max(), .01)
        for speed in (-.03, .03):
            v[0] = speed
            self.assertGreater(gate_errors(np, q, v, 0., bank, CompactReachConfig()).max(), 1)


if __name__ == '__main__':
    unittest.main()
