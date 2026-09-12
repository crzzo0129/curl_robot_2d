"""Standard-library-only sweep continuation checks; no trainer is imported."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_deploy_reward_sweep as sweep


class SweepResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="deploy_sweep_resume_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(sweep, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.directory = self.root / "results" / "old"
        self.directory.mkdir(parents=True)
        runs = []
        for i, (label, symmetry, rate) in enumerate(sweep.CASES):
            name = "old_" + label
            checkpoints = self.root / ("rollingquad_2_deploy_" + name + "_checkpoints")
            checkpoints.mkdir()
            for step in (9, 100 + i):
                (checkpoints / f"{step}.bin").write_bytes(b"placeholder for path selection only")
            runs.append(dict(name=name, gpu=str(i), symmetry_weight=symmetry,
                             action_rate_weight=rate, pid=None))
        manifest = dict(domain_randomization=True, terrain=False, collision_model="foot-spheres",
                        num_envs_per_run=512, batch_size_per_run=32, runs=runs)
        (self.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_selects_each_own_numeric_latest(self):
        _, _, sources = sweep.continuation_sources(self.directory)
        self.assertEqual([p.stem for p, _ in sources], ["100", "101", "102", "103"])
        self.assertEqual(len({p.parent for p, _ in sources}), 4)

    def test_missing_group_does_not_fall_back(self):
        _, _, sources = sweep.continuation_sources(self.directory)
        for path in sources[2][0].parent.glob("*.bin"):
            path.unlink()
        with self.assertRaises(FileNotFoundError):
            sweep.continuation_sources(self.directory)

    def test_preview_inherits_settings_without_launching(self):
        argv = ["launch", "--continue-from", str(self.directory), "--prefix", "new"]
        with patch("sys.argv", argv), patch.object(sweep.subprocess, "Popen") as launch:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                sweep.main()
            launch.assert_not_called()
        text = output.getvalue()
        self.assertEqual(text.count("--collision-model foot-spheres"), 4)
        self.assertEqual(text.count("--num-envs 512 --batch-size 32"), 4)
        for label, _, _ in sweep.CASES:
            self.assertIn("--run-name new_" + label, text)
        self.assertFalse((self.root / "results" / "new").exists())

    def test_symmetry_sweep_keeps_four_distinct_cases_on_resume(self):
        path = self.directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest['sweep'] = 'symmetry'
        for run, (_, fb, rate, lr, phase, balance) in zip(manifest['runs'], sweep.SYMMETRY_CASES):
            run.update(symmetry_weight=fb, action_rate_weight=rate, lr_symmetry_weight=lr,
                       trot_phase_weight=phase, cycle_balance_weight=balance)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        _, _, sources = sweep.continuation_sources(self.directory)
        self.assertEqual([p.stem for p, _ in sources], ['100', '101', '102', '103'])
        with patch('sys.argv', ['launch', '--continue-from', str(path), '--prefix', 'new']), \
                patch.object(sweep.subprocess, 'Popen') as launch:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                sweep.main()
            launch.assert_not_called()
        text = output.getvalue()
        self.assertEqual(text.count('--lr-symmetry-weight 0.01'), 2)
        self.assertEqual(text.count('--trot-phase-weight 0.05'), 2)
        self.assertEqual(text.count('--cycle-balance-weight 0.02'), 2)


if __name__ == "__main__":
    unittest.main()
