"""No-JAX CLI/config/orchestration tests; fixtures are NOT learned ROLL data."""

import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from curl_robot_2d_mjx.cem_reference import CEMReferenceConfig
from curl_robot_2d_mjx.config_3d import Rolling3DConfig
from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig
from curl_robot_2d_mjx.reward_3d import Rolling3DRewardConfig
from scripts import collect_transition_roll_snapshots as cli


class CollectionCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "params_best"
        self.checkpoint.write_bytes(b"synthetic checkpoint, never loaded with Brax")
        self.config = self.root / "training_config.json"
        self.out = self.root / "new" / "bank.npz"
        self.payload = {
            "task": asdict(Rolling3DConfig(explicit_phase_observation=True,
                floor_friction_scale=.9, body_mass_scale=1.2, actuator_gain_scale=.8,
                reset_pose="stand", stand_hold_s=.3, stand_to_compact_s=1.2)),
            "reference": asdict(CEMReferenceConfig(tuple([.1] * 8), 5., 2.,
                source="/a/different/machine/controller.json")),
            "reward": asdict(Rolling3DRewardConfig()),
            "hidden_layers": [256, 256, 128], "activation": "elu",
            "zero_residual_policy_init": True, "initial_policy_std": .1,
            "reflection_equivariant_policy": False,
        }
        self.payload = json.loads(json.dumps(self.payload))  # Same list/tuple representation as the saved JSON.
        self.write_config()

    def write_config(self):
        self.config.write_text(json.dumps(self.payload), encoding="utf-8")

    def args(self, *extra):
        return ["--roll-checkpoint", str(self.checkpoint), "--out", str(self.out), *extra]

    def test_dry_run_preserves_config_and_writes_nothing(self):
        stream = io.StringIO()
        with patch.object(cli, "configure_cloud_runtime", side_effect=AssertionError("no JAX runtime")), \
                contextlib.redirect_stdout(stream):
            cli.main(self.args("--dry-run"))
        report = json.loads(stream.getvalue())
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["task"], self.payload["task"])
        self.assertEqual(report["reference"], self.payload["reference"])
        self.assertEqual(report["steps_per_episode"], self.payload["task"]["episode_length"])
        self.assertEqual(len(report["checkpoint_sha256"]), 64)
        self.assertFalse(report["external_braking"])
        self.assertFalse(self.out.parent.exists())

    def test_missing_cloud_files_fail_before_runtime(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["--roll-checkpoint", str(self.root / "missing"), "--out", str(self.out)])

    def test_missing_required_network_metadata_is_not_guessed(self):
        for name in ("task", "reference", "hidden_layers", "activation", "zero_residual_policy_init"):
            changed = dict(self.payload)
            del changed[name]
            self.config.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "config missing"):
                cli.load_roll_config(self.config)

    def test_wrong_geometry_or_student_is_rejected(self):
        for changes in ({"geometry": "baseline"}, {"direct_effective_action": True}):
            self.payload["task"].update(changes)
            self.write_config()
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "requires rollingquad_2"):
                cli.load_roll_config(self.config)

    def test_invalid_network_architecture_is_rejected(self):
        for changes in ({"activation": "unknown"}, {"hidden_layers": [0]},
                        {"zero_residual_policy_init": "true"}, {"initial_policy_std": float("nan")},
                        {"zero_residual_policy_init": False, "reflection_equivariant_policy": True}):
            self.config.write_text(json.dumps({**self.payload, **changes}), encoding="utf-8")
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                cli.load_roll_config(self.config)

    def test_never_silently_extends_source_timeout(self):
        with self.assertRaisesRegex(ValueError, "exceeds saved ROLL episode_length"):
            cli.main(self.args("--steps-per-episode", "999999", "--dry-run"))
        self.assertFalse(self.out.parent.exists())

    def test_policy_loader_selects_saved_architecture_and_uses_restored_weights(self):
        from scripts import train_mjx_3d_residual_ppo, train_mjx_ppo
        restored, network = object(), object()
        networks = SimpleNamespace(make_inference_fn=lambda net:
            lambda params, deterministic: (net, params, deterministic))
        acme = ModuleType("brax.training.acme")
        acme.running_statistics = SimpleNamespace(normalize=lambda obs, stats: obs)
        ppo = ModuleType("brax.training.agents.ppo")
        ppo.networks = networks
        fake_modules = {"brax": ModuleType("brax"), "brax.training": ModuleType("brax.training"),
                        "brax.training.agents": ModuleType("brax.training.agents"),
                        "brax.training.acme": acme, "brax.training.agents.ppo": ppo}
        for custom in (True, False):
            payload = {**self.payload, "zero_residual_policy_init": custom,
                       "reflection_equivariant_policy": custom}
            with self.subTest(custom=custom), patch.dict("sys.modules", fake_modules), \
                 patch.object(train_mjx_3d_residual_ppo, "_zero_centered_residual_network_factory",
                              return_value=lambda *args, **kwargs: network) as residual_factory, \
                 patch.object(train_mjx_ppo, "_network_factory",
                              return_value=lambda *args, **kwargs: network) as default_factory:
                result = cli.make_frozen_roll_policy(SimpleNamespace(observation_size=65, action_size=8),
                                                     payload, restored)
            self.assertEqual(result, (network, restored, True))
            self.assertEqual(residual_factory.call_count, int(custom))
            self.assertEqual(default_factory.call_count, int(not custom))

    def test_output_and_report_are_never_overwritten(self):
        self.out.parent.mkdir()
        for path in (self.out, self.out.with_suffix(".summary.json")):
            path.write_text("preserve", encoding="utf-8")
            with self.subTest(path=path), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(self.args())
            self.assertEqual(path.read_text(), "preserve")
            path.unlink()  # Only this test's own temporary fixture.

    def run_mocked_collection(self, complete=True):
        import mujoco
        from curl_robot_2d_mjx.environment_transition_3d import TRANSITION_MODEL_PATH_3D
        from curl_robot_2d_mjx.transition_initialization_3d import save_roll_snapshots_3d, walking_start_state_3d
        model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        frozen_policy, restored_params = object(), object()
        env = SimpleNamespace(mj_model=model, observation_size=65, action_size=8)
        def collect(actual_env, actual_policy, path, **kwargs):
            self.assertIs(actual_env, env)
            self.assertIs(actual_policy, frozen_policy)
            self.assertIn("#sha256=", kwargs["source_policy"])
            turns = np.arange(40 if complete else 5) / 8 + .0625
            n = len(turns)
            initial = walking_start_state_3d(model, Transition3DConfig())
            save_roll_snapshots_3d(path, model, Transition3DConfig(),
                **{k: np.tile(initial[k], (n, 1)) for k in ("qpos", "qvel", "ctrl")},
                time_s=np.arange(n) * .02, episode_id=np.zeros(n),
                source_policy=kwargs["source_policy"], roll_phase_rad=turns * 2 * np.pi,
                roll_origin_phase_rad=np.zeros(n))
            kwargs["progress_fn"]({"episode": 0, "samples": n})
            return {"schema_version": 2, "samples": n, "external_braking": False}
        fake_io = ModuleType("brax.io")
        fake_io.model = SimpleNamespace(load_params=lambda path: restored_params)
        with patch.dict("sys.modules", {"brax": ModuleType("brax"), "brax.io": fake_io}), \
             patch.object(cli, "configure_cloud_runtime"), \
             patch("curl_robot_2d_mjx.environment_3d.make_brax_env_3d", return_value=env) as make_env, \
             patch.object(cli, "make_frozen_roll_policy", return_value=frozen_policy) as make_policy, \
             patch("curl_robot_2d_mjx.transition_initialization_3d.collect_roll_snapshots_3d", side_effect=collect), \
             contextlib.redirect_stdout(io.StringIO()):
            if complete:
                cli.main(self.args())
            else:
                with self.assertRaisesRegex(SystemExit, "coverage is incomplete"):
                    cli.main(self.args())
        self.assertEqual(asdict(make_env.call_args.args[0]), self.payload["task"])
        self.assertIs(make_policy.call_args.args[2], restored_params)
        self.assertTrue(self.out.is_file())
        return json.loads(self.out.with_suffix(".summary.json").read_text(encoding="utf-8"))

    def test_full_cli_writes_bank_and_per_course_report_without_jax(self):
        report = self.run_mocked_collection()
        self.assertEqual(report["status"], "ok")
        self.assertTrue(all(row["coverage_complete"] for row in report["stage_reports"].values()))

    def test_incomplete_collection_preserves_diagnostics_and_fails_gate(self):
        report = self.run_mocked_collection(complete=False)
        self.assertEqual(report["status"], "insufficient_coverage")
        self.assertFalse(report["stage_reports"]["brake_early"]["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
