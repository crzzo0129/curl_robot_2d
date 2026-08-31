"""Real JIT contract tests, skipped when the cloud training stack is absent."""

import importlib.util
import json
import unittest

import numpy as np


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("jax", "brax", "mujoco")),
    "JAX/Brax/MJX stack required; run on cloud",
)
class TransitionMJXTest(unittest.TestCase):
    def test_real_brax_actor_export_parity(self):
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
        from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig
        from curl_robot_2d_mjx.deployment_transition_3d import transition_controller_metadata_3d
        from curl_robot_2d_mjx.environment_transition_3d import TRANSITION_MODEL_PATH_3D
        from scripts.train_mjx_3d_transition_ppo import make_transition_networks
        from scripts.export_transition_rtneural import convert_transition
        from scripts.export_rtneural import _activation, _dense_layers, _run_layers
        sizes = {"state": 720, "privileged_state": 86}
        nets = make_transition_networks(sizes, 12, running_statistics.normalize,
                                        hidden_layers=(16, 16))
        normalizer = running_statistics.init_state({k: jp.zeros(v) for k, v in sizes.items()})
        obs = {k: jax.random.normal(jax.random.PRNGKey(v), (32, v))
               for k, v in sizes.items()}
        normalizer = running_statistics.update(normalizer, obs)
        actor = nets.policy_network.init(jax.random.PRNGKey(10))
        params = (normalizer, actor)
        policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
        key = jax.random.PRNGKey(20)
        configured_precision = jax.config.jax_default_matmul_precision
        # Diagnostic only: a GPU's default/high matmul may use reduced
        # precision for float32 inputs. That is not an FP32 export reference.
        configured_output = np.asarray(jax.jit(policy)(obs, key)[0])
        # The scope must cover JIT tracing AND execution, not just creating the
        # Python callable. Do not change global training/MJX precision or relax
        # the tolerance to hide a wrong normalization/activation/export.
        with jax.default_matmul_precision("highest"):
            expected = np.asarray(jax.jit(policy)(obs, key)[0])
            normalized_jax = np.asarray(
                jax.jit(running_statistics.normalize)(obs, normalizer)["state"])
        self.assertEqual(jax.config.jax_default_matmul_precision, configured_precision)

        raw = np.asarray(obs["state"], dtype=np.float32)
        normalized_numpy = (
            raw - np.asarray(normalizer.mean["state"], dtype=np.float32)
        ) / np.asarray(normalizer.std["state"], dtype=np.float32)
        # Independent unfused reference: normalize -> original dense layers ->
        # ELU -> tanh(location). This isolates exporter folding from Brax math.
        dense = _dense_layers(actor)
        unfused = normalized_numpy
        for index, (_, kernel, bias) in enumerate(dense):
            unfused = unfused @ kernel + bias
            unfused = (np.tanh(unfused[..., :12]) if index == len(dense) - 1
                       else _activation("elu", unfused))
        model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        config = transition_controller_metadata_3d(model, Transition3DConfig())
        exported = convert_transition(params, config)
        actual = _run_layers(raw, exported["layers"])
        comparisons = {
            "normalization": (normalized_numpy, normalized_jax),
            "unfused_numpy_vs_brax_fp32": (unfused, expected),
            "export_vs_unfused_numpy": (actual, unfused),
            "export_vs_brax_fp32": (actual, expected),
        }
        diagnostics = {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "configured_matmul_precision": configured_precision,
            "reference_matmul_precision": "highest",
            "configured_vs_fp32_max_abs": float(np.max(np.abs(configured_output - expected))),
            "export_vs_configured_max_abs": float(np.max(np.abs(actual - configured_output))),
            **{name + "_max_abs": float(np.max(np.abs(left - right)))
               for name, (left, right) in comparisons.items()},
        }
        print("\n[transition export parity] " + json.dumps(diagnostics, sort_keys=True), flush=True)
        for name, (left, right) in comparisons.items():
            with self.subTest(comparison=name):
                self.assertTrue(np.isfinite(left).all() and np.isfinite(right).all(),
                                f"nonfinite {name}: {diagnostics}")
                np.testing.assert_allclose(left, right, atol=2e-5, rtol=2e-5,
                                           err_msg=f"{name}: {diagnostics}")

    def test_reset_step_and_live_takeover(self):
        import jax
        import jax.numpy as jp
        from curl_robot_2d_mjx.config_transition_3d import (
            Transition3DConfig, TransitionMode3D,
        )
        from curl_robot_2d_mjx.environment_transition_3d import (
            make_brax_transition_env_3d,
        )
        from curl_robot_2d_mjx.transition_initialization_3d import walking_start_state_3d
        task = Transition3DConfig(observation_noise_enabled=False)
        env = make_brax_transition_env_3d(task)
        state = jax.jit(env.reset)(jax.random.PRNGKey(1))
        expected = walking_start_state_3d(env.mj_model, task)
        np.testing.assert_allclose(state.pipeline_state.qpos, expected["qpos"], atol=1e-7)
        np.testing.assert_array_equal(state.pipeline_state.qvel, expected["qvel"])
        self.assertEqual(state.obs["state"].shape, (720,))
        self.assertEqual(state.obs["privileged_state"].shape, (86,))
        old_obs = state.obs["state"]
        state = jax.jit(env.step)(state, jp.full(12, .05))
        self.assertTrue(np.isfinite(state.reward))
        np.testing.assert_array_equal(state.obs["state"][36:], old_obs[:-36])
        np.testing.assert_allclose(state.obs["state"][24:36], .05)
        np.testing.assert_array_equal(state.info["actor_history"], state.obs["state"])
        # Full live data handoff is exact even though root velocity is nonzero.
        data = state.pipeline_state.replace(
            qvel=state.pipeline_state.qvel.at[0].set(.3).at[4].set(3.0))
        captured = jax.jit(env.reset_from_roll_state)(data, jax.random.PRNGKey(2))
        np.testing.assert_array_equal(captured.pipeline_state.qpos, data.qpos)
        np.testing.assert_array_equal(captured.pipeline_state.qvel, data.qvel)
        np.testing.assert_array_equal(captured.pipeline_state.ctrl, data.ctrl)
        self.assertEqual(int(captured.info["mode"]), int(TransitionMode3D.BRAKE))
        stepped = jax.jit(env.step)(captured, jp.full(12, .1))
        self.assertTrue(np.isfinite(stepped.reward))
        hot = jax.jit(env.reset_from_roll_state)(
            data, jax.random.PRNGKey(2), state.obs["state"], jp.full(12, .05))
        np.testing.assert_array_equal(hot.obs["state"][36:], state.obs["state"][:-36])
        np.testing.assert_allclose(hot.obs["state"][24:36], .05)

    def test_privileged_velocity_never_enters_actor(self):
        import jax
        import jax.numpy as jp
        from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig
        from curl_robot_2d_mjx.environment_transition_3d import make_brax_transition_env_3d
        env = make_brax_transition_env_3d(Transition3DConfig(observation_noise_enabled=False))
        state = jax.jit(env.reset)(jax.random.PRNGKey(1))
        data = state.pipeline_state
        changed = data.replace(qvel=data.qvel.at[:3].set(.7).at[6:].set(1.5))
        # Same pose, gyro, command, previous action and history. Only privileged
        # velocities differ; do not advance physics between these observations.
        reset = jax.jit(env.reset_from_roll_state)
        first = reset(data, jax.random.PRNGKey(2))
        second = reset(changed, jax.random.PRNGKey(2))
        np.testing.assert_array_equal(first.obs["state"], second.obs["state"])
        self.assertFalse(np.array_equal(first.obs["privileged_state"],
                                        second.obs["privileged_state"]))


if __name__ == "__main__":
    unittest.main()
