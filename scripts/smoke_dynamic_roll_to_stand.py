"""CPU/GPU runtime contract check; this does not certify learned recovery."""

import json
from dataclasses import replace
from pathlib import Path


def main():
    import jax
    import jax.numpy as jp
    import numpy as np
    from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig, transition_physics_profile_3d
    from curl_robot_2d_mjx.environment_transition_3d import make_brax_transition_env_3d
    task = transition_physics_profile_3d("cg12", Transition3DConfig(geometry="rollingquad_2_primitive",
        physics_timestep=0.001, dynamic_roll_to_stand=True, ready_hold_s=1.0, observation_noise_enabled=False))
    env = make_brax_transition_env_3d(task)
    print("[smoke] compiling primitive reset", flush=True)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    # Live takeover must preserve even high velocities and elapsed source time.
    data = state.pipeline_state.replace(qvel=state.pipeline_state.qvel.at[4].set(8.),
                                       time=jp.asarray(4.25))
    takeover = jax.jit(env.reset_from_roll_state)(data, jax.random.PRNGKey(1))
    for name in ("qpos", "qvel", "ctrl", "time"):
        np.testing.assert_array_equal(np.asarray(getattr(data, name)),
                                      np.asarray(getattr(takeover.pipeline_state, name)))
    assert int(takeover.info["mode"]) == 1  # unrestricted recovery, not BRAKE
    print("[smoke] high-speed live takeover state preserved; compiling step", flush=True)
    step = jax.jit(env.step)
    next_state = step(takeover, jp.zeros(12))
    assert int(next_state.info["mode"]) == 1
    assert np.isfinite(np.asarray(next_state.obs["state"])).all()
    assert np.isfinite(float(next_state.reward))
    # Run the nominal stand baseline through the complete hold window.
    # This is a baseline measurement, not an assertion that zero action succeeds.
    initial = state
    for index in range(env.ready_hold_steps + 20):
        state = step(state, jp.zeros(12))
        if bool(state.done):
            break
    report = dict(status="runtime_contract_passed", backend=jax.default_backend(),
        geometry=task.geometry, live_state_preserved=True, mesh_collision_used=False,
        baseline_steps=index + 1, baseline_success=float(state.metrics["transition_success"]),
        baseline_failed=float(state.metrics["failed"]),
        baseline_height=float(state.pipeline_state.qpos[2]),
        baseline_foot_contacts=float(state.metrics["foot_contact_count"]),
        required_stand_steps=env.ready_hold_steps,
        initial_height=float(initial.pipeline_state.qpos[2]), policy_trained=False)
    out = Path("results/roll_to_stand_runtime")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
