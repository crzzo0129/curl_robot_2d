"""Continuous MJX student -> CEM (>=5 s, then +90 deg) -> absolute stand policy.

Run from the project root on the training runtime. Both actors use their own
normalizers and action/history contracts. One model and full MJX state are used
throughout; there are no snapshot splices, pose edits or velocity resets.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path


def pitch_from_quaternion(q, xp=None):
    """Project torso rotation onto the sagittal plane, matching handoff banks."""
    w, x, y, z = q
    atan2 = math.atan2 if xp is None else xp.arctan2
    return atan2(2 * (x * z - w * y), 1 - 2 * (x * x + y * y))


def pitch_gate(elapsed, pitch, *, minimum=5.0, target=math.pi / 2,
               tolerance=math.pi / 180, xp=None):
    trig = math if xp is None else xp
    atan2 = math.atan2 if xp is None else xp.arctan2
    error = atan2(trig.sin(pitch - target), trig.cos(pitch - target))
    return (elapsed >= minimum) & (abs(error) <= tolerance)


def lift_cem_targets(planar, compact_ctrl):
    """CEM drives eight sagittal joints; retain the reference's ABD calibration."""
    fh, fk, rh, rk = planar
    return (compact_ctrl[0], fh, fk, compact_ctrl[3], fh, fk,
            compact_ctrl[6], rh, rk, compact_ctrl[9], rh, rk)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-params", type=Path, default=Path(
        "results/stand_to_roll_symmetry_student/student_params"))
    p.add_argument("--stand-params", type=Path, default=Path(
        "results/roll_to_stand_absolute_guided_hold_robust_v1/brake_full/params_final"))
    p.add_argument("--student-config", type=Path,
                   help="Original training_config.json; otherwise resolved from student_source.json")
    p.add_argument("--bc-params", type=Path, default=Path("results/stand_to_roll_startup/bc/bc_params"))
    p.add_argument("--cem-data", type=Path, default=Path("results/cem_cycle_data/cem_cycles.npz"))
    p.add_argument("--reference", type=Path, default=Path(
        "results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/"
        "01_zero_contact_speed_refine/best_phase_controller.json"))
    p.add_argument("--out", type=Path, default=Path("results/stand_roll_stand_video"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cem-seconds", type=float, default=5.0)
    p.add_argument("--pitch-tolerance-deg", type=float, default=1.0)
    p.add_argument("--max-cem-seconds", type=float, default=30.0,
                   help="Failure timeout, including the initial 5 s; CEM stays active while waiting")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--mujoco-gl", choices=("egl", "osmesa", "glfw"), default="egl")
    p.add_argument("--preflight", action="store_true", help="Check assets/configs without importing JAX")
    args = p.parse_args(argv)
    if not (math.isfinite(args.cem_seconds) and args.cem_seconds >= 5):
        p.error("--cem-seconds must be finite and at least 5")
    if not (math.isfinite(args.max_cem_seconds) and args.max_cem_seconds > args.cem_seconds):
        p.error("--max-cem-seconds must exceed --cem-seconds")
    if not (math.isfinite(args.pitch_tolerance_deg) and 0 < args.pitch_tolerance_deg <= 10):
        p.error("--pitch-tolerance-deg must be in (0, 10]")
    if args.fps <= 0:
        p.error("--fps must be positive")
    return args


def load_assets(args):
    source_path = args.student_params.parent / "student_source.json"
    stand_config_path = args.stand_params.parent / "training_config.json"
    required = [args.student_params, args.stand_params, source_path,
                stand_config_path, args.bc_params, args.cem_data, args.reference]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required cloud assets missing:\n" + "\n".join(missing))
    source = read_json(source_path)
    config_path = args.student_config
    if config_path is None:
        checkpoint = Path(source["checkpoint"])
        config_path = next((p / "training_config.json" for p in (checkpoint, *checkpoint.parents)
                            if (p / "training_config.json").is_file()), None)
    if config_path is None:
        raise FileNotFoundError("Original student training config not found; provide --student-config")
    if digest(config_path) != source["training_config_sha256"]:
        raise ValueError("Student training configuration hash mismatch")
    if digest(args.bc_params) != source["bc_sha256"]:
        raise ValueError("BC normalizer hash mismatch")
    student_config = read_json(config_path)
    stand_config = read_json(stand_config_path)
    if not stand_config["task"]["dynamic_roll_to_stand"] or stand_config["task"]["handcrafted_reference_residual"]:
        raise ValueError("Expected the absolute dynamic roll-to-stand policy")
    return source, student_config, stand_config, {
        str(p): digest(p) for p in [*required, config_path]}


def use_common_model(student_env, stand_env, np):
    """Use the stand training model for ALL stages, after checking topology.

    Solver settings can differ between the two training runs. Choosing one
    explicitly avoids changing contact physics at either handoff.
    """
    a, b = student_env.mj_model, stand_env.mj_model
    for name in ("nq", "nv", "nu", "nbody", "ngeom", "nsite", "njnt"):
        if getattr(a, name) != getattr(b, name):
            raise ValueError(f"Incompatible actor model topology: {name}")
    for name in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
                 "jnt_qposadr", "jnt_dofadr", "jnt_axis", "jnt_pos", "jnt_range",
                 "actuator_trnid", "geom_bodyid", "geom_pos", "geom_quat", "site_pos"):
        if not np.allclose(getattr(a, name), getattr(b, name), atol=1e-7, rtol=1e-6):
            raise ValueError(f"Incompatible actor model morphology: {name}")
    if not np.array_equal(a.names, b.names):
        raise ValueError("Model object names/order differ")
    if not np.array_equal(student_env.controller_qpos_indices, stand_env.joint_qpos_indices):
        raise ValueError("Actor joint ordering differs")
    student_env.mj_model = b
    student_env.mjx_model = stand_env.sys
    student_env.base_data = stand_env.base_data
    student_env.physics_timestep = float(b.opt.timestep)
    repeats = student_env.config.control_timestep / b.opt.timestep
    if not np.isclose(repeats, round(repeats)):
        raise ValueError("Student period must be a multiple of the shared physics timestep")
    student_env.action_repeat = round(repeats)


def render_video(model, rows, output, fps):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    # Streaming export; actual simulator timestamps also cover a partial CEM
    # policy interval when the angle gate fires between two 20 ms ticks.
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("torso").id
    camera.distance, camera.azimuth, camera.elevation = 1.25, 105, -18
    times = np.asarray([r["time_s"] for r in rows])
    sample_times = np.append(np.arange(times[0], times[-1], 1 / fps), times[-1])
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8) as writer:
            for t in sample_times:
                index = min(len(rows) - 1, max(0, int(np.searchsorted(times, t, side="right") - 1)))
                row = rows[index]
                data.qpos[:] = row["qpos"]
                data.qvel[:] = row["qvel"]
                data.ctrl[:] = row["ctrl"]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((12, 12, 590, 58), fill=(20, 25, 30))
                draw.text((24, 24), f'{row["stage"]}    t={t-times[0]:.2f}s    pitch={row["pitch_deg"]:+.1f} deg', fill="white")
                writer.append_data(np.asarray(frame))
                if index == len(rows) - 1:
                    frame.save(output.with_name("final_frame.png"))


def main(argv=None):
    args = parse_args(argv)
    source, student_config, stand_config, hashes = load_assets(args)
    if args.preflight:
        print(json.dumps({"status": "assets_verified", "sha256": hashes}, indent=2))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Choose an empty --out directory")
    from curl_robot_2d_mjx.runtime import configure_cloud_runtime
    configure_cloud_runtime(preallocate=False, mujoco_gl=args.mujoco_gl)
    import jax
    import jax.numpy as jp
    import numpy as np
    from mujoco import mjx
    from brax.io import model as model_io
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from curl_robot_2d_mjx.config_stand_to_roll import StandToRollConfig
    from curl_robot_2d_mjx.config_transition_3d import Transition3DConfig
    from curl_robot_2d_mjx.environment_stand_to_roll_3d import make_stand_to_roll_env_3d
    from curl_robot_2d_mjx.environment_transition_3d import make_brax_transition_env_3d
    from curl_robot_2d_mjx.stand_to_roll_training import preprocess_observation
    from curl_robot_2d_mjx.cem_reference import CEMReferenceGeometry, load_cem_reference, reference_action, advance_oscillator
    from scripts.train_mjx_3d_transition_ppo import make_transition_networks

    student_task = replace(StandToRollConfig(**source["task"]),
        observation_noise_enabled=False, snapshot_reset_probability=0.0,
        reset_alpha_min=1.0, reset_alpha_max=1.0)
    # Only construction/reset sampling is changed: the live state is handed
    # directly to reset_from_roll_state; no training snapshot bank is needed.
    stand_task = replace(Transition3DConfig(**stand_config["task"]),
        curriculum_stage="walking_start", roll_snapshots_path=None,
        observation_noise_enabled=False, domain_randomization=False,
        push_acceleration_m_s2=0.0)
    stand_env = make_brax_transition_env_3d(stand_task)
    student_env = make_stand_to_roll_env_3d(student_task, matcher_npz=args.cem_data, seed=args.seed)
    use_common_model(student_env, stand_env, np)
    normalizer, _ = model_io.load_params(args.bc_params)
    def preprocess(obs, unused):
        return preprocess_observation(jp, obs, normalizer)
    hidden = tuple(student_config["arguments"]["hidden_layers"])
    student_nets = ppo_networks.make_ppo_networks(720, 12,
        preprocess_observations_fn=preprocess, policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=hidden, activation=jax.nn.elu, distribution_type="tanh_normal")
    student_policy = jax.jit(ppo_networks.make_inference_fn(student_nets)(
        model_io.load_params(args.student_params), deterministic=True))
    stand_nets = make_transition_networks(stand_env.observation_size, 12,
        running_statistics.normalize, hidden_layers=stand_config["training"]["hidden_layers"])
    stand_policy = jax.jit(ppo_networks.make_inference_fn(stand_nets)(
        model_io.load_params(args.stand_params), deterministic=True))
    student_step, stand_step = jax.jit(student_env.step), jax.jit(stand_env.step)
    forward = jax.jit(lambda d: mjx.forward(stand_env.sys, d))
    rows, events = [], []
    def record(data, stage):
        qpos, qvel = np.asarray(data.qpos), np.asarray(data.qvel)
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
            raise RuntimeError("Nonfinite simulation state")
        rows.append(dict(time_s=float(data.time), qpos=qpos.copy(), qvel=qvel.copy(),
            ctrl=np.asarray(data.ctrl).copy(), stage=stage,
            pitch_deg=math.degrees(pitch_from_quaternion(qpos[3:7]))))
    def event(name, data, **extra):
        item = dict(event=name, time_s=float(data.time),
            pitch_deg=math.degrees(pitch_from_quaternion(np.asarray(data.qpos)[3:7])), **extra)
        events.append(item)
        print(json.dumps(item), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", sha256=hashes, seed=args.seed,
        shared_model=str(stand_env.model_path), shared_physics=stand_config["task"]["physics_profile"],
        student_training_physics=student_task.physics_profile,
        note="All stages run on the nominal roll-to-stand training model; no model switch at handoff.",
        cem_minimum_s=args.cem_seconds, pitch_target_deg=90,
        pitch_tolerance_deg=args.pitch_tolerance_deg, events=events)
    try:
        print("Compiling student simulation...", flush=True)
        state = jax.jit(student_env.reset)(jax.random.PRNGKey(args.seed))
        record(state.pipeline_state, "STAND TO ROLL")
        for i in range(student_task.episode_length):
            action, _ = student_policy(state.obs, jax.random.PRNGKey(args.seed + i + 1))
            state = student_step(state, action)
            record(state.pipeline_state, "STAND TO ROLL")
            if float(state.metrics["failed"]) > 0:
                raise RuntimeError("Stand-to-roll failed before CEM handoff")
            if bool(state.info["captured"]):
                break
            if bool(state.done):
                raise RuntimeError("Stand-to-roll ended without capture")
        if not bool(state.info["captured"]):
            raise RuntimeError("Stand-to-roll capture timeout")
        data = forward(state.pipeline_state)
        phase, distance = jax.jit(student_env._match)(data)
        body_phase = jp.asarray(-pitch_from_quaternion(np.asarray(data.qpos)[3:7]))
        event("student_to_cem", data, matched_phase_rad=float(phase), match_distance=float(distance))
        cem_start = float(data.time)
        reference = load_cem_reference(args.reference)
        geometry = student_env.geometry_parameters
        ref_geometry = CEMReferenceGeometry(torso_length_m=geometry.torso_length,
            link_length_m=geometry.edge_length, foot_diameter_m=2 * geometry.foot_radius,
            upper_link_length_m=geometry.upper_length, lower_link_length_m=geometry.lower_length)
        compact = jp.asarray((geometry.compact_hip_angle, geometry.compact_knee_angle) * 2)
        low = jp.asarray((geometry.hip.shell_compatible_range[0], geometry.knee.shell_compatible_range[0]) * 2)
        high = jp.asarray((geometry.hip.shell_compatible_range[1], geometry.knee.shell_compatible_range[1]) * 2)
        scales = jp.maximum(high - compact, compact - low)
        dt = float(stand_env.mj_model.opt.timestep)
        ids = student_env.controller_actuator_indices
        compact_ctrl = jp.asarray(stand_env.mj_model.key_ctrl[
            stand_env.mj_model.key("compact").id])[ids]
        def gate(d):
            return pitch_gate(d.time-cem_start, pitch_from_quaternion(d.qpos[3:7], jp),
                minimum=args.cem_seconds, tolerance=math.radians(args.pitch_tolerance_deg), xp=jp)
        @jax.jit
        def cem_step(d, oscillator, body):
            def condition(c):
                return (c[3] < student_env.action_repeat) & (~gate(c[0]))
            def physics(c):
                d, oscillator, body, count = c
                oscillator = advance_oscillator(jp, body, oscillator, dt, reference)
                planar = compact + scales * reference_action(jp, oscillator, reference,
                    compact_ctrl=compact, action_scales=scales, joint_low=low, joint_high=high,
                    geometry=ref_geometry)
                target = jp.asarray(lift_cem_targets(planar, compact_ctrl))
                d = mjx.step(stand_env.sys, d.replace(ctrl=d.ctrl.at[ids].set(
                    jp.clip(target, student_env.joint_low, student_env.joint_high))))
                return d, oscillator, body+d.qvel[4]*dt, count+1
            d, oscillator, body, _ = jax.lax.while_loop(condition, physics, (d, oscillator, body, jp.asarray(0)))
            return d, oscillator, body
        print("Compiling CEM continuation (angle checked every physics step)...", flush=True)
        while not bool(gate(data)):
            if float(data.time)-cem_start >= args.max_cem_seconds:
                raise RuntimeError("CEM angle gate timeout; did not trigger stand policy")
            data, phase, body_phase = cem_step(data, phase, body_phase)
            record(data, "CEM ROLL" if float(data.time)-cem_start < args.cem_seconds else "CEM WAIT FOR +90")
        event("cem_to_stand", data, cem_elapsed_s=float(data.time)-cem_start)
        before = {k: np.asarray(getattr(data, k)).copy() for k in ("qpos", "qvel", "ctrl", "time")}
        data = forward(data)
        state = jax.jit(stand_env.reset_from_roll_state)(data, jax.random.PRNGKey(args.seed+10000))
        for name, value in before.items():
            if not np.array_equal(value, np.asarray(getattr(state.pipeline_state, name))):
                raise RuntimeError(f"Stand handoff modified {name}")
        print("Compiling roll-to-stand; keeping policy active through stand verification...", flush=True)
        for i in range(stand_task.episode_length):
            action, _ = stand_policy(state.obs, jax.random.PRNGKey(args.seed+10001+i))
            state = stand_step(state, action)
            record(state.pipeline_state, "ROLL TO STAND")
            if bool(state.done):
                break
        success = float(state.metrics["transition_success"]) > 0 and float(state.metrics["failed"]) == 0
        report["final_metrics"] = {k: float(v) for k, v in state.metrics.items()}
        report["stable_stand_success"] = success
        if not success:
            raise RuntimeError("Roll-to-stand did not satisfy its continuous stand/verification gate")
        event("stable_stand", state.pipeline_state,
              verified_hold_s=stand_env.ready_hold_steps*stand_task.control_timestep)
        report["status"] = "success"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        if rows:
            np.savez_compressed(args.out / "rollout.npz", **{
                key: np.asarray([row[key] for row in rows]) for key in rows[0]})
    print("Rendering continuous MP4...", flush=True)
    try:
        render_video(stand_env.mj_model, rows, args.out / "stand_roll_stand.mp4", args.fps)
        report["video_status"] = "complete"
    except Exception as error:
        report.update(video_status="failed", video_error=str(error))
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Video: {args.out / 'stand_roll_stand.mp4'}", flush=True)


if __name__ == "__main__":
    main()
