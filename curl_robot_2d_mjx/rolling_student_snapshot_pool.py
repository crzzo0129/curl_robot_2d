"""Generate reusable rolling reset states before PPO, never during resets."""

import numpy as np


def handoff_split_indices(trajectory_ids, *, evaluation=False):
    """Stable trajectory split, independent of command/reset random seeds."""
    ids = np.unique(trajectory_ids)
    held_out = ids[::4]
    selected = held_out if evaluation else np.setdiff1d(ids, held_out)
    return np.flatnonzero(np.isin(trajectory_ids, selected)), selected


def read_handoff_bank(path):
    import json
    from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        bank = {k:np.asarray(archive[k]) for k in ('qpos','qvel','ctrl','time','history',
                                                'previous_action','speed','rolling_phase','trajectory_id')}
    if metadata.get('schema') != 'rolling_handoff_bank_v1' or metadata.get('history_order') != 'newest_first_past_frames':
        raise ValueError('Unsupported handoff bank history/schema')
    if metadata.get('controller_joint_names') != list(CONTROLLER_JOINT_NAMES_3D):
        raise ValueError('Handoff bank joint order mismatch')
    if bank['qpos'].ndim != 2:
        raise ValueError('Handoff qpos must be a matrix')
    n = bank['qpos'].shape[0]
    if n < 8 or any(v.ndim == 0 or v.shape[0] != n or not np.all(np.isfinite(v)) for v in bank.values()):
        raise ValueError('Handoff bank must contain >=8 finite states with matching rows')
    if any(bank[k].shape != (n,) for k in ('time','speed','rolling_phase','trajectory_id')):
        raise ValueError('Handoff scalar fields must have one value per row')
    if any(bank[k].ndim != 2 for k in ('qvel','ctrl')):
        raise ValueError('Handoff qvel and ctrl must be matrices')
    if bank['history'].shape != (n,720) or bank['previous_action'].shape != (n,12):
        raise ValueError('Handoff bank must preserve 720 history and 12 previous actions')
    if not np.all((bank['speed'] >= .3) & (bank['speed'] <= 1.2)):
        raise ValueError('Handoff speed outside reviewed collection range')
    if len(np.unique(bank['trajectory_id'])) < 8:
        raise ValueError('At least eight distinct trajectories are needed for a held-out split')
    if not np.issubdtype(bank['trajectory_id'].dtype, np.integer):
        raise ValueError('Trajectory IDs must be integers')
    if np.any(np.abs(bank['previous_action']) > 1.0001):
        raise ValueError('Previous actions must be normalized controller actions')
    return bank, metadata


def build_handoff_snapshot_pool(base_env, path, *, count, seed, evaluation=False, num_devices=1):
    """Import measured physical state/history; reconstruct task bookkeeping."""
    import hashlib
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx
    from curl_robot_2d_mjx.distillation_execution import BatchExecution, timed_stage
    from curl_robot_2d_mjx.deployment_rolling_3d import controller_action_to_effective_action_3d
    bank, metadata = read_handoff_bank(path)
    model = base_env.mj_model
    if bank['qpos'].shape[1:] != (model.nq,) or bank['qvel'].shape[1:] != (model.nv,) or bank['ctrl'].shape[1:] != (model.nu,):
        raise ValueError('Handoff bank physical layout differs from the training model')
    if abs(metadata['control_period_s'] - base_env.config.control_timestep) > 1e-9:
        raise ValueError('Handoff control period mismatch')
    if base_env.config.rolling_hip_knee_kp_scale != 1.0:
        raise ValueError('Collected P=5 handoffs require rolling-hip-knee-kp-scale=1')
    xml_hash = hashlib.sha256(base_env.model_path.read_text(encoding='utf-8').encode()).hexdigest()
    if metadata.get('xml_text_sha256') != xml_hash:
        raise ValueError('Handoff XML differs from training geometry; recollect for this model')
    names = metadata['controller_joint_names']
    joints = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    actuators = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n+'_servo') for n in names]
    key = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'compact')
    centers = np.asarray(model.key_qpos[key])[np.asarray(model.jnt_qposadr)[joints]]
    contract = metadata['controller_metadata']
    for field, expected in (('default_joint_pos',centers), ('action_scale',[0,.8,1.2]*4),
                            ('kp',5.), ('kd',.1)):
        value = np.asarray(contract[field])
        if value.shape != np.asarray(expected).shape or not np.allclose(value,expected,rtol=0,atol=1e-6):
            raise ValueError('Handoff controller contract mismatch: '+field)
    if not np.allclose(model.actuator_gainprm[actuators,0],5.,atol=1e-6) or not np.allclose(model.actuator_biasprm[actuators,2],-.1,atol=1e-6):
        raise ValueError('Handoff PD differs from nominal training model')
    candidates, selected_ids = handoff_split_indices(bank['trajectory_id'], evaluation=evaluation)
    chosen = np.random.default_rng(seed).choice(candidates, size=count, replace=True)
    arrays = {k:jp.asarray(v[chosen]) for k,v in bank.items()}

    def one(key, row):
        state = base_env.reset(key)
        data = mjx.forward(base_env.sys, state.pipeline_state.replace(
            qpos=row['qpos'], qvel=row['qvel'], ctrl=row['ctrl'], time=row['time'],
            qacc_warmstart=jp.zeros_like(state.pipeline_state.qacc_warmstart)))
        action = controller_action_to_effective_action_3d(jp, row['previous_action'])
        axis = data.xmat[base_env.torso_body_id].reshape(3,3)[:,1]
        heading = jp.arctan2(-axis[0], axis[1])
        contacts = base_env._contact_metrics(data)
        from curl_robot_2d_mjx.reward_3d import stability_error_cost_3d
        stability = stability_error_cost_3d(jp,base_env.reward_config,{
            'lateral_velocity':data.qvel[1], 'lateral_drift':jp.asarray(0.),
            'yaw_rate':data.qvel[5], 'yaw':heading})
        info = {**state.info, 'initial_root_x':data.qpos[0], 'initial_root_y':data.qpos[1],
                'previous_root_x':data.qpos[0], 'previous_root_y':data.qpos[1],
                'previous_rolling_axis_heading':heading, 'last_action':action,
                'last_policy_action':action, 'last_reference_action':action,
                'rolling_phase':row['rolling_phase'], 'oscillator_phase':row['rolling_phase'],
                'previous_same_side_foot_contact':contacts['same_side_foot_count']>0,
                'previous_stability_cost':stability,
                'handoff_snapshot':jp.asarray(1.,dtype=jp.float32),
                'handoff_start_speed':jp.clip(row['speed'],base_env.config.forward_command_min_m_s,
                                            base_env.config.handoff_initial_speed_max_m_s)}
        observation = base_env._observation(data,action,contacts,
            axis_tilt=base_env._rolling_axis_tilt(data,info['yaw_rate_command']),
            reference_action_value=action,oscillator_phase=info['oscillator_phase'],
            rolling_phase=info['rolling_phase'],action_ramp=jp.asarray(1.),lateral_drift=jp.asarray(0.),
            lateral_velocity_command=info['lateral_velocity_command'],
            forward_velocity_command=info['forward_velocity_command'],yaw_rate_command=info['yaw_rate_command'],
            rolling_axis_heading_rate=jp.asarray(0.))
        return state.replace(pipeline_state=data,info=info,obs=observation), row['history'], row['previous_action']

    if count < 1 or num_devices < 1 or count % num_devices:
        raise ValueError('Handoff pool size must be positive and divisible by selected devices')
    generate = BatchExecution(num_devices).batch_jit(jax.vmap(one))
    with timed_stage(f"PPO handoff pool {'eval' if evaluation else 'train'} count={count}"):
        pool = jax.device_get(generate(jax.random.split(jax.random.PRNGKey(seed),count),arrays))
    return pool, {'source':metadata['source'],'bank_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'split':'evaluation' if evaluation else 'training','trajectory_ids':selected_ids.tolist(),
        'samples':count,'unique_physical_samples':int(len(np.unique(chosen))),
        'speed_summary_m_s':{name:float(fn(bank['speed'][candidates])) for name,fn in
                              (('min',np.min),('median',np.median),('max',np.max))},
        'bookkeeping':'Physical state and sensor/action history preserved; progress origin rebased and critic reference phase reconstructed.'}


def append_handoff_pool(mature, handoff):
    import jax
    return jax.tree_util.tree_map(lambda a,b:np.concatenate((np.asarray(a),np.asarray(b)),axis=0),mature,handoff)


def tracking_focus_snapshot_cdf(pool, *, speed_min, speed_max):
    """Reweight training resets by command group without changing evaluation."""
    if not speed_min < speed_max:
        raise ValueError("tracking-focused sampling requires a nonzero speed range")
    forward = np.asarray(pool[0].info["forward_velocity_command"])
    yaw = np.asarray(pool[0].info["yaw_rate_command"])
    edges = np.linspace(speed_min, speed_max, 4)
    bins = np.searchsorted(edges[1:-1], forward, side="right")
    probabilities = np.zeros(len(forward), dtype=np.float64)
    groups = {}
    directions = {"straight": np.abs(yaw) <= 1e-3,
                  "left": yaw > 1e-3, "right": yaw < -1e-3}
    for index, (speed, mass) in enumerate(zip(("low", "medium", "high"), (0.4, 0.2, 0.4))):
        for direction, turn_mask in directions.items():
            mask = (bins == index) & turn_mask
            count = int(np.sum(mask))
            if not count:
                raise ValueError(f"No snapshots for {speed}/{direction}; increase the training pool or revise command ranges")
            group_mass = mass * (0.6 if direction == "straight" else 0.2)
            probabilities[mask] = group_mass / count
            groups[f"{speed}/{direction}"] = {"snapshots": count, "reset_probability": group_mass}
    probabilities /= probabilities.sum()
    cdf = np.cumsum(probabilities).astype(np.float32)
    cdf[-1] = 1.0
    return cdf, {"mode": "tracking_focus", "speed_bin_edges_m_s": edges.tolist(),
                 "groups": groups, "note": "Training reset probabilities only; evaluation samples uniformly."}


def build_cem_snapshot_pool(teacher_env, observation_env, *, count, seed,
                            min_steps, max_steps, num_devices=1):
    import jax
    import jax.numpy as jp
    from curl_robot_2d_mjx.deployment_rolling_3d import (
        initial_rolling_deploy_history_3d,
        effective_action_to_controller_action_3d,
    )
    from curl_robot_2d_mjx.distillation_execution import BatchExecution, timed_stage

    execution = BatchExecution(num_devices)

    def generate_one(key):
        reset_key, warmup_key = jax.random.split(key)
        state = teacher_env.reset(reset_key)
        history = initial_rolling_deploy_history_3d(jp)
        previous = jp.zeros((12,))
        warmup_steps = jax.random.randint(warmup_key, (), min_steps, max_steps + 1)

        def advance(carry, index):
            current, old_history, old_previous = carry
            new_history = observation_env._actor_observation(
                current, old_history, old_previous, jp.zeros((12,)),
                jax.random.fold_in(key, index),
            )
            candidate = teacher_env.step(current, jp.zeros((8,)))
            new_previous = effective_action_to_controller_action_3d(
                jp, candidate.info["last_action"]
            )
            take = (index < warmup_steps) & (candidate.done < 0.5)
            return jax.tree_util.tree_map(
                lambda new, old: jp.where(take, new, old),
                (candidate, new_history, new_previous), carry,
            ), None

        result, _ = jax.lax.scan(advance, (state, history, previous), jp.arange(max_steps))
        return result

    generate = execution.batch_jit(jax.vmap(generate_one))
    with timed_stage(f"PPO CEM snapshot pool seed={seed} candidates={count}"):
        pool = jax.device_get(generate(jax.random.split(jax.random.PRNGKey(seed), count)))
    state, _, _ = pool
    # A warmup timer alone does not prove the robot is rolling. Reject stalled,
    # nonfinite, terminal and very early states instead of silently using compact.
    valid = ((np.asarray(state.info["step_count"]) >= min_steps)
             & (np.asarray(state.done) < 0.5)
             & (np.asarray(state.metrics["failed"]) < 0.5)
             & (np.asarray(state.metrics["forward_velocity_m_s"]) > 0.05)
             & (np.abs(np.asarray(state.pipeline_state.qvel)[:, 4]) > 0.5)
             & np.all(np.isfinite(np.asarray(state.pipeline_state.qpos)), axis=-1)
             & np.all(np.isfinite(np.asarray(state.pipeline_state.qvel)), axis=-1))
    indices = np.flatnonzero(valid)
    if len(indices) < max(4, count // 8):
        raise RuntimeError(
            f"Only {len(indices)}/{count} valid rolling snapshots. "
            "Check the CEM reference or increase warmup; refusing compact reset fallback."
        )
    pool = jax.tree_util.tree_map(lambda value: np.asarray(value)[indices], pool)
    selected = pool[0]
    summary = {
        "source": "cem_rolling_proxy_not_actual_stand_to_roll_handoff",
        "seed": seed, "candidate_count": count, "accepted_count": len(indices),
        "warmup_min_steps": min_steps, "warmup_max_steps": max_steps,
        "forward_commands_m_s": np.asarray(selected.info["forward_velocity_command"]).tolist(),
        "yaw_commands_rad_s": np.asarray(selected.info["yaw_rate_command"]).tolist(),
        "actual_warmup_steps": np.asarray(selected.info["step_count"]).tolist(),
        "minimum_forward_speed_m_s": 0.05, "minimum_abs_roll_rate_rad_s": 0.5,
    }
    print(f"[PPO snapshots] retained {len(indices)}/{count}; "
          "resets sample this pool without teacher warmup", flush=True)
    return pool, summary
