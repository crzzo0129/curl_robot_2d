"""Measured trot timing and cycle balance. Array-module API; no JAX imports.

No fixed gait clock is fed to the policy. Completed, supported foot swings
provide touchdown timestamps, hip means/excursions and body-frame foot spans.
"""


METRICS = ("trot_phase_reward", "trot_phase_quality", "cycle_balance_penalty",
           "cycle_hip_mean_error", "cycle_hip_rom_error", "cycle_foot_span_error",
           "cycle_valid_fraction", "cycle_completed", "cycle_motion_gate")


def init_cycle_gait(xp, hip, foot_x, command):
    return {
        "cg_command": command, "cg_age": xp.zeros(()),
        "cg_contact": xp.ones(4, dtype=bool), "cg_started": xp.zeros(4, dtype=bool),
        "cg_touchdown": xp.zeros(4), "cg_swing": xp.zeros(4), "cg_peak": xp.zeros(4),
        "cg_hip_sum": xp.zeros(4), "cg_samples": xp.zeros(4),
        "cg_hip_min": hip, "cg_hip_max": hip,
        "cg_foot_min": foot_x, "cg_foot_max": foot_x,
        "cg_mean": hip, "cg_rom": xp.zeros(4), "cg_span": xp.zeros(4),
        "cg_valid": xp.zeros(4, dtype=bool), "cg_phase": xp.zeros(4),
    }


def update_cycle_gait(xp, state, hip, foot_x, command, contact, clearance, vx, dt, *,
                      phase_weight, balance_weight, warmup_s=0.5,
                      min_cycle_s=0.2, max_cycle_s=1.2, min_swing_s=0.06,
                      min_clearance=0.008, phase_tolerance=0.15,
                      mean_scale=0.20, rom_scale=0.20, span_scale=0.03):
    straight = ((xp.abs(command[0]) > 0.05) & (xp.abs(command[1]) < 0.05)
                & (xp.abs(command[2]) < 0.15))
    initial = init_cycle_gait(xp, hip, foot_x, command)
    reset = ~straight | xp.any(xp.abs(command - state["cg_command"]) > 1e-6)
    s = {key: xp.where(reset, value, state[key]) for key, value in initial.items()}
    age = s["cg_age"] + dt
    settled = straight & (age >= warmup_s)
    touchdown = contact & ~s["cg_contact"] & settled
    elapsed = age - s["cg_touchdown"]
    supported_swing = ~contact & xp.any(contact)
    swing = s["cg_swing"] + dt * supported_swing
    peak = xp.maximum(s["cg_peak"], xp.where(supported_swing, clearance, 0.0))
    hip_sum = s["cg_hip_sum"] + hip
    samples = s["cg_samples"] + 1
    hip_min, hip_max = xp.minimum(s["cg_hip_min"], hip), xp.maximum(s["cg_hip_max"], hip)
    foot_min, foot_max = xp.minimum(s["cg_foot_min"], foot_x), xp.maximum(s["cg_foot_max"], foot_x)
    completed = touchdown & s["cg_started"]
    valid_cycle = (completed & (elapsed >= min_cycle_s) & (elapsed <= max_cycle_s)
                   & (swing >= min_swing_s) & (peak >= min_clearance))
    valid = xp.where(completed, valid_cycle, s["cg_valid"]) & settled
    valid &= (elapsed <= max_cycle_s) | valid_cycle
    stamps = xp.where(touchdown, age, s["cg_touchdown"])
    means = xp.where(valid_cycle, hip_sum / xp.maximum(samples, 1), s["cg_mean"])
    roms = xp.where(valid_cycle, hip_max - hip_min, s["cg_rom"])
    spans = xp.where(valid_cycle, foot_max - foot_min, s["cg_span"])
    all_valid = xp.all(valid)

    # At each owner's touchdown, compare all partner touchdowns modulo its
    # measured period. FL/RR share phase 0; FR/RL share phase 1/2.
    phases = xp.asarray([0.0, 0.5, 0.5, 0.0], dtype=hip.dtype)
    target = (phases[:, None] - phases[None, :]) % 1.0
    measured = (age - stamps[None, :]) / xp.maximum(elapsed[:, None], min_cycle_s)
    circular_error = (measured - target + 0.5) % 1.0 - 0.5
    pair_mask = 1.0 - xp.eye(4, dtype=hip.dtype)
    phase_score = xp.exp(-xp.sum(circular_error**2 * pair_mask, axis=1)
                         / (3.0 * phase_tolerance**2))
    phase = xp.where(completed, xp.where(valid_cycle & all_valid, phase_score, 0.0), s["cg_phase"])
    phase = xp.where(valid, phase, 0.0)
    # Compare cycle statistics between left/right homologues, not instantaneous
    # angles and not front-vs-rear means during a single forward trajectory.
    left, right = xp.asarray([0, 2]), xp.asarray([1, 3])
    mean_error = means[left] - means[right]
    rom_error = roms[left] - roms[right]
    span_error = spans[left] - spans[right]
    balance = (xp.mean(xp.minimum((mean_error / mean_scale)**2, 1.0))
               + xp.mean(xp.minimum((rom_error / rom_scale)**2, 1.0))
               + xp.mean(xp.minimum((span_error / span_scale)**2, 1.0))) / 3.0
    # Correct-direction, realized motion is necessary for credit. Old cycles
    # expire, and standing still or jumping with all feet cannot earn reward.
    progress = vx * xp.sign(command[0])
    # Full credit at the requested speed for slow commands; do not create an
    # incentive to overshoot a 0.1 m/s request just to reach a fixed 0.15 m/s.
    full_credit_speed = xp.minimum(xp.abs(command[0]), 0.15)
    motion_gate = xp.clip((progress - 0.05) / xp.maximum(full_credit_speed - 0.05, 0.01), 0.0, 1.0)
    gate = straight * all_valid * motion_gate
    quality = xp.mean(phase) * all_valid
    metrics = {
        "trot_phase_reward": phase_weight * quality * motion_gate * straight,
        "trot_phase_quality": quality,
        "cycle_balance_penalty": balance_weight * balance * gate,
        "cycle_hip_mean_error": xp.mean(xp.abs(mean_error)) * all_valid,
        "cycle_hip_rom_error": xp.mean(xp.abs(rom_error)) * all_valid,
        "cycle_foot_span_error": xp.mean(xp.abs(span_error)) * all_valid,
        "cycle_valid_fraction": xp.mean(valid.astype(hip.dtype)),
        "cycle_completed": xp.sum(valid_cycle.astype(hip.dtype)),
        "cycle_motion_gate": motion_gate * straight,
    }
    restart = touchdown | ~settled
    next_state = {
        "cg_command": command, "cg_age": xp.where(straight, age, 0.0),
        "cg_contact": contact, "cg_started": (s["cg_started"] | touchdown) & settled,
        "cg_touchdown": xp.where(restart, age, s["cg_touchdown"]),
        "cg_swing": xp.where(restart, 0.0, swing), "cg_peak": xp.where(restart, 0.0, peak),
        "cg_hip_sum": xp.where(restart, 0.0, hip_sum), "cg_samples": xp.where(restart, 0.0, samples),
        "cg_hip_min": xp.where(restart, hip, hip_min), "cg_hip_max": xp.where(restart, hip, hip_max),
        "cg_foot_min": xp.where(restart, foot_x, foot_min), "cg_foot_max": xp.where(restart, foot_x, foot_max),
        "cg_mean": means, "cg_rom": roms, "cg_span": spans, "cg_valid": valid, "cg_phase": phase,
    }
    return next_state, metrics
