"""JAX command sampling and measured hip excursion for the deploy walker."""

import jax
import jax.numpy as jp


def sample_command(rng, vx_range, vy_range, wz_range, straight_prob,
                   stand_prob, min_speed):
    """Equal forward/backward straight buckets, plus mixed motion and stand."""
    kx, ky, kw, kb, ks = jax.random.split(rng, 5)
    mixed = jp.array([
        jax.random.uniform(kx, (), minval=vx_range[0], maxval=vx_range[1]),
        jax.random.uniform(ky, (), minval=vy_range[0], maxval=vy_range[1]),
        jax.random.uniform(kw, (), minval=wz_range[0], maxval=wz_range[1]),
    ])
    bucket = jax.random.uniform(kb)
    speed = jax.random.uniform(
        ks, (), minval=min_speed, maxval=min(-vx_range[0], vx_range[1]))
    sign = jp.where(bucket < straight_prob / 2.0, 1.0, -1.0)
    straight = jp.array([sign * speed, 0.0, 0.0])
    command = jp.where(bucket < straight_prob, straight, mixed)
    return jp.where(bucket >= 1.0 - stand_prob, jp.zeros(3), command)


def init_hip_rom(hip, command):
    """Fixed-shape state; angles are measured qpos in FL, FR, RL, RR order."""
    return {
        "hip_rom_command": command,
        "hip_rom_age": jp.zeros(()),
        "hip_rom_min": hip,
        "hip_rom_max": hip,
        "hip_rom_elapsed": jp.zeros(4),
        "hip_rom_swing": jp.zeros(4),
        "hip_rom_clearance": jp.zeros(4),
        "hip_rom_started": jp.zeros(4, dtype=bool),
        "hip_rom_contact": jp.ones(4, dtype=bool),
        "hip_rom_last": jp.zeros(4),
        "hip_rom_valid": jp.zeros(4, dtype=bool),
    }


def update_hip_rom(state, hip, command, contact, clearance, dt, *,
                   weight, target_rad, target_min_rad, reference_speed,
                   warmup_s, min_cycle_s, max_cycle_s, min_swing_s,
                   min_clearance):
    """Dense deficit cost from the last complete touchdown-to-touchdown cycle.

    Only a cycle with a supported swing and sufficient clearance counts.
    Invalid/stale cycles yield zero credited ROM, so stopping stepping cannot
    avoid the cost. No cost during command settling and the first cycle grace
    period. No comparison of raw front/rear angle signs is needed.
    """
    straight = ((jp.abs(command[0]) > 0.05)
                & (jp.abs(command[1]) < 0.05)
                & (jp.abs(command[2]) < 0.15))
    changed = jp.any(jp.abs(command - state["hip_rom_command"]) > 1e-6)
    reset = changed | ~straight
    initial = init_hip_rom(hip, command)
    s = {k: jp.where(reset, initial[k], state[k]) for k in initial}
    age = s["hip_rom_age"] + dt
    settled = straight & (age >= warmup_s)
    touchdown = contact & ~s["hip_rom_contact"] & settled
    started = s["hip_rom_started"]
    elapsed = s["hip_rom_elapsed"] + dt
    low = jp.minimum(s["hip_rom_min"], hip)
    high = jp.maximum(s["hip_rom_max"], hip)
    # Four feet in flight do not accumulate supported swing time/clearance.
    supported_swing = ~contact & jp.any(contact)
    swing_time = s["hip_rom_swing"] + dt * supported_swing
    peak = jp.maximum(s["hip_rom_clearance"],
                      jp.where(supported_swing, clearance, 0.0))
    completed = touchdown & started
    valid_cycle = (completed & (elapsed >= min_cycle_s)
                   & (elapsed <= max_cycle_s)
                   & (swing_time >= min_swing_s)
                   & (peak >= min_clearance))
    last = jp.where(completed, jp.where(valid_cycle, high - low, 0.0),
                    s["hip_rom_last"])
    valid = jp.where(completed, valid_cycle, s["hip_rom_valid"])
    valid &= elapsed <= max_cycle_s
    valid &= settled
    credited = jp.where(valid, last, 0.0)
    target = jp.clip(target_rad * jp.abs(command[0]) / reference_speed,
                     target_min_rad, target_rad)
    deficit = jp.square(jp.maximum(1.0 - credited / target, 0.0))
    penalty = (weight * jp.mean(deficit) * straight
               * (age >= warmup_s + max_cycle_s))

    restart = touchdown | ~settled
    next_state = {
        "hip_rom_command": command,
        "hip_rom_age": jp.where(straight, age, 0.0),
        "hip_rom_min": jp.where(restart, hip, low),
        "hip_rom_max": jp.where(restart, hip, high),
        "hip_rom_elapsed": jp.where(restart, 0.0, elapsed),
        "hip_rom_swing": jp.where(restart, 0.0, swing_time),
        "hip_rom_clearance": jp.where(restart, 0.0, peak),
        "hip_rom_started": (started | touchdown) & settled,
        "hip_rom_contact": contact,
        "hip_rom_last": credited,
        "hip_rom_valid": valid,
    }
    metrics = {
        "hip_rom_penalty": penalty,
        "hip_rom_target": jp.where(straight, target, 0.0),
        "hip_rom_front": jp.mean(credited[:2]),
        "hip_rom_rear": jp.mean(credited[2:]),
        "hip_rom_valid_fraction": jp.mean(valid.astype(jp.float32)),
        "hip_rom_cycles": jp.sum(valid_cycle.astype(jp.float32)),
    }
    return next_state, metrics
