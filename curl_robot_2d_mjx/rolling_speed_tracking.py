"""Array-backend independent horizontal speed and takeover command definitions."""


def heading_displacement(xp, dx, dy, previous_heading, heading):
    delta = xp.arctan2(xp.sin(heading - previous_heading), xp.cos(heading - previous_heading))
    middle = previous_heading + 0.5 * delta
    return dx * xp.cos(middle) + dy * xp.sin(middle)


def takeover_commands(xp, initial_speed, target_speed, target_yaw, elapsed_s,
                      speed_slew, yaw_slew):
    elapsed_s = xp.maximum(elapsed_s, 0.0)
    speed = initial_speed + xp.clip(target_speed - initial_speed,
                                   -speed_slew * elapsed_s, speed_slew * elapsed_s)
    yaw = xp.clip(target_yaw, -yaw_slew * elapsed_s, yaw_slew * elapsed_s)
    transitioning = (xp.abs(speed - target_speed) > 1e-6) | (xp.abs(yaw - target_yaw) > 1e-6)
    return speed, yaw, transitioning


def update_speed_window(xp, buffer, speed, command, transition, elapsed_steps):
    """buffer (..., window, 3): speed, applied command, command-transition flag."""
    row = xp.stack((speed, command, transition), axis=-1)
    buffer = xp.concatenate((buffer[..., 1:, :], row[..., None, :]), axis=-2)
    average = xp.mean(buffer, axis=-2)
    valid = (elapsed_steps >= buffer.shape[-2]) & (xp.max(buffer[..., 2], axis=-1) < .5)
    return buffer, average[..., 0] - average[..., 1], valid


def update_speed_settling(xp, streak, first_time, error, valid, healthy,
                          elapsed_s, *, dt, hold_steps, tolerance=.05):
    """Earliest start of a confirmed in-band interval; -1 means not observed."""
    inside = valid & healthy & (xp.abs(error) <= tolerance)
    streak = xp.where(inside, streak + 1, 0)
    first_time = xp.where((first_time < 0) & (streak >= hold_steps),
                          elapsed_s - (hold_steps - 1) * dt, first_time)
    return streak, first_time, inside


def speed_checkpoint_eligible(record, baseline, tolerance):
    """Survival constraints before comparing tracking error on a fixed panel."""
    if record['success_rate'] < baseline['success_rate'] - tolerance:
        return False
    for source, old in baseline['tracking_by_reset_source'].items():
        new = record['tracking_by_reset_source'][source]
        if old['episodes'] and new['full_horizon_rate'] < old['full_horizon_rate'] - tolerance:
            return False
    return True


def select_speed_checkpoint(records, tolerance=.03):
    eligible = [r for r in records if speed_checkpoint_eligible(r, records[0], tolerance)]
    def score(record):
        group = record['tracking_by_reset_source']['handoff']
        error = group.get('windowed_forward_mae_m_s')
        steady = group.get('steady_forward_mae_m_s')
        return (float('inf') if error is None else error,
                float('inf') if steady is None else steady, record['yaw_mae_rad_s'], -record['success_rate'])
    return min(eligible, key=score)


def exploration_should_stop(records, *, drop=.15, patience=3, warmup_steps=245760):
    """Allow transient regressions; only persistent post-warmup decline stops PPO."""
    assessed = [r for r in records if r['step'] > 0 and r['step'] >= warmup_steps]
    return len(assessed) >= patience and all(
        not speed_checkpoint_eligible(r, records[0], drop) for r in assessed[-patience:])
