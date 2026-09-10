"""Mild model DR adapted from scripts/train_ppo_deploy.py; no external pushes.

Resampled per episode and held constant during that episode. Snapshot qpos,
qvel and ctrl are untouched; dynamics variation begins at the handoff.
"""

DR_RANGES = dict(friction_scale=(0.85, 1.15), torso_mass=(0.95, 1.05),
                 leg_mass=(0.95, 1.05), inertia=(0.95, 1.05),
                 torso_com_m=(0.003, 0.003, 0.002), kp=(0.95, 1.05),
                 kd=(0.90, 1.10), torque=(0.90, 1.00))


def apply_model_samples(xp, model, torso_id, samples):
    """Map independent unit samples to model parameters; no nominal mutation."""
    def scale(name, value):
        low, high = DR_RANGES[name]
        return low + (high - low) * value

    mass_scale = scale('leg_mass', samples['mass'])
    # World body has no physical mass. Use the actual named torso body index.
    mask = xp.arange(model.nbody)
    mass_scale = xp.where(mask == 0, 1., mass_scale)
    mass_scale = xp.where(mask == torso_id, scale('torso_mass', samples['torso']), mass_scale)
    com = ((samples['com'] * 2 - 1) * xp.asarray(DR_RANGES['torso_com_m']))
    ipos = model.body_ipos + (mask == torso_id)[:, None] * com
    kp_scale = scale('kp', samples['kp'])
    kd_scale = scale('kd', samples['kd'])
    gain_factors = xp.ones_like(model.actuator_gainprm)
    gain_factors = xp.where(xp.arange(gain_factors.shape[1])[None, :] == 0,
                            kp_scale[:, None], gain_factors)
    bias_factors = xp.where(xp.arange(model.actuator_biasprm.shape[1])[None, :] == 1,
                            kp_scale[:, None], 1.)
    bias_factors = xp.where(xp.arange(model.actuator_biasprm.shape[1])[None, :] == 2,
                            kd_scale[:, None], bias_factors)
    friction_factors = xp.asarray([scale('friction_scale', samples['friction']), 1., 1.])
    return model.replace(
        geom_friction=model.geom_friction * friction_factors,
        body_mass=model.body_mass * mass_scale,
        body_inertia=model.body_inertia * mass_scale[:, None] * scale('inertia', samples['inertia'])[:, None],
        body_ipos=ipos, actuator_gainprm=model.actuator_gainprm * gain_factors,
        actuator_biasprm=model.actuator_biasprm * bias_factors,
        actuator_forcerange=model.actuator_forcerange * scale('torque', samples['torque'])[:, None])


def randomize_transition_model(model, rng, torso_id):
    import jax
    import jax.numpy as jp
    shapes = dict(friction=(), mass=(model.nbody,), torso=(), inertia=(model.nbody,),
                  com=(3,), kp=(model.nu,), kd=(model.nu,), torque=(model.nu,))
    keys = jax.random.split(jax.random.fold_in(rng, 8317), len(shapes))
    samples = {name: jax.random.uniform(key, shape) for (name, shape), key in zip(shapes.items(), keys)}
    return apply_model_samples(jp, model, torso_id, samples)
