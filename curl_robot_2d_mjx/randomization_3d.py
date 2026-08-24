"""Dependency-light domain randomization for the 3-D MJX task."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Rolling3DDomainRandomization:
    """Multiplicative ranges sampled once for each parallel MJX model."""

    geom_friction_scale: tuple[float, float] = (1.0, 1.0)
    floor_friction_scale: tuple[float, float] = (1.0, 1.0)
    body_mass_scale: tuple[float, float] = (1.0, 1.0)
    actuator_gain_scale: tuple[float, float] = (1.0, 1.0)

    @property
    def enabled(self) -> bool:
        return any(
            limits != (1.0, 1.0)
            for limits in (
                self.geom_friction_scale,
                self.floor_friction_scale,
                self.body_mass_scale,
                self.actuator_gain_scale,
            )
        )


def validate_domain_randomization_3d(
    settings: Rolling3DDomainRandomization,
) -> None:
    for limits, name in (
        (settings.geom_friction_scale, "geom_friction_scale"),
        (settings.floor_friction_scale, "floor_friction_scale"),
        (settings.body_mass_scale, "body_mass_scale"),
        (settings.actuator_gain_scale, "actuator_gain_scale"),
    ):
        if len(limits) != 2:
            raise ValueError(f"{name} must contain two values")
        low, high = limits
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or low <= 0.0
            or high < low
        ):
            raise ValueError(
                f"{name} must be a finite positive [low, high] range"
            )


def make_domain_randomization_fn_3d(
    settings: Rolling3DDomainRandomization,
    *,
    floor_geom_id: int | None = None,
):
    """Build the callback expected by Brax PPO's randomization wrapper."""

    validate_domain_randomization_3d(settings)
    if not settings.enabled:
        return None
    floor_randomization_enabled = (
        settings.floor_friction_scale != (1.0, 1.0)
    )
    if floor_randomization_enabled and floor_geom_id is None:
        raise ValueError(
            "floor_geom_id is required when floor friction randomization is enabled"
        )

    import jax

    def domain_randomize(model, rng):
        @jax.vmap
        def randomize_one(key):
            friction_key, mass_key, gain_key = jax.random.split(key, 3)
            friction_scale = jax.random.uniform(
                friction_key,
                shape=(),
                minval=settings.geom_friction_scale[0],
                maxval=settings.geom_friction_scale[1],
            )
            geom_friction = model.geom_friction * friction_scale
            if floor_randomization_enabled:
                floor_scale = jax.random.uniform(
                    jax.random.fold_in(key, 0xF100),
                    shape=(),
                    minval=settings.floor_friction_scale[0],
                    maxval=settings.floor_friction_scale[1],
                )
                geom_friction = geom_friction.at[floor_geom_id].multiply(
                    floor_scale
                )

            mass_scale = jax.random.uniform(
                mass_key,
                shape=(model.nbody,),
                minval=settings.body_mass_scale[0],
                maxval=settings.body_mass_scale[1],
            )
            body_mass = model.body_mass * mass_scale
            body_inertia = model.body_inertia * mass_scale[:, None]

            gain_scale = jax.random.uniform(
                gain_key,
                shape=(model.nu,),
                minval=settings.actuator_gain_scale[0],
                maxval=settings.actuator_gain_scale[1],
            )
            kp = model.actuator_gainprm[:, 0] * gain_scale
            actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
            actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
            return (
                geom_friction,
                body_mass,
                body_inertia,
                actuator_gainprm,
                actuator_biasprm,
            )

        (
            geom_friction,
            body_mass,
            body_inertia,
            actuator_gainprm,
            actuator_biasprm,
        ) = randomize_one(rng)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace(
            {
                "geom_friction": 0,
                "body_mass": 0,
                "body_inertia": 0,
                "actuator_gainprm": 0,
                "actuator_biasprm": 0,
            }
        )
        randomized_model = model.tree_replace(
            {
                "geom_friction": geom_friction,
                "body_mass": body_mass,
                "body_inertia": body_inertia,
                "actuator_gainprm": actuator_gainprm,
                "actuator_biasprm": actuator_biasprm,
            }
        )
        return randomized_model, in_axes

    return domain_randomize


@dataclass(frozen=True)
class Walking3DDomainRandomization:
    """ANYmal-style model randomization ranges for command locomotion."""

    geom_friction_scale: tuple[float, float] = (0.60, 1.40)
    body_mass_scale: tuple[float, float] = (0.90, 1.10)
    actuator_gain_scale: tuple[float, float] = (0.90, 1.10)
    joint_damping_scale: tuple[float, float] = (0.80, 1.20)
    joint_armature_scale: tuple[float, float] = (0.80, 1.20)

    @property
    def enabled(self) -> bool:
        return any(
            limits != (1.0, 1.0)
            for limits in (
                self.geom_friction_scale,
                self.body_mass_scale,
                self.actuator_gain_scale,
                self.joint_damping_scale,
                self.joint_armature_scale,
            )
        )


def validate_walking_domain_randomization_3d(
    settings: Walking3DDomainRandomization,
) -> None:
    for limits, name in (
        (settings.geom_friction_scale, "geom_friction_scale"),
        (settings.body_mass_scale, "body_mass_scale"),
        (settings.actuator_gain_scale, "actuator_gain_scale"),
        (settings.joint_damping_scale, "joint_damping_scale"),
        (settings.joint_armature_scale, "joint_armature_scale"),
    ):
        if len(limits) != 2:
            raise ValueError(f"{name} must contain two values")
        low, high = limits
        if not all(math.isfinite(x) for x in limits) or low <= 0 or high < low:
            raise ValueError(f"{name} must be a finite positive [low, high] range")


def make_walking_domain_randomization_fn_3d(
    settings: Walking3DDomainRandomization,
):
    """Build Brax's batched MJX model randomizer for walking PPO."""

    validate_walking_domain_randomization_3d(settings)
    if not settings.enabled:
        return None

    import jax

    def domain_randomize(model, rng):
        @jax.vmap
        def randomize_one(key):
            friction_key, mass_key, gain_key, damping_key, armature_key = (
                jax.random.split(key, 5)
            )
            friction = jax.random.uniform(
                friction_key, (),
                minval=settings.geom_friction_scale[0],
                maxval=settings.geom_friction_scale[1],
            )
            mass = jax.random.uniform(
                mass_key, (model.nbody,),
                minval=settings.body_mass_scale[0],
                maxval=settings.body_mass_scale[1],
            )
            gain = jax.random.uniform(
                gain_key, (model.nu,),
                minval=settings.actuator_gain_scale[0],
                maxval=settings.actuator_gain_scale[1],
            )
            damping = jax.random.uniform(
                damping_key, (model.nv,),
                minval=settings.joint_damping_scale[0],
                maxval=settings.joint_damping_scale[1],
            )
            armature = jax.random.uniform(
                armature_key, (model.nv,),
                minval=settings.joint_armature_scale[0],
                maxval=settings.joint_armature_scale[1],
            )
            kp = model.actuator_gainprm[:, 0] * gain
            return (
                model.geom_friction * friction,
                model.body_mass * mass,
                model.body_inertia * mass[:, None],
                model.actuator_gainprm.at[:, 0].set(kp),
                model.actuator_biasprm.at[:, 1].set(-kp),
                model.dof_damping * damping,
                model.dof_armature * armature,
            )

        values = randomize_one(rng)
        names = (
            "geom_friction", "body_mass", "body_inertia",
            "actuator_gainprm", "actuator_biasprm", "dof_damping",
            "dof_armature",
        )
        replacements = dict(zip(names, values))
        in_axes = jax.tree_util.tree_map(lambda _: None, model).tree_replace(
            {name: 0 for name in names}
        )
        return model.tree_replace(replacements), in_axes

    return domain_randomize
