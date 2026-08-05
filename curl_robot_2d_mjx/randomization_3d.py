"""Dependency-light domain randomization for the 3-D MJX task."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Rolling3DDomainRandomization:
    """Multiplicative ranges sampled once for each parallel MJX model."""

    geom_friction_scale: tuple[float, float] = (1.0, 1.0)
    body_mass_scale: tuple[float, float] = (1.0, 1.0)
    actuator_gain_scale: tuple[float, float] = (1.0, 1.0)

    @property
    def enabled(self) -> bool:
        return any(
            limits != (1.0, 1.0)
            for limits in (
                self.geom_friction_scale,
                self.body_mass_scale,
                self.actuator_gain_scale,
            )
        )


def validate_domain_randomization_3d(
    settings: Rolling3DDomainRandomization,
) -> None:
    for limits, name in (
        (settings.geom_friction_scale, "geom_friction_scale"),
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
):
    """Build the callback expected by Brax PPO's randomization wrapper."""

    validate_domain_randomization_3d(settings)
    if not settings.enabled:
        return None

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
