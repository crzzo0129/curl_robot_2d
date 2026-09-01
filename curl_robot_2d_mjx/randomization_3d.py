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
class RollingStudentDeployDomainRandomization:
    """Real-robot uncertainty used while continuing the rolling Student.

    The full-strength values intentionally match ``train_ppo_deploy.py``.
    ``scaled`` contracts every range toward the nominal model so an existing
    Student can enter domain randomization as a curriculum.
    """

    sliding_friction: tuple[float, float] = (0.60, 1.40)
    torso_mass_scale: tuple[float, float] = (0.85, 1.20)
    leg_mass_scale: tuple[float, float] = (0.90, 1.10)
    inertia_scale: tuple[float, float] = (0.85, 1.15)
    torso_com_xy_m: float = 0.010
    torso_com_z_m: float = 0.005
    motor_kp_scale: tuple[float, float] = (0.85, 1.15)
    motor_kd_scale: tuple[float, float] = (0.80, 1.20)
    motor_torque_scale: tuple[float, float] = (0.85, 1.15)
    action_latency_probabilities: tuple[float, float, float] = (
        0.60,
        0.30,
        0.10,
    )
    control_deadline_miss_probability: float = 0.05
    motor_zero_bias_rad: float = 0.020
    encoder_fixed_bias_rad: float = 0.010

    def scaled(self, strength: float):
        """Return a curriculum stage between nominal (0) and full DR (1)."""

        validate_deploy_domain_randomization_strength_3d(strength)

        def scale_range(limits):
            return tuple(1.0 + strength * (value - 1.0) for value in limits)

        latency = (
            1.0 - strength
            + strength * self.action_latency_probabilities[0],
            strength * self.action_latency_probabilities[1],
            strength * self.action_latency_probabilities[2],
        )
        return RollingStudentDeployDomainRandomization(
            sliding_friction=scale_range(self.sliding_friction),
            torso_mass_scale=scale_range(self.torso_mass_scale),
            leg_mass_scale=scale_range(self.leg_mass_scale),
            inertia_scale=scale_range(self.inertia_scale),
            torso_com_xy_m=strength * self.torso_com_xy_m,
            torso_com_z_m=strength * self.torso_com_z_m,
            motor_kp_scale=scale_range(self.motor_kp_scale),
            motor_kd_scale=scale_range(self.motor_kd_scale),
            motor_torque_scale=scale_range(self.motor_torque_scale),
            action_latency_probabilities=latency,
            control_deadline_miss_probability=(
                strength * self.control_deadline_miss_probability
            ),
            motor_zero_bias_rad=strength * self.motor_zero_bias_rad,
            encoder_fixed_bias_rad=strength * self.encoder_fixed_bias_rad,
        )


def validate_deploy_domain_randomization_strength_3d(strength: float) -> None:
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("deploy DR strength must be between zero and one")


def validate_student_deploy_domain_randomization_3d(
    settings: RollingStudentDeployDomainRandomization,
) -> None:
    for limits, name in (
        (settings.sliding_friction, "sliding_friction"),
        (settings.torso_mass_scale, "torso_mass_scale"),
        (settings.leg_mass_scale, "leg_mass_scale"),
        (settings.inertia_scale, "inertia_scale"),
        (settings.motor_kp_scale, "motor_kp_scale"),
        (settings.motor_kd_scale, "motor_kd_scale"),
        (settings.motor_torque_scale, "motor_torque_scale"),
    ):
        if len(limits) != 2:
            raise ValueError(f"{name} must contain two values")
        low, high = limits
        if not all(math.isfinite(value) for value in limits) or low <= 0 or high < low:
            raise ValueError(f"{name} must be a finite positive [low, high] range")
    for value, name in (
        (settings.torso_com_xy_m, "torso_com_xy_m"),
        (settings.torso_com_z_m, "torso_com_z_m"),
        (
            settings.control_deadline_miss_probability,
            "control_deadline_miss_probability",
        ),
        (settings.motor_zero_bias_rad, "motor_zero_bias_rad"),
        (settings.encoder_fixed_bias_rad, "encoder_fixed_bias_rad"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if settings.control_deadline_miss_probability > 1.0:
        raise ValueError("control_deadline_miss_probability cannot exceed one")
    probabilities = settings.action_latency_probabilities
    if len(probabilities) != 3 or any(
        not math.isfinite(value) or value < 0.0 for value in probabilities
    ):
        raise ValueError("action latency probabilities must be three nonnegative values")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("action latency probabilities must sum to one")


def make_student_deploy_domain_randomization_fn_3d(
    settings: RollingStudentDeployDomainRandomization,
    *,
    torso_body_id: int,
):
    """Build the deploy DR callback used by Brax's vectorization wrapper."""

    validate_student_deploy_domain_randomization_3d(settings)
    if torso_body_id <= 0:
        raise ValueError("torso_body_id must identify a non-world body")

    import jax
    import jax.numpy as jp

    def domain_randomize(model, rng):
        @jax.vmap
        def randomize_one(key):
            (
                friction_key,
                torso_mass_key,
                leg_mass_key,
                inertia_key,
                com_key,
                kp_key,
                kd_key,
                torque_key,
            ) = jax.random.split(key, 8)

            sliding_friction = jax.random.uniform(
                friction_key,
                (),
                minval=settings.sliding_friction[0],
                maxval=settings.sliding_friction[1],
            )
            geom_friction = model.geom_friction.at[:, 0].set(sliding_friction)

            torso_mass = jax.random.uniform(
                torso_mass_key,
                (),
                minval=settings.torso_mass_scale[0],
                maxval=settings.torso_mass_scale[1],
            )
            leg_mass = jax.random.uniform(
                leg_mass_key,
                (model.nbody,),
                minval=settings.leg_mass_scale[0],
                maxval=settings.leg_mass_scale[1],
            )
            mass_scale = (
                jp.ones((model.nbody,))
                .at[torso_body_id]
                .set(torso_mass)
            )
            dynamic_body = jp.arange(model.nbody) > 0
            leg_body = dynamic_body & (jp.arange(model.nbody) != torso_body_id)
            mass_scale = jp.where(leg_body, leg_mass, mass_scale)
            body_mass = model.body_mass * mass_scale

            inertia_uncertainty = jax.random.uniform(
                inertia_key,
                (model.nbody,),
                minval=settings.inertia_scale[0],
                maxval=settings.inertia_scale[1],
            )
            body_inertia = (
                model.body_inertia
                * mass_scale[:, None]
                * inertia_uncertainty[:, None]
            )

            com_unit = jax.random.uniform(
                com_key, (3,), minval=-1.0, maxval=1.0
            )
            com_extent = jp.asarray(
                (
                    settings.torso_com_xy_m,
                    settings.torso_com_xy_m,
                    settings.torso_com_z_m,
                )
            )
            body_ipos = model.body_ipos.at[torso_body_id].add(
                com_unit * com_extent
            )

            kp_scale = jax.random.uniform(
                kp_key,
                (model.nu,),
                minval=settings.motor_kp_scale[0],
                maxval=settings.motor_kp_scale[1],
            )
            kd_scale = jax.random.uniform(
                kd_key,
                (model.nu,),
                minval=settings.motor_kd_scale[0],
                maxval=settings.motor_kd_scale[1],
            )
            kp = model.actuator_gainprm[:, 0] * kp_scale
            actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
            actuator_biasprm = (
                model.actuator_biasprm
                .at[:, 1]
                .set(-kp)
                .at[:, 2]
                .set(model.actuator_biasprm[:, 2] * kd_scale)
            )
            torque_scale = jax.random.uniform(
                torque_key,
                (model.nu,),
                minval=settings.motor_torque_scale[0],
                maxval=settings.motor_torque_scale[1],
            )
            actuator_forcerange = (
                model.actuator_forcerange * torque_scale[:, None]
            )
            return (
                geom_friction,
                body_mass,
                body_inertia,
                body_ipos,
                actuator_gainprm,
                actuator_biasprm,
                actuator_forcerange,
            )

        values = randomize_one(rng)
        names = (
            "geom_friction",
            "body_mass",
            "body_inertia",
            "body_ipos",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
        )
        replacements = dict(zip(names, values))
        in_axes = jax.tree_util.tree_map(lambda _: None, model).tree_replace(
            {name: 0 for name in names}
        )
        return model.tree_replace(replacements), in_axes

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
