"""Stage definitions for 3-D rolling robustness training."""

from __future__ import annotations

from dataclasses import dataclass, replace

from curl_robot_2d_mjx.config_3d import Rolling3DConfig
from curl_robot_2d_mjx.randomization_3d import (
    Rolling3DDomainRandomization,
)


CURRICULUM_NAMES_3D = (
    "none",
    "reset_v1",
    "reset_v2",
    "nominal_reset_v3",
    "independent_reset_v4",
    "friction_v1",
    "floor_friction_v2",
    "floor_mass_v2",
    "floor_mass_gain_v3",
    "friction_low_v1",
    "mass_v1",
    "robustness_v1",
)


@dataclass(frozen=True)
class Rolling3DCurriculumStage:
    name: str
    weight: float
    reset_joint_noise_rad: float | None = None
    reset_velocity_noise: float | None = None
    reset_root_velocity_noise: float | None = None
    reset_pair_differential_scale: float | None = None
    reset_independent: bool = False
    reset_axis_tilt_noise_rad: float | None = None
    domain_randomization: Rolling3DDomainRandomization = (
        Rolling3DDomainRandomization()
    )

    def task_config(self, base: Rolling3DConfig) -> Rolling3DConfig:
        overrides = {
            name: value
            for name, value in (
                ("reset_joint_noise_rad", self.reset_joint_noise_rad),
                ("reset_velocity_noise", self.reset_velocity_noise),
                (
                    "reset_root_velocity_noise",
                    self.reset_root_velocity_noise,
                ),
                (
                    "reset_pair_differential_scale",
                    self.reset_pair_differential_scale,
                ),
                (
                    "reset_axis_tilt_noise_rad",
                    self.reset_axis_tilt_noise_rad,
                ),
            )
            if value is not None
        }
        if self.reset_independent:
            overrides["reset_pair_differential_scale"] = None
        if self.domain_randomization.floor_friction_scale != (1.0, 1.0):
            overrides["floor_contact_friction_override"] = True
        return replace(base, **overrides)


RESET_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="symmetric_reset",
        weight=0.20,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_pair_differential_scale=0.0,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="differential_005",
        weight=0.15,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_pair_differential_scale=0.05,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="differential_010",
        weight=0.20,
        reset_joint_noise_rad=0.0075,
        reset_velocity_noise=0.010,
        reset_pair_differential_scale=0.10,
        reset_axis_tilt_noise_rad=0.005,
    ),
    Rolling3DCurriculumStage(
        name="differential_025",
        weight=0.45,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
    ),
)

RESET_V2_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="tilt_v2_0000",
        weight=0.10,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="tilt_v2_0100",
        weight=0.10,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.010,
    ),
    Rolling3DCurriculumStage(
        name="tilt_v2_0150",
        weight=0.15,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.015,
    ),
    Rolling3DCurriculumStage(
        name="tilt_v2_0175",
        weight=0.20,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.0175,
    ),
    Rolling3DCurriculumStage(
        name="tilt_v2_0200",
        weight=0.20,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.020,
    ),
    Rolling3DCurriculumStage(
        name="tilt_v2_0300",
        weight=0.25,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
    ),
)


NOMINAL_RESET_V3_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="reset3_tiny_symmetric",
        weight=0.05,
        reset_joint_noise_rad=0.0005,
        reset_velocity_noise=0.0005,
        reset_pair_differential_scale=0.0,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset3_low_symmetric",
        weight=0.10,
        reset_joint_noise_rad=0.001,
        reset_velocity_noise=0.001,
        reset_pair_differential_scale=0.0,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset3_nominal_symmetric",
        weight=0.15,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_pair_differential_scale=0.0,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset3_differential_010",
        weight=0.20,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_pair_differential_scale=0.10,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset3_differential_025",
        weight=0.20,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset3_independent",
        weight=0.30,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
    ),
)

INDEPENDENT_RESET_V4_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="reset4_independent_0005",
        weight=0.05,
        reset_joint_noise_rad=0.0005,
        reset_velocity_noise=0.0005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset4_independent_0010",
        weight=0.10,
        reset_joint_noise_rad=0.001,
        reset_velocity_noise=0.001,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset4_independent_0020",
        weight=0.15,
        reset_joint_noise_rad=0.002,
        reset_velocity_noise=0.002,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset4_independent_0030",
        weight=0.20,
        reset_joint_noise_rad=0.003,
        reset_velocity_noise=0.003,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset4_independent_0040",
        weight=0.20,
        reset_joint_noise_rad=0.004,
        reset_velocity_noise=0.004,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
    ),
    Rolling3DCurriculumStage(
        name="reset4_independent_0050",
        weight=0.30,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
    ),
)

FRICTION_V1_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="friction_02",
        weight=0.20,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.98, 1.02),
        ),
    ),
    Rolling3DCurriculumStage(
        name="friction_05",
        weight=0.30,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.95, 1.05),
        ),
    ),
    Rolling3DCurriculumStage(
        name="friction_10",
        weight=0.50,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.90, 1.10),
        ),
    ),
)

FLOOR_FRICTION_V2_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="floor_friction_02",
        weight=0.20,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
        domain_randomization=Rolling3DDomainRandomization(
            floor_friction_scale=(0.98, 1.02),
        ),
    ),
    Rolling3DCurriculumStage(
        name="floor_friction_05",
        weight=0.30,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
        domain_randomization=Rolling3DDomainRandomization(
            floor_friction_scale=(0.95, 1.05),
        ),
    ),
    Rolling3DCurriculumStage(
        name="floor_friction_10",
        weight=0.50,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
        domain_randomization=Rolling3DDomainRandomization(
            floor_friction_scale=(0.90, 1.10),
        ),
    ),
)


FLOOR_MASS_V2_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="floor_mass_02",
        weight=0.30,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
        domain_randomization=Rolling3DDomainRandomization(
            floor_friction_scale=(0.90, 1.10),
            body_mass_scale=(0.98, 1.02),
        ),
    ),
    Rolling3DCurriculumStage(
        name="floor_mass_05",
        weight=0.70,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
        domain_randomization=Rolling3DDomainRandomization(
            floor_friction_scale=(0.90, 1.10),
            body_mass_scale=(0.95, 1.05),
        ),
    ),
)


FLOOR_MASS_GAIN_V3_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="floor_mass_gain_02",
        weight=0.30,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
        domain_randomization=Rolling3DDomainRandomization(
            floor_friction_scale=(0.90, 1.10),
            body_mass_scale=(0.95, 1.05),
            actuator_gain_scale=(0.98, 1.02),
        ),
    ),
    Rolling3DCurriculumStage(
        name="floor_mass_gain_05",
        weight=0.70,
        reset_joint_noise_rad=0.005,
        reset_velocity_noise=0.005,
        reset_root_velocity_noise=0.0,
        reset_independent=True,
        reset_axis_tilt_noise_rad=0.0,
        domain_randomization=Rolling3DDomainRandomization(
            floor_friction_scale=(0.90, 1.10),
            body_mass_scale=(0.95, 1.05),
            actuator_gain_scale=(0.95, 1.05),
        ),
    ),
)

FRICTION_LOW_V1_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="friction_low_090",
        weight=0.30,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.90, 1.10),
        ),
    ),
    Rolling3DCurriculumStage(
        name="friction_low_080",
        weight=0.40,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.80, 1.10),
        ),
    ),
    Rolling3DCurriculumStage(
        name="friction_low_070",
        weight=0.30,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.70, 1.10),
        ),
    ),
)


MASS_V1_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="mass_02",
        weight=0.30,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.90, 1.10),
            body_mass_scale=(0.98, 1.02),
        ),
    ),
    Rolling3DCurriculumStage(
        name="mass_05",
        weight=0.70,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.90, 1.10),
            body_mass_scale=(0.95, 1.05),
        ),
    ),
)


PHYSICS_STAGES_3D = (
    Rolling3DCurriculumStage(
        name="friction",
        weight=0.20,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.90, 1.10),
        ),
    ),
    Rolling3DCurriculumStage(
        name="dynamics",
        weight=0.20,
        reset_joint_noise_rad=0.015,
        reset_velocity_noise=0.030,
        reset_pair_differential_scale=0.25,
        reset_axis_tilt_noise_rad=0.030,
        domain_randomization=Rolling3DDomainRandomization(
            geom_friction_scale=(0.90, 1.10),
            body_mass_scale=(0.95, 1.05),
            actuator_gain_scale=(0.95, 1.05),
        ),
    ),
)

CURRICULUM_STAGE_NAMES_3D = tuple(
    stage.name
    for stage in (
        *RESET_STAGES_3D,
        *RESET_V2_STAGES_3D,
        *NOMINAL_RESET_V3_STAGES_3D,
        *INDEPENDENT_RESET_V4_STAGES_3D,
        *FRICTION_V1_STAGES_3D,
        *FLOOR_FRICTION_V2_STAGES_3D,
        *FLOOR_MASS_V2_STAGES_3D,
        *FLOOR_MASS_GAIN_V3_STAGES_3D,
        *FRICTION_LOW_V1_STAGES_3D,
        *MASS_V1_STAGES_3D,
        *PHYSICS_STAGES_3D,
    )
)


def curriculum_stages_3d(
    name: str,
    *,
    only_stage: str | None = None,
) -> tuple[Rolling3DCurriculumStage, ...]:
    if name == "none":
        if only_stage is not None:
            raise ValueError("only_stage requires a nontrivial curriculum")
        return (Rolling3DCurriculumStage(name="nominal", weight=1.0),)
    if name == "reset_v1":
        stages = RESET_STAGES_3D
    elif name == "reset_v2":
        stages = RESET_V2_STAGES_3D
    elif name == "nominal_reset_v3":
        stages = NOMINAL_RESET_V3_STAGES_3D
    elif name == "independent_reset_v4":
        stages = INDEPENDENT_RESET_V4_STAGES_3D
    elif name == "friction_v1":
        stages = FRICTION_V1_STAGES_3D
    elif name == "floor_friction_v2":
        stages = FLOOR_FRICTION_V2_STAGES_3D
    elif name == "floor_mass_v2":
        stages = FLOOR_MASS_V2_STAGES_3D
    elif name == "floor_mass_gain_v3":
        stages = FLOOR_MASS_GAIN_V3_STAGES_3D
    elif name == "friction_low_v1":
        stages = FRICTION_LOW_V1_STAGES_3D
    elif name == "mass_v1":
        stages = MASS_V1_STAGES_3D
    elif name == "robustness_v1":
        reset_weight_scale = 0.60 / sum(
            stage.weight for stage in RESET_STAGES_3D
        )
        stages = tuple(
            replace(stage, weight=stage.weight * reset_weight_scale)
            for stage in RESET_STAGES_3D
        ) + PHYSICS_STAGES_3D
    else:
        raise ValueError(f"unknown 3-D curriculum: {name}")
    if only_stage is None:
        return stages
    selected = tuple(stage for stage in stages if stage.name == only_stage)
    if not selected:
        raise ValueError(
            f"stage {only_stage!r} is not part of curriculum {name!r}"
        )
    return (replace(selected[0], weight=1.0),)
