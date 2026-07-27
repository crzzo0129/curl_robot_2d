"""Dependency-light configuration for nominal-COM rolling RL."""

from __future__ import annotations

from dataclasses import dataclass, replace


PHYSICS_PROFILE_NAMES = (
    "reference",
    "newton4",
    "cg12",
)


@dataclass(frozen=True)
class NominalRLConfig:
    """Task constants shared by smoke tests and PPO training.

    COM, mass, inertia, friction, gains and torque limits all come directly
    from ``assets/curl_robot_2d.xml``.  This first RL stage deliberately does
    not randomize any of them.
    """

    physics_profile: str = "reference"
    physics_timestep: float = 0.001
    solver_name: str = "newton"
    integrator_name: str = "implicitfast"
    cone_name: str = "elliptic"
    jacobian_name: str = "dense"
    action_repeat: int = 20
    episode_length: int = 500
    action_scales: tuple[float, float, float, float] = (
        0.8,
        1.2,
        0.8,
        1.2,
    )
    reset_joint_noise_rad: float = 0.01
    reset_velocity_noise: float = 0.01

    terminate_root_z_min: float = 0.06
    terminate_root_z_max: float = 0.70
    maximum_foot_center_distance_m: float = 0.28

    # MJX-JAX is much faster with a small Newton iteration count.  The final
    # policy must still be replayed in the unmodified 20/10 CPU MuJoCo model.
    solver_iterations: int = 20
    solver_ls_iterations: int = 10

    @property
    def control_timestep(self) -> float:
        return self.physics_timestep * self.action_repeat


def physics_profile(
    name: str,
    config: NominalRLConfig | None = None,
) -> NominalRLConfig:
    """Apply a measured MJX physics profile without changing the XML."""

    base = config or NominalRLConfig()
    if name == "reference":
        return replace(
            base,
            physics_profile="reference",
            physics_timestep=0.001,
            solver_name="newton",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=20,
            solver_ls_iterations=10,
        )
    if name == "newton4":
        return replace(
            base,
            physics_profile="newton4",
            physics_timestep=0.001,
            solver_name="newton",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=4,
            solver_ls_iterations=4,
        )
    if name == "cg12":
        return replace(
            base,
            physics_profile="cg12",
            physics_timestep=0.001,
            solver_name="cg",
            integrator_name="implicitfast",
            cone_name="elliptic",
            jacobian_name="dense",
            action_repeat=20,
            solver_iterations=12,
            solver_ls_iterations=6,
        )
    raise ValueError(f"unknown physics profile: {name}")
