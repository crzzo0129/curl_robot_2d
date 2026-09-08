"""CEM state matcher: find the CEM phase that best matches a query state.

This is the stand-to-roll "capture detector". Given the mature CEM rollout
cycle data (q, q_dot, R, omega vs phase theta), it finds the phase theta* whose
reference state is closest to the current robot state, together with the
distance D_min. The RL transition is considered "captured" when D_min drops
below a threshold while the robot is rolling forward.

The distance is a weighted sum of normalized squared errors:

    D = w_q D_q + w_qd D_qd + w_R D_R + w_w D_w

Only the *moving* joints (the eight hip/knee joints) participate in D_q / D_qd:
the four abduction joints are locked, and their tiny cycle std would otherwise
blow up any small difference after normalization. D_R is the squared geodesic
(relative-rotation) angle and is the most reliable phase disambiguator, so it
is weighted most heavily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# The 12 joints (8 rolling + 4 abduction) live after the free-joint DOFs:
# qpos: 7 free-joint values (3 pos + 4 quat) then 12 joints.
# qvel: 6 free-joint values (3 lin + 3 ang) then 12 joints.
QPOS_JOINT_SLICE = slice(7, 19)
QVEL_JOINT_SLICE = slice(6, 18)

# A joint with cycle std below this is treated as locked (abduction) and is
# excluded from the position/velocity distance terms.
LOCKED_JOINT_STD_RAD = np.radians(1.0)
SIGMA_FLOOR_Q_RAD = np.radians(1.0)
SIGMA_FLOOR_Q_DOT = 0.1
SIGMA_FLOOR_OMEGA = 0.1


@dataclass(frozen=True)
class CEMMatcherWeights:
    """Per-term weights for the CEM state distance."""

    q: float = 1.0
    q_dot: float = 0.5
    orientation: float = 5.0
    angular_velocity: float = 0.5


def geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Angle (rad) of the relative rotation between two rotation matrices.

    ``R1`` is a single (3, 3) matrix; ``R2`` may be (..., 3, 3).
    """

    relative = np.einsum("ij,...jk->...ik", R1.T, R2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


class CEMStateMatcher:
    """Bucket the CEM cycle by phase and match query states to phase."""

    def __init__(
        self,
        npz_path: Path,
        num_bins: int = 200,
        weights: CEMMatcherWeights = CEMMatcherWeights(),
    ):
        data = np.load(npz_path)
        self.phase = np.asarray(data["cem_phase_wrapped"])
        self.q = np.asarray(data["qpos"])[:, QPOS_JOINT_SLICE]
        self.q_dot = np.asarray(data["qvel"])[:, QVEL_JOINT_SLICE]
        self.orientation = np.asarray(data["orientation"])
        self.angular_velocity = np.asarray(data["angular_velocity"])
        self.weights = weights

        # Identify moving vs locked joints from the cycle std of positions.
        std_q = self.q.std(axis=0)
        self.active_mask = std_q > LOCKED_JOINT_STD_RAD
        self.active_indices = np.flatnonzero(self.active_mask)
        self.q_active = self.q[:, self.active_mask]
        self.q_dot_active = self.q_dot[:, self.active_mask]

        self.num_bins = int(num_bins)
        self.bin_edges = np.linspace(0.0, 2.0 * np.pi, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_idx = np.clip(
            np.digitize(self.phase, self.bin_edges) - 1, 0, self.num_bins - 1
        )

        n_active = int(self.active_mask.sum())
        self.q_mean = np.zeros((self.num_bins, n_active))
        self.q_dot_mean = np.zeros((self.num_bins, n_active))
        self.omega_mean = np.zeros((self.num_bins, 3))
        self.orientation_ref = np.zeros((self.num_bins, 3, 3))
        self.bin_counts = np.zeros(self.num_bins, dtype=int)

        for b in range(self.num_bins):
            mask = self.bin_idx == b
            self.bin_counts[b] = int(mask.sum())
            if self.bin_counts[b] == 0:
                continue
            self.q_mean[b] = self.q_active[mask].mean(axis=0)
            self.q_dot_mean[b] = self.q_dot_active[mask].mean(axis=0)
            self.omega_mean[b] = self.angular_velocity[mask].mean(axis=0)
            closest = np.argmin(np.abs(self.phase[mask] - self.bin_centers[b]))
            self.orientation_ref[b] = self.orientation[mask][closest]


        self.sigma_q = np.maximum(
            self.q_active.std(axis=0), SIGMA_FLOOR_Q_RAD
        )
        self.sigma_q_dot = np.maximum(
            self.q_dot_active.std(axis=0), SIGMA_FLOOR_Q_DOT
        )
        self.sigma_omega = np.maximum(
            self.angular_velocity.std(axis=0), SIGMA_FLOOR_OMEGA
        )

    def distance_breakdown(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return the per-term distance vectors across all bins."""

        q_a = np.asarray(q)[self.active_mask]
        q_dot_a = np.asarray(q_dot)[self.active_mask]
        d_q = np.mean(
            ((q_a[None, :] - self.q_mean) / self.sigma_q[None, :]) ** 2, axis=1
        )
        d_q_dot = np.mean(
            (
                (q_dot_a[None, :] - self.q_dot_mean)
                / self.sigma_q_dot[None, :]
            )
            ** 2,
            axis=1,
        )
        d_omega = np.mean(
            (
                (angular_velocity[None, :] - self.omega_mean)
                / self.sigma_omega[None, :]
            )
            ** 2,
            axis=1,
        )
        d_orientation = geodesic_angle(orientation, self.orientation_ref) ** 2
        valid = self.bin_counts > 0
        return {
            "q": np.where(valid, d_q, np.inf),
            "q_dot": np.where(valid, d_q_dot, np.inf),
            "orientation": np.where(valid, d_orientation, np.inf),
            "angular_velocity": np.where(valid, d_omega, np.inf),
        }

    def distance(self, q, q_dot, orientation, angular_velocity) -> np.ndarray:
        """Return the total distance D across all bins."""

        terms = self.distance_breakdown(q, q_dot, orientation, angular_velocity)
        return (
            self.weights.q * terms["q"]
            + self.weights.q_dot * terms["q_dot"]
            + self.weights.orientation * terms["orientation"]
            + self.weights.angular_velocity * terms["angular_velocity"]
        )

    def match(self, q, q_dot, orientation, angular_velocity) -> tuple[float, float]:
        """Return ``(theta_star, D_min)`` for a query state."""

        D = self.distance(q, q_dot, orientation, angular_velocity)
        best = int(np.argmin(D))
        return float(self.bin_centers[best]), float(D[best])


def load_matcher(npz_path: Path, num_bins: int = 200) -> CEMStateMatcher:
    return CEMStateMatcher(npz_path, num_bins=num_bins)


def build_reference_dict(matcher: CEMStateMatcher) -> dict[str, np.ndarray]:
    """Return the matcher's bucketed reference as plain arrays.

    The returned dict is host-side (numpy) and is meant to be converted to JAX
    arrays once for use inside the RL environment step.
    """

    return {
        "q_mean": matcher.q_mean,
        "q_dot_mean": matcher.q_dot_mean,
        "omega_mean": matcher.omega_mean,
        "orientation_ref": matcher.orientation_ref,
        "sigma_q": matcher.sigma_q,
        "sigma_q_dot": matcher.sigma_q_dot,
        "sigma_omega": matcher.sigma_omega,
        "bin_centers": matcher.bin_centers,
        "active_mask": matcher.active_mask,
        "active_indices": matcher.active_indices,
        "valid": (matcher.bin_counts > 0),
        "w_q": matcher.weights.q,
        "w_q_dot": matcher.weights.q_dot,
        "w_orientation": matcher.weights.orientation,
        "w_angular_velocity": matcher.weights.angular_velocity,
    }


def cem_distance_xp(xp, reference, q, q_dot, orientation, angular_velocity):
    """Array-module-agnostic CEM distance across all bins.

    ``reference`` is the dict produced by :func:`build_reference_dict` (or an
    equivalent dict of ``xp`` arrays). ``q`` / ``q_dot`` are the full 12-joint
    position/velocity vectors; the locked abduction joints are dropped via
    ``active_mask``. ``orientation`` is a (3, 3) rotation matrix.
    """

    # Integer gather works under jax.jit; traced boolean indexing does not.
    active = xp.asarray(reference["active_indices"])
    q_a = xp.take(xp.asarray(q), active, axis=0)
    q_dot_a = xp.take(xp.asarray(q_dot), active, axis=0)
    d_q = xp.mean(
        ((q_a[None, :] - reference["q_mean"]) / reference["sigma_q"][None, :])
        ** 2,
        axis=1,
    )
    d_q_dot = xp.mean(
        (
            (q_dot_a[None, :] - reference["q_dot_mean"])
            / reference["sigma_q_dot"][None, :]
        )
        ** 2,
        axis=1,
    )
    d_omega = xp.mean(
        (
            (xp.asarray(angular_velocity)[None, :] - reference["omega_mean"])
            / reference["sigma_omega"][None, :]
        )
        ** 2,
        axis=1,
    )
    relative = xp.einsum(
        "ij,...jk->...ik", xp.asarray(orientation).T, reference["orientation_ref"]
    )
    trace = xp.trace(relative, axis1=-2, axis2=-1)
    d_orientation = xp.arccos(xp.clip((trace - 1.0) / 2.0, -1.0, 1.0)) ** 2
    distance = (
        reference["w_q"] * d_q
        + reference["w_q_dot"] * d_q_dot
        + reference["w_orientation"] * d_orientation
        + reference["w_angular_velocity"] * d_omega
    )
    return xp.where(xp.asarray(reference["valid"]), distance, xp.inf)


def cem_match_xp(xp, reference, q, q_dot, orientation, angular_velocity):
    """Return ``(theta_star, D_min)`` for a query state (numpy or jax)."""

    distance = cem_distance_xp(xp, reference, q, q_dot, orientation, angular_velocity)
    best = xp.argmin(distance)
    return reference["bin_centers"][best], distance[best]


def _random_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula for a rotation matrix from an axis and angle."""

    axis = axis / np.linalg.norm(axis)
    K = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def self_match_report(
    matcher: CEMStateMatcher,
    num_samples: int = 100,
    q_noise: float = 0.10,
    q_dot_noise: float = 0.50,
    orientation_noise: float = 0.10,
    omega_noise: float = 0.30,
    seed: int = 0,
) -> dict[str, float]:
    """Perturb random CEM frames and measure phase-recovery accuracy."""

    rng = np.random.default_rng(seed)
    data_len = matcher.phase.shape[0]
    errors = []
    for _ in range(num_samples):
        i = int(rng.integers(0, data_len))
        q = matcher.q[i] + rng.normal(0.0, q_noise, size=matcher.q.shape[1])
        q_dot = matcher.q_dot[i] + rng.normal(
            0.0, q_dot_noise, size=matcher.q_dot.shape[1]
        )
        omega = matcher.angular_velocity[i] + rng.normal(
            0.0, omega_noise, size=3
        )
        axis = rng.normal(size=3)
        orientation = (
            _random_rotation(axis, rng.normal(0.0, orientation_noise))
            @ matcher.orientation[i]
        )
        theta_star, _ = matcher.match(q, q_dot, orientation, omega)
        true_theta = matcher.phase[i]
        error = np.abs(
            (theta_star - true_theta + np.pi) % (2.0 * np.pi) - np.pi
        )
        errors.append(float(error))

    errors = np.asarray(errors)
    return {
        "num_samples": num_samples,
        "mean_abs_phase_error_rad": float(np.mean(errors)),
        "median_abs_phase_error_rad": float(np.median(errors)),
        "max_abs_phase_error_rad": float(np.max(errors)),
        "p90_abs_phase_error_rad": float(np.percentile(errors, 90)),
        "within_0p1_rad": float(np.mean(errors < 0.1)),
        "within_0p2_rad": float(np.mean(errors < 0.2)),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("results/cem_cycle_data/cem_cycles.npz"),
    )
    parser.add_argument("--num-bins", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matcher = load_matcher(args.npz, num_bins=args.num_bins)
    report = self_match_report(matcher, num_samples=args.num_samples, seed=args.seed)
    print("CEM state matcher self-match (noise) report:")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
