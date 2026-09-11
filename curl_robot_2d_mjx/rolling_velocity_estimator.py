"""Standalone rolling speed / trajectory turn-rate estimation (NumPy runtime).

The 36-value deployment frame is unchanged. Only gyro, gravity, joint offsets
and previous actions enter the estimator; commands never enter its features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

FEATURE_INDICES = np.r_[0:6, 12:36]
OUTPUT_NAMES = ("rolling_speed_m_s", "trajectory_turn_rate_rad_s")


@dataclass(frozen=True)
class EstimatorConfig:
    history: int = 20
    control_dt: float = 1.0 / 52.0
    velocity_filter_tau_s: float = 0.06
    min_turn_speed_m_s: float = 0.08
    min_axis_projection: float = 0.2

    def __post_init__(self):
        if not isinstance(self.history, int) or self.history < 1:
            raise ValueError("history must be a positive integer")
        for name in ("control_dt", "min_turn_speed_m_s", "min_axis_projection"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.velocity_filter_tau_s < np.inf:
            raise ValueError("velocity_filter_tau_s must be finite and nonnegative")
        if self.min_axis_projection > 1:
            raise ValueError("min_axis_projection must be <= 1")


class MotionLabeler:
    """Causal labels at the current observation time; reset every episode.

    Speed = dot(raw world root velocity, normalize(body_y cross world_z)).
    Turn = signed angle between successive EMA horizontal velocities / dt.
    Masked values are stored as zero, never interpreted as zero-rate truth.
    """

    def __init__(self, config: EstimatorConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.filtered = None
        self.previous_raw_speed = 0.0

    def update(self, velocity_world, body_y_world):
        velocity = np.asarray(velocity_world, dtype=np.float64)
        axis = np.asarray(body_y_world, dtype=np.float64)
        if velocity.shape != (3,) or axis.shape != (3,):
            raise ValueError("velocity and body_y_world must have shape (3,)")
        if not np.all(np.isfinite(np.r_[velocity, axis])):
            raise ValueError("nonfinite label state")
        c = self.config
        axis_norm = np.linalg.norm(axis[:2])
        speed_valid = axis_norm >= c.min_axis_projection
        heading = np.array([axis[1], -axis[0]]) / max(axis_norm, 1e-12)
        speed = float(velocity[:2] @ heading) if speed_valid else 0.0
        alpha = 1.0 if c.velocity_filter_tau_s == 0 else -np.expm1(
            -c.control_dt / c.velocity_filter_tau_s
        )
        previous = self.filtered
        filtered = velocity[:2].copy() if previous is None else (
            previous + alpha * (velocity[:2] - previous)
        )
        raw_speed = float(np.linalg.norm(velocity[:2]))
        turn_valid = previous is not None and min(
            self.previous_raw_speed, raw_speed,
            np.linalg.norm(previous), np.linalg.norm(filtered),
        ) >= c.min_turn_speed_m_s
        turn = 0.0
        if turn_valid:
            cross = previous[0] * filtered[1] - previous[1] * filtered[0]
            turn = float(np.arctan2(cross, previous @ filtered) / c.control_dt)
        self.filtered = filtered
        self.previous_raw_speed = raw_speed
        return np.array([speed, turn], np.float32), np.array(
            [speed_valid, turn_valid], dtype=bool
        )


class ObservationHistory:
    """Newest-first history, with no prediction until a complete window exists."""

    def __init__(self, length: int):
        if length < 1:
            raise ValueError("history length must be positive")
        self.values = np.zeros((length, len(FEATURE_INDICES)), np.float32)
        self.count = 0

    def reset(self):
        self.values.fill(0)
        self.count = 0

    def update(self, frame):
        frame = np.asarray(frame, np.float32)
        if frame.shape != (36,) or not np.all(np.isfinite(frame)):
            raise ValueError("expected one finite raw 36-value deployment frame")
        self.values[1:] = self.values[:-1].copy()
        self.values[0] = frame[FEATURE_INDICES]
        self.count += 1
        return self.values.ravel().copy(), self.count >= len(self.values)


def prepare_dataset(path: Path, config: EstimatorConfig):
    """Build episode-local causal windows. Input NPZ contains raw rollout states."""
    with np.load(path, allow_pickle=False) as archive:
        required = ("frames", "velocity_world", "body_y_world", "episode", "time_s")
        arrays = {key: archive[key] for key in required}
        metadata = json.loads(str(archive["metadata_json"]))
    n = len(arrays["frames"])
    shapes = {"frames": (n, 36), "velocity_world": (n, 3),
              "body_y_world": (n, 3), "episode": (n,), "time_s": (n,)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"invalid dataset array: {key}, expected {shape}")
    if not np.isclose(metadata["control_dt"], config.control_dt, rtol=1e-5):
        raise ValueError("dataset and estimator control_dt disagree")
    if not np.all(arrays["episode"] == arrays["episode"].astype(np.int64)):
        raise ValueError("episode IDs must be integers")
    features, targets, masks, episode_ids, source_rows = [], [], [], [], []
    history, labeler = ObservationHistory(config.history), MotionLabeler(config)
    # Group by entire episode, preserving temporal ordering; never mix windows.
    for episode in np.unique(arrays["episode"]):
        rows = np.flatnonzero(arrays["episode"] == episode)
        delta = np.diff(arrays["time_s"][rows])
        if not np.allclose(delta, config.control_dt, rtol=1e-4, atol=1e-8):
            raise ValueError(f"episode {episode}: irregular or unordered timestamps")
        history.reset()
        labeler.reset()
        for row in rows:
            x, ready = history.update(arrays["frames"][row])
            y, valid = labeler.update(arrays["velocity_world"][row], arrays["body_y_world"][row])
            if ready:
                features.append(x)
                targets.append(y)
                masks.append(valid)
                episode_ids.append(int(episode))
                source_rows.append(row)
    if not features:
        raise ValueError("no complete observation windows in dataset")
    return dict(x=np.asarray(features), y=np.asarray(targets), mask=np.asarray(masks),
                episode=np.asarray(episode_ids), source_row=np.asarray(source_rows),
                metadata=metadata)


def split_episodes(episode_ids, seed=0):
    """60/20/20 split for tiny runs, approaching 80/10/10 for larger runs."""
    episodes = np.unique(episode_ids)
    if len(episodes) < 5:
        raise ValueError("at least five complete episodes required for train/validation/test")
    episodes = np.random.default_rng(seed).permutation(episodes)
    holdout = max(1, int(round(0.1 * len(episodes))))
    groups = (episodes[2 * holdout:], episodes[:holdout], episodes[holdout:2 * holdout])
    return {name: np.isin(episode_ids, group)
            for name, group in zip(("train", "validation", "test"), groups)}


def regression_metrics(prediction, target, mask):
    result = {}
    for column, name in enumerate(OUTPUT_NAMES):
        error = (prediction[:, column] - target[:, column])[mask[:, column]]
        result[name] = {"samples": int(error.size),
                        "rmse": float(np.sqrt(np.mean(error**2))) if error.size else None,
                        "mae": float(np.mean(np.abs(error))) if error.size else None,
                        "bias": float(np.mean(error)) if error.size else None}
    return result


def save_estimator(path, config, layers, x_mean, x_std, y_mean, y_std, provenance):
    arrays = dict(config_json=np.array(json.dumps(asdict(config))),
                  provenance_json=np.array(json.dumps(provenance)),
                  schema_version=np.array(1), feature_indices=FEATURE_INDICES,
                  output_names=np.asarray(OUTPUT_NAMES),
                  x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                  layer_count=np.array(len(layers)))
    for i, (weight, bias) in enumerate(layers):
        arrays[f"weight_{i}"] = np.asarray(weight, np.float32)
        arrays[f"bias_{i}"] = np.asarray(bias, np.float32)
    np.savez_compressed(path, **arrays)


class RollingVelocityEstimator:
    """CPU inference needs only NumPy. Call reset() whenever a stream restarts.

    Turn validity at runtime is NOT known from simulator masks. The estimate
    is only meaningful during translational motion above the training cutoff.
    """

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["schema_version"]) != 1:
                raise ValueError("unsupported estimator schema")
            self.config = EstimatorConfig(**json.loads(str(archive["config_json"])))
            if not np.array_equal(archive["feature_indices"], FEATURE_INDICES):
                raise ValueError("incompatible estimator features")
            self.layers = [(archive[f"weight_{i}"], archive[f"bias_{i}"])
                           for i in range(int(archive["layer_count"]))]
            self.x_mean, self.x_std = archive["x_mean"], archive["x_std"]
            self.y_mean, self.y_std = archive["y_mean"], archive["y_std"]
        self.history = ObservationHistory(self.config.history)

    def reset(self):
        self.history.reset()

    def predict_features(self, features):
        features = np.asarray(features, np.float32)
        if features.shape[-1] != self.config.history * len(FEATURE_INDICES):
            raise ValueError("wrong estimator feature count")
        if not np.all(np.isfinite(features)):
            raise ValueError("nonfinite estimator input")
        value = (features - self.x_mean) / self.x_std
        for i, (weight, bias) in enumerate(self.layers):
            value = value @ weight + bias
            if i + 1 < len(self.layers):
                value = np.maximum(value, 0)
        return value * self.y_std + self.y_mean

    def update(self, frame):
        features, ready = self.history.update(frame)
        return self.predict_features(features) if ready else None
