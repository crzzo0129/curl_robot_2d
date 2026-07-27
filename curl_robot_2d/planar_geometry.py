"""Small, dependency-free geometry helpers for sagittal-plane diagnostics."""

from __future__ import annotations

import numpy as np


def _as_point(value: np.ndarray) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,):
        raise ValueError("planar point must have shape (2,)")
    return point


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def proper_segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Return true only for an interior/interior segment crossing.

    Endpoint touching is deliberately excluded.  This lets the compact pose's
    two shank centerlines meet at the intended lower pentagon vertex without
    being counted as a leg crossing.
    """

    a = _as_point(first_start)
    b = _as_point(first_end)
    c = _as_point(second_start)
    d = _as_point(second_end)
    first_side_c = _cross_2d(b - a, c - a)
    first_side_d = _cross_2d(b - a, d - a)
    second_side_a = _cross_2d(d - c, a - c)
    second_side_b = _cross_2d(d - c, b - c)
    return (
        first_side_c * first_side_d < -(tolerance**2)
        and second_side_a * second_side_b < -(tolerance**2)
    )


def point_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    point = _as_point(point)
    start = _as_point(segment_start)
    end = _as_point(segment_end)
    direction = end - start
    squared_length = float(direction @ direction)
    if squared_length <= np.finfo(float).eps:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ direction) / squared_length, 0.0, 1.0))
    closest = start + fraction * direction
    return float(np.linalg.norm(point - closest))


def segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Return the minimum Euclidean distance between two planar segments."""

    if proper_segments_intersect(
        first_start, first_end, second_start, second_end
    ):
        return 0.0
    return min(
        point_segment_distance(first_start, second_start, second_end),
        point_segment_distance(first_end, second_start, second_end),
        point_segment_distance(second_start, first_start, first_end),
        point_segment_distance(second_end, first_start, first_end),
    )


def trim_segment_distal(
    start: np.ndarray,
    end: np.ndarray,
    distal_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim a fraction from the distal end while preserving the start."""

    if not 0.0 <= distal_fraction < 1.0:
        raise ValueError("distal_fraction must be in [0, 1)")
    start_point = _as_point(start)
    end_point = _as_point(end)
    trimmed_end = start_point + (1.0 - distal_fraction) * (
        end_point - start_point
    )
    return start_point, trimmed_end
