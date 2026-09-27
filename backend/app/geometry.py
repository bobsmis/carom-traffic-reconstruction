from __future__ import annotations

from collections.abc import Sequence
from math import cos, radians, sin

import numpy as np


Point = tuple[float, float]


def solve_homography(image_points: Sequence[Point], ground_points: Sequence[Point]) -> np.ndarray:
    """Return a least-squares homography mapping image pixels onto the ground plane.

    CAROM labels more than four road correspondences.  Solving the overdetermined
    system keeps all lane/stop-line observations in the calibration instead of
    silently discarding every point after the four corners.
    """
    if len(image_points) != len(ground_points) or len(image_points) < 4:
        raise ValueError("at least four paired point correspondences are required")

    rows: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(image_points, ground_points, strict=True):
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v])
        values.append(x)
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v])
        values.append(y)

    coefficients, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    return np.append(coefficients, 1.0).reshape(3, 3)


def project_points(homography: np.ndarray, points: Sequence[Point]) -> list[Point]:
    """Project pixel coordinates through a homography and normalize them."""
    if not points:
        return []
    pixels = np.asarray([[u, v, 1.0] for u, v in points], dtype=float).T
    projected = homography @ pixels
    if np.any(np.isclose(projected[2], 0.0)):
        raise ValueError("point projects to infinity")
    normalized = projected[:2] / projected[2]
    return [(float(x), float(y)) for x, y in normalized.T]


def vehicle_footprint(center: Point, length_m: float, width_m: float, heading_deg: float) -> list[Point]:
    """Build an oriented metric footprint; heading 0 points along world +Y."""
    angle = radians(heading_deg)
    forward = np.asarray([sin(angle), cos(angle)])
    left = np.asarray([-cos(angle), sin(angle)])
    origin = np.asarray(center, dtype=float)
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    return [
        tuple(origin + half_length * forward + half_width * left),
        tuple(origin + half_length * forward - half_width * left),
        tuple(origin - half_length * forward - half_width * left),
        tuple(origin - half_length * forward + half_width * left),
    ]


def velocity_vector(speed_kmh: float, heading_deg: float) -> Point:
    """Convert scalar speed/heading into world-frame metres per second."""
    speed_mps = speed_kmh / 3.6
    angle = radians(heading_deg)
    return speed_mps * sin(angle), speed_mps * cos(angle)
