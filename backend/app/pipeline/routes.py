from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class InferredRoute:
    route_id: str
    centerline: list[tuple[float, float]]
    support_tracks: list[int]
    width_m: float


def _smooth_and_reject_spikes(points: list[tuple[float, float]]) -> np.ndarray:
    """Median-gate localization spikes, then smooth without moving route endpoints."""
    values = np.asarray(points, dtype=float)
    if len(values) < 4:
        return values
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    median = float(np.median(steps))
    mad = float(np.median(np.abs(steps - median)))
    keep = np.concatenate(([True], steps <= max(2.0, median + 5.0 * max(mad, 0.05))))
    values = values[keep]
    if len(values) >= 7:
        window = min(11, len(values) if len(values) % 2 else len(values) - 1)
        values = np.column_stack(
            [savgol_filter(values[:, axis], window, 2, mode="interp") for axis in range(2)]
        )
    return values


def resample_polyline(points: list[tuple[float, float]] | np.ndarray, count: int = 30) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(values) == 1:
        return np.repeat(values, count, axis=0)
    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] < 1e-6:
        return np.repeat(values[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, values[:, axis]) for axis in range(2)]
    )


def _principal_axis(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def _fit_lane_centerline(
    tracks: list[np.ndarray],
    axis: np.ndarray,
    normal: np.ndarray,
    origin: np.ndarray,
    road_curve: np.ndarray,
    count: int = 40,
) -> tuple[np.ndarray, float]:
    """Offset the common road curve to form a parallel, smooth lane centerline."""
    values = np.concatenate(tracks)
    longitudinal = (values - origin) @ axis
    lateral = (values - origin) @ normal
    residual = lateral - np.polyval(road_curve, longitudinal)
    lane_offset = float(np.median(residual))
    inlier = np.abs(residual - lane_offset) <= max(
        0.7,
        3.5 * 1.4826 * float(np.median(np.abs(residual - lane_offset))),
    )
    lo, hi = np.percentile(longitudinal[inlier], [2, 98])
    samples = np.linspace(float(lo), float(hi), count)
    offsets = np.polyval(road_curve, samples) + lane_offset
    centerline = origin + samples[:, None] * axis + offsets[:, None] * normal
    lane_width = float(np.clip(2.0 + 2.0 * np.percentile(np.abs(residual[inlier] - lane_offset), 90), 3.2, 4.4))
    return centerline, lane_width


def infer_routes(
    trajectories: dict[int, list[tuple[float, float]]],
    minimum_points: int = 8,
    threshold_m: float = 4.5,
) -> list[InferredRoute]:
    """Infer directed lane centerlines in the calibrated metric road plane."""
    valid = {
        track_id: _smooth_and_reject_spikes(points)
        for track_id, points in trajectories.items()
        if len(points) >= minimum_points
    }
    valid = {
        track_id: points
        for track_id, points in valid.items()
        if len(points) >= minimum_points and np.linalg.norm(points[-1] - points[0]) >= 3.0
    }
    if not valid:
        return []

    ids = list(valid)
    all_values = np.concatenate(list(valid.values()))
    global_axis = _principal_axis(all_values)
    normal = np.asarray([-global_axis[1], global_axis[0]])
    origin = np.median(all_values, axis=0)
    all_longitudinal = (all_values - origin) @ global_axis
    all_lateral = (all_values - origin) @ normal
    curve_degree = 2 if len(all_values) >= 20 and float(np.ptp(all_lateral)) > 0.5 else 1
    road_curve = np.polyfit(all_longitudinal, all_lateral, curve_degree)
    features = []
    for track_id in ids:
        points = valid[track_id]
        lateral = float(np.median((points - origin) @ normal))
        direction = 1.0 if float((points[-1] - points[0]) @ global_axis) >= 0 else -1.0
        features.append([lateral, direction * threshold_m * 2.5])

    if len(ids) == 1:
        labels = np.ones(1, dtype=int)
    else:
        labels = fcluster(
            linkage(np.asarray(features), method="complete"),
            t=threshold_m,
            criterion="distance",
        )

    routes: list[InferredRoute] = []
    clusters = sorted(
        set(labels),
        key=lambda label: float(np.mean([features[i][0] for i in np.flatnonzero(labels == label)])),
    )
    for route_index, label in enumerate(clusters, start=1):
        indexes = np.flatnonzero(labels == label)
        centerline, width = _fit_lane_centerline(
            [valid[ids[index]] for index in indexes],
            global_axis,
            normal,
            origin,
            road_curve,
        )
        routes.append(
            InferredRoute(
                route_id=f"R{route_index:02d}",
                centerline=[(round(float(x), 3), round(float(y), 3)) for x, y in centerline],
                support_tracks=[ids[index] for index in indexes],
                width_m=round(width, 2),
            )
        )
    return routes
