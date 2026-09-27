from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, hypot

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


@dataclass
class KalmanTrack:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 4.0)
    positions: list[tuple[int, float, float]] = field(default_factory=list)
    heading_samples: list[tuple[int, float]] = field(default_factory=list)
    last_frame: int | None = None

    @classmethod
    def create(cls, track_id: int, x: float, y: float) -> "KalmanTrack":
        return cls(track_id=track_id, state=np.asarray([x, y, 0.0, 0.0], dtype=float))

    def update(self, frame_index: int, x: float, y: float, dt: float) -> dict[str, float | str]:
        elapsed = dt if self.last_frame is None else max(dt, (frame_index - self.last_frame) * dt)
        self.last_frame = frame_index
        transition = np.asarray(
            [[1, 0, elapsed, 0], [0, 1, 0, elapsed], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        observation = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        process_noise = np.diag([0.05, 0.05, 0.8, 0.8])
        measurement_noise = np.eye(2) * 0.35
        predicted = transition @ self.state
        predicted_covariance = transition @ self.covariance @ transition.T + process_noise
        innovation = np.asarray([x, y]) - observation @ predicted
        maximum_innovation = max(4.0, 55.0 * elapsed)
        innovation_norm = float(np.linalg.norm(innovation))
        if innovation_norm > maximum_innovation:
            innovation *= maximum_innovation / innovation_norm
        innovation_covariance = observation @ predicted_covariance @ observation.T + measurement_noise
        gain = predicted_covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state = predicted + gain @ innovation
        self.covariance = (np.eye(4) - gain @ observation) @ predicted_covariance

        sx, sy, vx, vy = (float(value) for value in self.state)
        self.positions.append((frame_index, sx, sy))
        speed_mps = hypot(vx, vy)
        if len(self.positions) >= 4:
            window_start = self.positions[max(0, len(self.positions) - 5)]
            displacement_x = sx - window_start[1]
            displacement_y = sy - window_start[2]
            if hypot(displacement_x, displacement_y) > 0.35:
                raw_heading = degrees(atan2(displacement_y, displacement_x))
                self.heading_samples.append((frame_index, raw_heading))

        def circular_mean(samples: list[tuple[int, float]]) -> float:
            radians = np.deg2rad([sample[1] for sample in samples])
            return degrees(atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean())))

        heading = circular_mean(self.heading_samples[-3:]) if self.heading_samples else 0.0
        yaw_rate = 0.0
        if len(self.heading_samples) >= 6:
            previous = circular_mean(self.heading_samples[-6:-3])
            current = circular_mean(self.heading_samples[-3:])
            frame_delta = self.heading_samples[-1][0] - self.heading_samples[-4][0]
            yaw_rate = wrap_angle(current - previous) / max(dt, frame_delta * dt)
        state = "turning" if abs(yaw_rate) >= 8.0 and speed_mps >= 1.0 else ("moving" if speed_mps >= 0.5 else "stopped")
        return {
            "x": sx,
            "y": sy,
            "vx": vx,
            "vy": vy,
            "speed_kmh": speed_mps * 3.6,
            "heading_deg": heading,
            "yaw_rate_deg_s": yaw_rate,
            "motion_state": state,
        }

    def coast(self, frame_index: int, dt: float) -> dict[str, float | str]:
        """Advance a temporarily missing vehicle with the constant-velocity model."""
        elapsed = dt if self.last_frame is None else max(dt, (frame_index - self.last_frame) * dt)
        self.last_frame = frame_index
        transition = np.asarray(
            [[1, 0, elapsed, 0], [0, 1, 0, elapsed], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        process_noise = np.diag([0.08, 0.08, 0.9, 0.9])
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        x, y, vx, vy = (float(value) for value in self.state)
        speed_mps = hypot(vx, vy)
        heading = degrees(atan2(vy, vx)) if speed_mps >= 0.2 else (
            self.heading_samples[-1][1] if self.heading_samples else 0.0
        )
        self.positions.append((frame_index, x, y))
        return {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "speed_kmh": speed_mps * 3.6,
            "heading_deg": heading,
            "yaw_rate_deg_s": 0.0,
            "motion_state": "coasting",
        }


class MotionRegistry:
    def __init__(self) -> None:
        self.tracks: dict[int, KalmanTrack] = {}

    def update(self, track_id: int, frame_index: int, x: float, y: float, dt: float) -> dict[str, float | str]:
        track = self.tracks.setdefault(track_id, KalmanTrack.create(track_id, x, y))
        return track.update(frame_index, x, y, dt)

    def coast(self, track_id: int, frame_index: int, dt: float) -> dict[str, float | str]:
        return self.tracks[track_id].coast(frame_index, dt)
