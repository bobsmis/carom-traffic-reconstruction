from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class GlobalTrack:
    global_id: int
    local_ids: set[int] = field(default_factory=set)
    last_observed_frame: int = 0
    lost_frames: int = 0
    retired: bool = False
    ground: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    class_name: str = "car"
    last_detection: dict[str, Any] | None = None

    def predicted_ground(self, frame_index: int, fps: float) -> np.ndarray:
        elapsed = max(0, frame_index - self.last_observed_frame) / fps
        return self.ground + self.velocity * elapsed


class GlobalTrackManager:
    """Associate detector-local IDs into session-global, gap-tolerant identities."""

    def __init__(self, max_lost_frames: int = 12, max_match_distance_m: float = 7.0) -> None:
        self.max_lost_frames = max_lost_frames
        self.max_match_distance_m = max_match_distance_m
        self.next_global_id = 1
        self.tracks: dict[int, GlobalTrack] = {}
        self.matched_ids: set[int] = set()
        self.retired_ids: list[int] = []
        self.reacquisitions = 0
        self.max_recovered_gap = 0

    def _new_track(self, frame_index: int, observation: dict[str, Any]) -> GlobalTrack:
        track = GlobalTrack(
            global_id=self.next_global_id,
            local_ids={int(observation["detector_track_id"])},
            last_observed_frame=frame_index,
            ground=np.asarray(observation["ground_measurement"], dtype=float),
            class_name=str(observation["class_name"]),
        )
        self.tracks[track.global_id] = track
        self.next_global_id += 1
        return track

    def assign(
        self,
        frame_index: int,
        observations: list[dict[str, Any]],
        fps: float,
    ) -> list[tuple[int, dict[str, Any]]]:
        active = [track for track in self.tracks.values() if not track.retired]
        self.matched_ids = set()
        assignments: dict[int, int] = {}

        # Preserve a reliable ByteTrack association first, while checking metric plausibility.
        for observation_index, observation in enumerate(observations):
            local_id = int(observation["detector_track_id"])
            measurement = np.asarray(observation["ground_measurement"], dtype=float)
            candidates = [
                track
                for track in active
                if track.global_id not in self.matched_ids and local_id in track.local_ids
            ]
            if not candidates:
                continue
            track = min(candidates, key=lambda item: np.linalg.norm(measurement - item.predicted_ground(frame_index, fps)))
            distance = float(np.linalg.norm(measurement - track.predicted_ground(frame_index, fps)))
            gate = self.max_match_distance_m + float(np.linalg.norm(track.velocity)) / fps * 2.0
            if distance <= gate:
                assignments[observation_index] = track.global_id
                self.matched_ids.add(track.global_id)

        remaining_observations = [index for index in range(len(observations)) if index not in assignments]
        remaining_tracks = [track for track in active if track.global_id not in self.matched_ids]
        if remaining_observations and remaining_tracks:
            costs = np.full((len(remaining_tracks), len(remaining_observations)), 1e6, dtype=float)
            for row, track in enumerate(remaining_tracks):
                predicted = track.predicted_ground(frame_index, fps)
                gate = self.max_match_distance_m + float(np.linalg.norm(track.velocity)) / fps * 2.0
                for column, observation_index in enumerate(remaining_observations):
                    observation = observations[observation_index]
                    measurement = np.asarray(observation["ground_measurement"], dtype=float)
                    distance = float(np.linalg.norm(measurement - predicted))
                    class_penalty = 0.0 if observation["class_name"] == track.class_name else 0.8
                    if distance <= gate:
                        costs[row, column] = distance + class_penalty
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] >= 1e5:
                    continue
                track = remaining_tracks[row]
                observation_index = remaining_observations[column]
                assignments[observation_index] = track.global_id
                self.matched_ids.add(track.global_id)

        output: list[tuple[int, dict[str, Any]]] = []
        for observation_index, observation in enumerate(observations):
            global_id = assignments.get(observation_index)
            track = self.tracks[global_id] if global_id is not None else self._new_track(frame_index, observation)
            if global_id is not None and track.lost_frames > 0:
                self.reacquisitions += 1
                self.max_recovered_gap = max(self.max_recovered_gap, track.lost_frames)
            track.local_ids.add(int(observation["detector_track_id"]))
            track.last_observed_frame = frame_index
            track.lost_frames = 0
            track.ground = np.asarray(observation["ground_measurement"], dtype=float)
            track.class_name = str(observation["class_name"])
            self.matched_ids.add(track.global_id)
            output.append((track.global_id, observation))

        for track in active:
            if track.global_id in self.matched_ids:
                continue
            track.lost_frames = frame_index - track.last_observed_frame
            if track.lost_frames > self.max_lost_frames:
                track.retired = True
                self.retired_ids.append(track.global_id)
        return output

    def update_state(self, global_id: int, detection: dict[str, Any]) -> None:
        track = self.tracks[global_id]
        track.ground = np.asarray(
            [detection["ground"]["x"], detection["ground"]["y"]],
            dtype=float,
        )
        track.velocity = np.asarray(
            [detection["velocity_mps"]["x"], detection["velocity_mps"]["y"]],
            dtype=float,
        )
        track.class_name = str(detection["class_name"])
        track.last_detection = detection

    def set_predicted_detection(self, global_id: int, detection: dict[str, Any]) -> None:
        self.tracks[global_id].last_detection = detection

    def coast_tracks(self) -> list[GlobalTrack]:
        return [
            track
            for track in self.tracks.values()
            if not track.retired
            and track.global_id not in self.matched_ids
            and 0 < track.lost_frames <= self.max_lost_frames
            and track.last_detection is not None
        ]

    @property
    def active_ids(self) -> list[int]:
        return sorted(track.global_id for track in self.tracks.values() if not track.retired)
