from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import MetricCamera


class RegisteredRoadMap:
    """A metric world-to-map registration with a camera-visibility overlay."""

    def __init__(self, root: Path, config: dict[str, Any], camera: MetricCamera) -> None:
        self.config = config
        self.camera = camera
        self._visibility_cache: dict[tuple[int, int], np.ndarray] = {}
        self.image = cv2.imread(str(root / config["image_path"]), cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError(root / config["image_path"])
        world = np.asarray(config["world_control_points_m"], dtype=np.float32)
        image = np.asarray(config["image_control_points_px"], dtype=np.float32)
        self.world_to_image = cv2.getPerspectiveTransform(world, image)
        self.image_to_world = np.linalg.inv(self.world_to_image)
        self.drivable_polygon = np.asarray(config["drivable_polygon_px"], dtype=np.int32)
        display_config = config.get("display", {})
        self.display_mode = display_config.get("mode", "legacy")
        self.metric_world_extent = tuple(
            float(value)
            for value in display_config.get("world_extent_m", [-16.0, 16.0, 0.0, 100.0])
        )
        self.metric_padding = int(display_config.get("padding_px", 6))
        self.rotate = display_config.get("rotate", "clockwise")
        if self.rotate not in {"clockwise", "counterclockwise"}:
            raise ValueError("road map display rotation must be clockwise or counterclockwise")
        self.lane_centerlines = [
            np.asarray(points, dtype=np.float32)
            for points in config.get("lane_centerlines_px", [])
        ]
        self.explicit_lane_directions = "lane_directions" in config
        configured_directions = config.get("lane_directions", [1] * len(self.lane_centerlines))
        if len(configured_directions) != len(self.lane_centerlines):
            raise ValueError("lane_directions must match lane_centerlines")
        self.lane_directions = [1 if float(value) >= 0 else -1 for value in configured_directions]
        self.vehicle_sprites: dict[str, np.ndarray] = {}
        for class_name, relative_path in config.get("vehicle_sprites", {}).items():
            sprite = cv2.imread(str(root / relative_path), cv2.IMREAD_UNCHANGED)
            if sprite is None:
                raise FileNotFoundError(root / relative_path)
            if sprite.ndim != 3 or sprite.shape[2] != 4:
                raise ValueError(f"vehicle sprite must be BGRA: {relative_path}")
            self.vehicle_sprites[class_name] = sprite
        rotated_height, rotated_width = self.image.shape[1], self.image.shape[0]
        self.crop = tuple(
            int(value)
            for value in config.get("display", {}).get(
                "crop_xywh",
                [0, 0, rotated_width, rotated_height],
            )
        )

    def _world_to_display_matrix(self, width: int, height: int) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.metric_world_extent
        padding = self.metric_padding
        scale = min(
            (width - 2 * padding) / max(y_max - y_min, 1e-9),
            (height - 2 * padding) / max(x_max - x_min, 1e-9),
        )
        offset_x = (width - scale * (y_max - y_min)) / 2.0
        offset_y = (height - scale * (x_max - x_min)) / 2.0
        return np.asarray(
            [
                [0.0, -scale, offset_x + scale * y_max],
                [scale, 0.0, offset_y - scale * x_min],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def world_to_original(self, points: list[tuple[float, float]] | np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(values, self.world_to_image).reshape(-1, 2)

    def original_to_display(
        self,
        points: list[tuple[float, float]] | np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if self.display_mode == "metric":
            transform = self._world_to_display_matrix(width, height) @ self.image_to_world
            return cv2.perspectiveTransform(
                values.astype(np.float32).reshape(-1, 1, 2),
                transform.astype(np.float32),
            ).reshape(-1, 2)
        if self.rotate == "clockwise":
            rotated = np.column_stack((self.image.shape[0] - 1 - values[:, 1], values[:, 0]))
        else:
            rotated = np.column_stack((values[:, 1], self.image.shape[1] - 1 - values[:, 0]))
        crop_x, crop_y, crop_width, crop_height = self.crop
        rotated[:, 0] = (rotated[:, 0] - crop_x) * width / crop_width
        rotated[:, 1] = (rotated[:, 1] - crop_y) * height / crop_height
        return rotated

    def world_to_display(
        self,
        points: list[tuple[float, float]] | np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        if self.display_mode == "metric":
            values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(
                values,
                self._world_to_display_matrix(width, height).astype(np.float32),
            ).reshape(-1, 2)
        return self.original_to_display(self.world_to_original(points), width, height)

    def _camera_footprint_original(self) -> np.ndarray:
        top = int(self.config.get("visibility_image_top_px", 40))
        width = self.camera.frame_width - 1
        height = self.camera.frame_height - 1
        border: list[tuple[float, float]] = []
        for u in np.linspace(0, width, 24):
            border.append((float(u), float(top)))
        for v in np.linspace(top, height, 18)[1:]:
            border.append((float(width), float(v)))
        for u in np.linspace(width, 0, 24)[1:]:
            border.append((float(u), float(height)))
        for v in np.linspace(height, top, 18)[1:-1]:
            border.append((0.0, float(v)))
        ground = [self.camera.pixel_to_ground(u, v) for u, v in border]
        return self.world_to_original(ground).astype(np.int32)

    def _display_image(self, image: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
        if self.display_mode == "metric":
            transform = self._world_to_display_matrix(width, height) @ self.image_to_world
            return cv2.warpPerspective(
                image,
                transform,
                (width, height),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        rotation = (
            cv2.ROTATE_90_CLOCKWISE
            if self.rotate == "clockwise"
            else cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        rotated = cv2.rotate(image, rotation)
        x, y, crop_width, crop_height = self.crop
        cropped = rotated[y : y + crop_height, x : x + crop_width]
        return cv2.resize(cropped, (width, height), interpolation=interpolation)

    def visibility_mask(self, width: int, height: int) -> np.ndarray:
        cached = self._visibility_cache.get((width, height))
        if cached is not None:
            return cached
        if self.config.get("geometry_render") == "metric_lanes":
            # Test displayed ground points directly in the real camera image.
            # No clipping against the unrelated decorative texture extent.
            yy, xx = np.indices((height, width), dtype=np.float32)
            pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
            world = cv2.perspectiveTransform(pixels, np.linalg.inv(self._world_to_display_matrix(width, height)))
            # Build inverse image-to-ground homography from four actual camera
            # rays, so both explicit and intrinsic/extrinsic calibration work.
            image_points = np.asarray([[0, self.camera.frame_height - 1], [self.camera.frame_width - 1, self.camera.frame_height - 1], [0, self.camera.frame_height * .5], [self.camera.frame_width - 1, self.camera.frame_height * .5]], dtype=np.float32)
            grounds = np.asarray([self.camera.pixel_to_ground(float(u), float(v)) for u, v in image_points], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(grounds, image_points)
            uv = cv2.perspectiveTransform(world, transform).reshape(height, width, 2)
            ground_x = world.reshape(height, width, 2)[:, :, 0]
            visible = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < self.camera.frame_width) & (uv[:, :, 1] >= self.config.get("visibility_image_top_px", 255)) & (uv[:, :, 1] < self.camera.frame_height) & (np.abs(ground_x) <= 14.0)
            result = visible.astype(np.uint8) * 255
            self._visibility_cache[(width, height)] = result
            return result
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self._camera_footprint_original()], 255)
        drivable = np.zeros_like(mask)
        cv2.fillPoly(drivable, [self.drivable_polygon], 255)
        mask = cv2.bitwise_and(mask, drivable)
        return self._display_image(mask, width, height, cv2.INTER_NEAREST)

    def render(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        base = self._display_image(self.image, width, height, cv2.INTER_AREA)
        if self.config.get("geometry_render") == "metric_lanes":
            base = np.full_like(base, (24, 28, 30))
            _, _, near, far = self.metric_world_extent
            bounds = self.world_to_display([(-14, near), (14, near), (14, far), (-14, far)], width, height).astype(np.int32)
            cv2.fillPoly(base, [bounds], (69, 72, 75))
            for lane_border in np.linspace(-14, 14, 9):
                if abs(lane_border) < .1:
                    for center in (-.10, .10):
                        ends = self.world_to_display([(center, near), (center, far)], width, height).astype(int)
                        cv2.line(base, tuple(ends[0]), tuple(ends[1]), (40, 210, 235), 1, cv2.LINE_AA)
                elif abs(lane_border) > 13:
                    ends = self.world_to_display([(lane_border, near), (lane_border, far)], width, height).astype(int)
                    cv2.line(base, tuple(ends[0]), tuple(ends[1]), (225, 225, 225), 1, cv2.LINE_AA)
                else:
                    for start in np.arange(near, far, 9):
                        ends = self.world_to_display([(lane_border, start), (lane_border, min(start + 4, far))], width, height).astype(int)
                        cv2.line(base, tuple(ends[0]), tuple(ends[1]), (210, 215, 218), 1, cv2.LINE_AA)
        if self.display_mode == "metric":
            # Beyond the supplied texture, show an explicitly schematic road;
            # do not stretch a 100 m texture to pretend it represents 200 m.
            extent = self.metric_world_extent
            schematic = np.full_like(base, (24, 28, 30))
            for lane in self.lane_centerlines:
                center = cv2.perspectiveTransform(lane[:1].reshape(-1, 1, 2), self.image_to_world)[0, 0]
                points = self.world_to_display([(float(center[0]), extent[2]), (float(center[0]), extent[3])], width, height).astype(int)
                scale = abs(self._world_to_display_matrix(width, height)[1, 0])
                cv2.line(schematic, tuple(points[0]), tuple(points[1]), (66, 70, 71), max(1, round(2.7 * scale)))
            valid = np.any(base != 0, axis=2)
            base = np.where(valid[:, :, None], base, schematic)
        mask = self.visibility_mask(width, height)
        dark = np.clip(base.astype(np.float32) * 0.30, 0, 255).astype(np.uint8)
        highlight = cv2.addWeighted(base, 0.86, np.full_like(base, (52, 74, 35)), 0.14, 10)
        output = np.where(mask[:, :, None] > 0, highlight, dark)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (214, 222, 49), 2, cv2.LINE_AA)
        return output, mask

    def camera_marker_display(self, width: int, height: int) -> tuple[int, int]:
        """Return the visible-map edge nearest the physical camera."""
        mask = self.visibility_mask(width, height)
        rows, columns = np.nonzero(mask)
        if not len(columns):
            return width - 1, height // 2
        column = int(columns.max()) if self.rotate == "counterclockwise" else int(columns.min())
        edge_rows = rows[columns == column]
        return column, int(np.mean(edge_rows))

    @staticmethod
    def _project_to_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, float, int]:
        projected, distance, segment, _ = RegisteredRoadMap._project_to_polyline_progress(
            point, polyline
        )
        return projected, distance, segment

    @staticmethod
    def _project_to_polyline_progress(
        point: np.ndarray,
        polyline: np.ndarray,
    ) -> tuple[np.ndarray, float, int, float]:
        best = polyline[0].astype(float)
        best_distance = float("inf")
        best_segment = 0
        best_progress = 0.0
        cumulative = 0.0
        for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:], strict=True)):
            delta = end.astype(float) - start.astype(float)
            denominator = float(delta @ delta)
            # Extend endpoint segments instead of collapsing far/near vehicles
            # onto one finite-map endpoint. Keep the extrapolation auditable.
            lower = -np.inf if index == 0 else 0.0
            upper = np.inf if index == len(polyline) - 2 else 1.0
            amount = 0.0 if denominator < 1e-9 else float(np.clip((point - start) @ delta / denominator, lower, upper))
            candidate = start + amount * delta
            distance = float(np.linalg.norm(point - candidate))
            if distance < best_distance:
                best = candidate
                best_distance = distance
                best_segment = index
                best_progress = cumulative + amount * float(np.linalg.norm(delta))
            cumulative += float(np.linalg.norm(delta))
        return best, best_distance, best_segment, best_progress

    @staticmethod
    def _point_at_progress(polyline: np.ndarray, progress: float) -> tuple[np.ndarray, int]:
        remaining = float(progress)
        for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:], strict=True)):
            delta = end.astype(float) - start.astype(float)
            length = float(np.linalg.norm(delta))
            if remaining <= length or index == len(polyline) - 2:
                amount = 0.0 if length < 1e-9 else float(remaining / length)
                return start.astype(float) + amount * delta, index
            remaining -= length
        return polyline[-1].astype(float), max(0, len(polyline) - 2)

    def assign_map_lanes(self, records: list[dict[str, Any]]) -> None:
        """Associate global tracks with lane geometry and filter presentation.

        Preserve raw measurements. Metric lanes support hysteretic lane changes
        and retain measured lateral separation instead of forcing center snapping.
        """
        if not self.lane_centerlines:
            return
        observations: dict[int, list[np.ndarray]] = {}
        ground_observations: dict[int, list[np.ndarray]] = {}
        for record in records:
            for detection in record["detections"]:
                for field in list(detection):
                    if field.startswith("map_"):
                        detection.pop(field)
                if detection.get("prediction_only"):
                    continue
                ground = detection["ground"]
                ground_point = np.asarray([ground["x"], ground["y"]], dtype=float)
                mapped = self.world_to_original([(ground["x"], ground["y"])])[0]
                observations.setdefault(int(detection["track_id"]), []).append(mapped)
                ground_observations.setdefault(int(detection["track_id"]), []).append(ground_point)

        assignment_method = self.config.get("lane_assignment", "trajectory_cluster")
        if assignment_method == "nearest_centerline":
            assignments: dict[int, int] = {}
            residuals: dict[int, float] = {}
            centers = np.arange(len(self.lane_centerlines), dtype=float)
            for track_id, points in observations.items():
                distances = []
                for lane in self.lane_centerlines:
                    lane_distances = [
                        self._project_to_polyline(point, lane)[1]
                        for point in points
                    ]
                    distances.append(float(np.median(lane_distances)))
                lane_index = int(np.argmin(distances))
                assignments[track_id] = lane_index
                residuals[track_id] = distances[lane_index]
        else:
            # Remove shared road curvature before clustering. The longitudinal
            # and lateral axes are configurable because not every road frame is
            # encoded as world-x-forward/world-y-lateral.
            longitudinal_axis = 0 if self.config.get("longitudinal_axis", "x") == "x" else 1
            lateral_axis = 1 - longitudinal_axis
            reliable = [points for points in ground_observations.values() if len(points) >= 8]
            curve_points = np.concatenate(reliable if reliable else list(ground_observations.values()))
            curve_degree = 2 if len(curve_points) >= 20 else 1
            road_curve = np.polyfit(
                curve_points[:, longitudinal_axis],
                curve_points[:, lateral_axis],
                curve_degree,
            )
            residuals = {
                track_id: float(
                    np.median(
                        values[:, lateral_axis]
                        - np.polyval(road_curve, values[:, longitudinal_axis])
                    )
                )
                for track_id, points in ground_observations.items()
                for values in [np.asarray(points, dtype=float)]
            }
            reliable_ids = [track_id for track_id, points in ground_observations.items() if len(points) >= 8]
            lane_count = min(
                int(self.config.get("visible_lane_count", len(self.lane_centerlines))),
                len(self.lane_centerlines),
                max(1, len(reliable_ids)),
            )
            values = np.asarray([residuals[track_id] for track_id in reliable_ids], dtype=float)
            weights = np.asarray([len(ground_observations[track_id]) for track_id in reliable_ids], dtype=float)
            if len(values) == 0:
                centers = np.asarray([0.0])
            elif lane_count == 1:
                centers = np.asarray([float(np.average(values, weights=weights))])
            else:
                centers = np.linspace(float(values.min()), float(values.max()), lane_count)
                for _ in range(64):
                    labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
                    updated = centers.copy()
                    for index in range(lane_count):
                        selected = labels == index
                        if np.any(selected):
                            updated[index] = float(np.average(values[selected], weights=weights[selected]))
                    updated.sort()
                    if np.allclose(updated, centers, atol=1e-5):
                        break
                    centers = updated
            assignments = {
                track_id: int(np.argmin(np.abs(centers - residual)))
                for track_id, residual in residuals.items()
            }

        lane_states: dict[int, tuple[int, int, int]] = {}
        for record in records:
            for detection in record["detections"]:
                track_id = int(detection["track_id"])
                if track_id not in assignments:
                    continue
                ground = detection["ground"]
                mapped = self.world_to_original([(ground["x"], ground["y"])])[0]
                lane_index = assignments[track_id]
                if assignment_method == "nearest_centerline":
                    distances = [self._project_to_polyline(mapped, lane)[1] for lane in self.lane_centerlines]
                    candidate = int(np.argmin(distances))
                    active, pending, count = lane_states.get(track_id, (candidate, candidate, 0))
                    if not detection.get("prediction_only"):
                        if candidate != active and distances[candidate] + 12.0 < distances[active]:
                            count = count + 1 if pending == candidate else 1
                            pending = candidate
                            if count >= 8:
                                active, count = candidate, 0
                        else:
                            pending, count = active, 0
                    lane_states[track_id] = (active, pending, count)
                    lane_index = active
                    detection["map_lane_pending"] = f"L{pending + 1:02d}"
                    detection["map_lane_assignment_method"] = "geometric distance with 8-observation hysteresis"
                snapped, error, segment_index, progress = self._project_to_polyline_progress(
                    mapped,
                    self.lane_centerlines[lane_index],
                )
                detection["map_lane_id"] = f"L{lane_index + 1:02d}"
                if assignment_method == "nearest_centerline":
                    detection["map_lane_residual_px"] = round(error, 3)
                else:
                    detection["map_lane_residual_m"] = round(residuals[track_id], 3)
                    detection["map_lane_cluster_center_m"] = round(float(centers[lane_index]), 3)
                detection["map_position_raw_px"] = {
                    "x": round(float(snapped[0]), 3),
                    "y": round(float(snapped[1]), 3),
                }
                detection["map_measurement_px"] = {"x": float(mapped[0]), "y": float(mapped[1])}
                detection["map_coordinate_status"] = "estimated; nominal lane-width calibration"
                detection["map_position_px"] = dict(detection["map_position_raw_px"])
                detection["map_lane_progress_raw_px"] = round(progress, 3)
                detection["map_lateral_error_px"] = round(error, 3)
                detection["map_lane_segment"] = segment_index
                detection["map_lane_direction"] = self.lane_directions[lane_index]
                detection["sources"]["map_position_px"] = (
                    "persistent global track projected to configured registered map lane"
                )

        # One-dimensional lane progress is smoother and direction-constrained.
        # This prevents a noisy anchor from making a vehicle reverse on the map.
        display_states: dict[int, tuple[float, float, bool]] = {}
        lateral_states: dict[int, float] = {}
        for record in records:
            lane_members: dict[int, list[dict[str, Any]]] = {}
            for detection in record["detections"]:
                if "map_position_px" not in detection:
                    continue
                track_id = int(detection["track_id"])
                lane_index = int(detection["map_lane_id"][1:]) - 1
                lane = self.lane_centerlines[lane_index]
                direction = self.lane_directions[lane_index]
                raw_progress = float(detection["map_lane_progress_raw_px"])
                previous = display_states.get(track_id)
                if previous is None:
                    progress = raw_progress
                    velocity = 0.0
                    reacquired = False
                else:
                    old_progress, old_velocity, was_prediction = previous
                    predicted = old_progress + old_velocity
                    residual = float(np.clip(raw_progress - predicted, -18.0, 18.0))
                    is_prediction = bool(detection.get("prediction_only"))
                    weight = 0.12 if is_prediction else (0.22 if was_prediction else 0.48)
                    progress = predicted + weight * residual
                    if direction * (progress - old_progress) < 0.0:
                        progress = old_progress
                    measured_velocity = progress - old_progress
                    velocity = 0.68 * old_velocity + 0.32 * measured_velocity
                    if direction * velocity < 0.0:
                        velocity = 0.0
                    reacquired = was_prediction and not is_prediction
                smooth, segment_index = self._point_at_progress(lane, progress)
                if self.config.get("geometry_render") == "metric_lanes":
                    # Preserve measured lateral separation. A lane label is an
                    # association, not permission to snap several cars together.
                    raw_lateral = detection["map_measurement_px"]["x"]
                    lateral = lateral_states.get(track_id, raw_lateral)
                    lateral = 0.55 * lateral + 0.45 * raw_lateral
                    lateral_states[track_id] = lateral
                    smooth[0] = lateral
                    # The displayed lane must describe the displayed position.
                    # Keep the hysteretic association separately for tracking.
                    geometric_lane = int(np.argmin([self._project_to_polyline(smooth, item)[1] for item in self.lane_centerlines]))
                    detection["map_tracking_lane_id"] = detection["map_lane_id"]
                    detection["map_lane_id"] = f"L{geometric_lane + 1:02d}"
                    detection["map_lane_status"] = "stable" if geometric_lane == lane_index else "association_pending"
                    world_point = cv2.perspectiveTransform(np.asarray([[smooth]], dtype=np.float32), self.image_to_world)[0, 0]
                    if abs(world_point[0]) > 14.0:
                        detection["map_lane_status"] = "outside_road_model"
                    detection["map_lane_direction"] = self.lane_directions[geometric_lane]
                detection["map_position_px"] = {
                    "x": round(float(smooth[0]), 3),
                    "y": round(float(smooth[1]), 3),
                }
                detection["map_lane_progress_px"] = round(progress, 3)
                detection["map_lane_segment"] = segment_index
                detection["map_visible"] = True
                detection["map_smoothing"] = {
                    "model": "direction-constrained lane-progress filter",
                    "reacquisition_blend": reacquired,
                    "raw_progress_px": raw_progress,
                    "correction_px": round(progress - raw_progress, 3),
                    "direction_prior": "configured lane direction; not independent motion truth",
                }
                display_states[track_id] = (progress, velocity, bool(detection.get("prediction_only")))
                lane_members.setdefault(lane_index, []).append(detection)

            # Audit all footprint pairs, including different tentative lane IDs.
            # Retain every object and coordinate; never solve overlap by hiding it.
            objects = [item for members in lane_members.values() for item in members]
            polygons = []
            for item in objects:
                render_item = dict(item)
                render_item.setdefault("dimensions_m", {"length": 4.55, "width": 1.82})
                polygons.append(self.vehicle_polygon_display(render_item, 1000, 500).astype(np.float32))
            for i, first in enumerate(objects):
                for j in range(i + 1, len(objects)):
                    if len(polygons[i]) != 4 or len(polygons[j]) != 4:
                        continue
                    area, _ = cv2.intersectConvexConvex(polygons[i], polygons[j])
                    if area > 0.01:
                        second = objects[j]
                        first.setdefault("map_overlap_with", []).append(second["track_id"])
                        second.setdefault("map_overlap_with", []).append(first["track_id"])

    def lane_centerlines_display(self, width: int, height: int) -> list[np.ndarray]:
        return [self.original_to_display(line, width, height) for line in self.lane_centerlines]

    def detection_position_display(
        self,
        detection: dict[str, Any],
        width: int,
        height: int,
    ) -> np.ndarray:
        if "map_position_px" in detection:
            position = detection["map_position_px"]
            return self.original_to_display([(position["x"], position["y"])], width, height)[0]
        ground = detection.get("lane_ground", detection["ground"])
        return self.world_to_display([(ground["x"], ground["y"])], width, height)[0]

    def vehicle_polygon_display(
        self,
        detection: dict[str, Any],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Create an image-map footprint aligned with its registered lane."""
        position = detection.get("map_position_px")
        if position is None:
            return np.empty((0, 2), dtype=float)
        lane_index = int(detection["map_lane_id"][1:]) - 1
        segment = min(
            int(detection.get("map_lane_segment", 0)),
            len(self.lane_centerlines[lane_index]) - 2,
        )
        start = self.lane_centerlines[lane_index][segment].astype(float)
        end = self.lane_centerlines[lane_index][segment + 1].astype(float)
        direction_sign = self.lane_directions[lane_index]
        if self.display_mode == "metric":
            center_value = np.asarray([position["x"], position["y"]], dtype=np.float32)
            world_values = cv2.perspectiveTransform(
                np.asarray([[start, end, center_value]], dtype=np.float32),
                self.image_to_world.astype(np.float32),
            )[0]
            start_world, end_world, center_world = world_values
            forward_world = (end_world - start_world).astype(float) * direction_sign
            forward_world /= max(float(np.linalg.norm(forward_world)), 1e-9)
            left_world = np.asarray([-forward_world[1], forward_world[0]])
            dimensions = detection["dimensions_m"]
            half_length = float(dimensions["length"]) / 2.0
            half_width = float(dimensions["width"]) / 2.0
            corners_world = np.asarray(
                [
                    center_world + half_length * forward_world + half_width * left_world,
                    center_world + half_length * forward_world - half_width * left_world,
                    center_world - half_length * forward_world - half_width * left_world,
                    center_world - half_length * forward_world + half_width * left_world,
                ]
            )
            return self.world_to_display(corners_world, width, height)
        forward = end - start
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        if self.explicit_lane_directions:
            forward *= direction_sign
        else:
            velocity = detection.get("velocity_mps", {"x": 1.0, "y": 0.0})
            mapped_velocity = self.world_to_original(
                [
                    (detection["ground"]["x"], detection["ground"]["y"]),
                    (
                        detection["ground"]["x"] + velocity["x"],
                        detection["ground"]["y"] + velocity["y"],
                    ),
                ]
            )
            if float((mapped_velocity[1] - mapped_velocity[0]) @ forward) < 0:
                forward = -forward
        left = np.asarray([-forward[1], forward[0]])
        world = np.asarray(self.config["world_control_points_m"], dtype=float)
        image = np.asarray(self.config["image_control_points_px"], dtype=float)
        longitudinal_scale = float(np.linalg.norm(image[2] - image[0]) / np.linalg.norm(world[2] - world[0]))
        lateral_scale = float(np.linalg.norm(image[1] - image[0]) / np.linalg.norm(world[1] - world[0]))
        dimensions = detection["dimensions_m"]
        center = np.asarray([position["x"], position["y"]], dtype=float)
        half_length = float(dimensions["length"]) * longitudinal_scale / 2
        half_width = float(dimensions["width"]) * lateral_scale / 2
        corners = np.asarray(
            [
                center + half_length * forward + half_width * left,
                center + half_length * forward - half_width * left,
                center - half_length * forward - half_width * left,
                center - half_length * forward + half_width * left,
            ]
        )
        return self.original_to_display(corners, width, height)

    def vehicle_icon_display(
        self,
        detection: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, np.ndarray]:
        """Return a compact top-view car silhouette and glass geometry."""
        corners = self.vehicle_polygon_display(detection, width, height)
        if len(corners) != 4:
            empty = np.empty((0, 2), dtype=float)
            return {"body": empty, "glass": empty, "heading": empty}
        front_left, front_right, rear_right, rear_left = corners

        def cross_section(amount: float, width_scale: float) -> tuple[np.ndarray, np.ndarray]:
            left = front_left * (1.0 - amount) + rear_left * amount
            right = front_right * (1.0 - amount) + rear_right * amount
            center = (left + right) / 2.0
            return (
                center + (left - center) * width_scale,
                center + (right - center) * width_scale,
            )

        front_l, front_r = cross_section(0.0, 0.64)
        shoulder_l, shoulder_r = cross_section(0.18, 1.0)
        rear_shoulder_l, rear_shoulder_r = cross_section(0.82, 1.0)
        rear_l, rear_r = cross_section(1.0, 0.72)
        glass_front_l, glass_front_r = cross_section(0.30, 0.72)
        glass_rear_l, glass_rear_r = cross_section(0.68, 0.72)
        nose = (front_left + front_right) / 2.0
        hood = (shoulder_l + shoulder_r) / 2.0
        return {
            "body": np.asarray(
                [front_l, front_r, shoulder_r, rear_shoulder_r, rear_r, rear_l, rear_shoulder_l, shoulder_l]
            ),
            "glass": np.asarray([glass_front_l, glass_front_r, glass_rear_r, glass_rear_l]),
            "heading": np.asarray([hood, nose]),
        }

    def has_vehicle_sprite(self, class_name: str) -> bool:
        return class_name in self.vehicle_sprites or (
            class_name == "car" and "suv" in self.vehicle_sprites
        )

    def paste_vehicle_sprite(
        self,
        image: np.ndarray,
        detection: dict[str, Any],
        width: int,
        height: int,
        opacity: float = 1.0,
    ) -> bool:
        """Perspective-warp a transparent top-view vehicle sprite onto the map."""
        class_name = str(detection.get("class_name", "car"))
        sprite = self.vehicle_sprites.get(class_name)
        if sprite is None and class_name == "car":
            sprite = self.vehicle_sprites.get("suv")
        if sprite is None:
            return False
        corners = self.vehicle_polygon_display(detection, width, height)
        if len(corners) != 4:
            return False
        source_height, source_width = sprite.shape[:2]
        source = np.asarray(
            [[0.0, 0.0], [source_width - 1.0, 0.0], [source_width - 1.0, source_height - 1.0], [0.0, source_height - 1.0]],
            dtype=np.float32,
        )
        destination = np.asarray(corners, dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(
            sprite,
            transform,
            (width, height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        alpha = (warped[:, :, 3:4].astype(np.float32) / 255.0) * float(opacity)
        if not np.any(alpha > 0):
            return False
        image[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + image.astype(np.float32) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
        return True
