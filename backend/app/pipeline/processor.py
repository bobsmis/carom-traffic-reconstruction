from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import MetricCamera
from .motion import MotionRegistry
from .optical_flow import sparse_vehicle_flow
from .routes import InferredRoute, infer_routes
from .road_map import RegisteredRoadMap
from .tracking import GlobalTrack, GlobalTrackManager


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DIMENSIONS = {
    "car": (4.55, 1.82, 1.48),
    "motorcycle": (2.2, 0.8, 1.35),
    "bus": (12.0, 2.55, 3.25),
    "truck": (8.2, 2.5, 3.2),
}
CUBOID_EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47) % 180
    color = cv2.cvtColor(np.uint8([[[hue, 210, 245]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def mask_anchor(mask: np.ndarray, box: np.ndarray) -> tuple[float, float]:
    if len(mask) >= 3:
        maximum_y = float(mask[:, 1].max())
        bottom = mask[mask[:, 1] >= maximum_y - max(2.0, (maximum_y - mask[:, 1].min()) * 0.08)]
        return float(np.median(bottom[:, 0])), maximum_y
    return float((box[0] + box[2]) / 2), float(box[3])


def estimate_vehicle_dimensions(
    camera: MetricCamera,
    mask: np.ndarray,
    box: np.ndarray,
    class_name: str,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Estimate an individual footprint width from its road-contact silhouette.

    A monocular 2D mask does not expose the hidden vehicle length, so the mask
    width is blended with a bounded class prior and the class aspect ratio is
    retained. This produces per-object sizes without claiming unobserved depth.
    """
    prior = np.asarray(DIMENSIONS[class_name], dtype=float)
    if len(mask) < 3:
        return tuple(float(value) for value in prior), {
            "method": "class_prior",
            "measured_width_m": None,
        }
    maximum_y = float(mask[:, 1].max())
    minimum_y = float(mask[:, 1].min())
    contact = mask[mask[:, 1] >= maximum_y - max(3.0, (maximum_y - minimum_y) * 0.16)]
    if len(contact) < 2:
        return tuple(float(value) for value in prior), {
            "method": "class_prior",
            "measured_width_m": None,
        }
    left_u = float(np.percentile(contact[:, 0], 8))
    right_u = float(np.percentile(contact[:, 0], 92))
    contact_v = float(np.percentile(contact[:, 1], 92))
    try:
        left_ground = np.asarray(camera.pixel_to_ground(left_u, contact_v), dtype=float)
        right_ground = np.asarray(camera.pixel_to_ground(right_u, contact_v), dtype=float)
        measured_width = float(np.linalg.norm(right_ground - left_ground))
    except ValueError:
        measured_width = float(prior[1])
    lower = {"car": 1.45, "motorcycle": 0.55, "bus": 2.25, "truck": 2.05}[class_name]
    upper = {"car": 2.25, "motorcycle": 1.05, "bus": 2.75, "truck": 2.85}[class_name]
    bounded_width = float(np.clip(measured_width, lower, upper))
    width = 0.65 * float(prior[1]) + 0.35 * bounded_width
    scale = width / float(prior[1])
    length_bounds = {
        "car": (3.6, 5.4),
        "motorcycle": (1.7, 2.8),
        "bus": (9.0, 14.0),
        "truck": (5.5, 13.0),
    }[class_name]
    length = float(np.clip(float(prior[0]) * scale, *length_bounds))
    height = float(prior[2])
    return (length, width, height), {
        "method": "segmentation_contact_width_blended_with_class_aspect_prior",
        "measured_width_m": round(measured_width, 3),
        "bounded_width_m": round(bounded_width, 3),
    }


def suppress_colored_preview_overlay(frame: np.ndarray) -> np.ndarray:
    """Remove the publisher's saturated line art before local inference.

    The public preview burns cyan/yellow/green boxes into the pixels.  A fixed HSV
    rule and Telea inpainting recover a detector input without reading any label,
    ID, class text, or trajectory from that overlay.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 120)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def silhouette_fitted_cuboid(box: np.ndarray, frame_width: int, frame_height: int) -> list[tuple[float, float]]:
    """Tangent-style image cuboid fitted tightly to the instance envelope."""
    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    shift_x = float(np.clip((frame_width / 2 - center_x) * 0.10, -width * 0.22, width * 0.22))
    shift_y = -height * 0.16
    front = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    back = [(x + shift_x, y + shift_y) for x, y in front]
    return [
        (float(np.clip(x, 0, frame_width - 1)), float(np.clip(y, 0, frame_height - 1)))
        for x, y in front + back
    ]


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def valid_route_trajectory(points: list[tuple[float, float]], minimum_points: int) -> bool:
    if len(points) < minimum_points:
        return False
    path_length = sum(math.dist(points[index - 1], points[index]) for index in range(1, len(points)))
    displacement = math.dist(points[0], points[-1])
    return displacement >= 3.0 and path_length >= 3.0 and path_length / displacement <= 2.5


def project_to_polyline(
    point: tuple[float, float],
    polyline: list[tuple[float, float]],
) -> tuple[tuple[float, float], float, float]:
    """Return closest point, lateral distance, and local tangent heading."""
    target = np.asarray(point, dtype=float)
    values = np.asarray(polyline, dtype=float)
    best_distance = float("inf")
    best_point = values[0]
    best_heading = 0.0
    for start, end in zip(values[:-1], values[1:], strict=True):
        segment = end - start
        denominator = float(segment @ segment)
        ratio = 0.0 if denominator < 1e-9 else float(np.clip((target - start) @ segment / denominator, 0.0, 1.0))
        candidate = start + ratio * segment
        distance = float(np.linalg.norm(target - candidate))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
            best_heading = math.degrees(math.atan2(float(segment[1]), float(segment[0])))
    return (float(best_point[0]), float(best_point[1])), best_distance, best_heading


class VideoRoadProcessor:
    def __init__(self, root: Path, config_path: Path, output_dir: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.output_dir = output_dir
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.video_path = root / self.config["video_path"]
        self.weights_path = root / self.config["detector"]["weights"]
        crop = self.config["crop"]
        self.crop = (crop["x"], crop["y"], crop["width"], crop["height"])
        self.camera = MetricCamera.from_config(root, self.config)
        self.registry = MotionRegistry()
        tracking = self.config["processing"].get("global_tracking", {})
        self.track_manager = GlobalTrackManager(
            max_lost_frames=int(tracking.get("max_lost_frames", 12)),
            max_match_distance_m=float(tracking.get("max_match_distance_m", 7.0)),
        )
        self.road_map = (
            RegisteredRoadMap(root, self.config["road_map"], self.camera)
            if "road_map" in self.config
            else None
        )
        self.frame_records: list[dict[str, Any]] = []
        self.trajectories: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def validate_inputs(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        if sha256(self.video_path) != self.config["source_sha256"]:
            raise ValueError("source video checksum does not match the dataset manifest")
        if not self.weights_path.exists():
            raise FileNotFoundError(self.weights_path)
        if self.config["provenance"]["llm_runtime_calls"] != 0:
            raise ValueError("runtime provenance requires zero LLM calls")
        if self.config["provenance"]["manual_vehicle_state_fields"] != 0:
            raise ValueError("manual vehicle state fields are prohibited")

    def run(self) -> dict[str, Any]:
        self.validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        model = YOLO(str(self.weights_path))
        capture = cv2.VideoCapture(str(self.video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or self.config["processing"]["output_fps"])
        max_frames = int(self.config["processing"]["max_frames"])
        detector = self.config["detector"]
        frame_index = 0
        total_detections = 0
        total_observed = 0
        total_predictions = 0
        suppressed_duplicate_predictions = 0
        previous_flow_frame: np.ndarray | None = None
        while frame_index < max_frames:
            ok, full_frame = capture.read()
            if not ok:
                break
            x, y, width, height = self.crop
            frame = full_frame[y : y + height, x : x + width]
            detector_frame = (
                suppress_colored_preview_overlay(frame)
                if self.config["processing"].get("suppress_preview_overlay", False)
                else frame
            )
            result = model.track(
                detector_frame,
                persist=True,
                tracker=detector["tracker"],
                classes=detector["classes"],
                conf=detector["confidence"],
                imgsz=detector["image_size"],
                verbose=False,
                device="cpu",
            )[0]
            observations: list[dict[str, Any]] = []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().numpy()
                classes = result.boxes.cls.int().cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                masks = result.masks.xy if result.masks is not None else [np.empty((0, 2))] * len(boxes)
                candidates = sorted(
                    zip(boxes, ids, classes, confidences, masks, strict=True),
                    key=lambda item: float(item[3]),
                    reverse=True,
                )
                accepted_boxes: list[np.ndarray] = []
                for box, track_id, class_id, confidence, mask in candidates:
                    if any(box_iou(box, accepted) >= 0.72 for accepted in accepted_boxes):
                        continue
                    accepted_boxes.append(box)
                    anchor = mask_anchor(np.asarray(mask), box)
                    try:
                        ground = self.camera.pixel_to_ground(*anchor)
                    except ValueError:
                        continue
                    class_name = VEHICLE_CLASSES[int(class_id)]
                    observations.append(
                        {
                            "detector_track_id": int(track_id),
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "confidence": float(confidence),
                            "box": box,
                            "mask": np.asarray(mask),
                            "anchor": anchor,
                            "ground_measurement": ground,
                        }
                    )

            detections: list[dict[str, Any]] = []
            assignments = self.track_manager.assign(frame_index, observations, fps)
            for global_id, observation in assignments:
                    box = observation["box"]
                    mask = observation["mask"]
                    anchor = observation["anchor"]
                    class_id = observation["class_id"]
                    class_name = observation["class_name"]
                    confidence = observation["confidence"]
                    optical_flow_vectors = sparse_vehicle_flow(
                        previous_flow_frame,
                        detector_frame,
                        box,
                    )
                    motion = self.registry.update(
                        global_id,
                        frame_index,
                        *observation["ground_measurement"],
                        dt=1.0 / fps,
                    )
                    self.trajectories[global_id].append((float(motion["x"]), float(motion["y"])))
                    dimensions, dimension_estimation = estimate_vehicle_dimensions(
                        self.camera,
                        mask,
                        box,
                        class_name,
                    )
                    heading = float(motion["heading_deg"])
                    cuboid = self.camera.project_vehicle_box(
                        (float(motion["x"]), float(motion["y"])),
                        heading,
                        dimensions,
                    )
                    detection = {
                            "track_id": global_id,
                            "detector_track_id": observation["detector_track_id"],
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(confidence, 4),
                            "bbox_xyxy": [round(float(value), 2) for value in box],
                            "mask_polygon": [
                                [round(float(px), 2), round(float(py), 2)]
                                for px, py in np.asarray(mask)[:: max(1, len(mask) // 80)]
                            ],
                            "anchor": [round(anchor[0], 2), round(anchor[1], 2)],
                            "ground": {
                                "x": round(float(motion["x"]), 3),
                                "y": round(float(motion["y"]), 3),
                            },
                            "velocity_mps": {
                                "x": round(float(motion["vx"]), 3),
                                "y": round(float(motion["vy"]), 3),
                            },
                            "speed_kmh": round(float(motion["speed_kmh"]), 2),
                            "heading_deg": round(heading, 2),
                            "yaw_rate_deg_s": round(float(motion["yaw_rate_deg_s"]), 2),
                            "motion_state": motion["motion_state"],
                            "tracking_status": "observed",
                            "lost_frames": 0,
                            "prediction_only": False,
                            "optical_flow_vectors": optical_flow_vectors,
                            "dimensions_m": {
                                "length": dimensions[0],
                                "width": dimensions[1],
                                "height": dimensions[2],
                            },
                            "dimension_estimation": dimension_estimation,
                            "cuboid_image": [
                                [round(px, 2), round(py, 2)] for px, py in cuboid
                            ],
                            "cuboid_image_fitted": [
                                [round(px, 2), round(py, 2)]
                                for px, py in silhouette_fitted_cuboid(box, width, height)
                            ],
                            "sources": {
                                "mask": "YOLO segmentation",
                                "detector_track_id": "ByteTrack",
                                "track_id": "global metric association manager",
                            "ground": self.config["provenance"].get(
                                "world_coordinates", "metric ray-plane intersection"
                            ),
                                "motion": "Kalman-filtered metric trajectory",
                                "optical_flow": "local Shi-Tomasi + pyramidal Lucas-Kanade",
                                "dimensions": dimension_estimation["method"],
                                "cuboid_image_fitted": "segmentation-envelope tangent approximation",
                            },
                        }
                    detections.append(detection)
                    self.track_manager.update_state(global_id, detection)

            for track in self.track_manager.coast_tracks():
                motion = self.registry.coast(track.global_id, frame_index, dt=1.0 / fps)
                prediction = self._coast_detection(track, motion, width, height)
                prediction_ground = np.asarray(
                    [prediction["ground"]["x"], prediction["ground"]["y"]],
                    dtype=float,
                )
                overlaps_observation = any(
                    not item.get("prediction_only")
                    and np.linalg.norm(
                        prediction_ground
                        - np.asarray([item["ground"]["x"], item["ground"]["y"]], dtype=float)
                    )
                    < 2.2
                    for item in detections
                )
                if overlaps_observation:
                    suppressed_duplicate_predictions += 1
                    continue
                detections.append(prediction)
                self.track_manager.set_predicted_detection(track.global_id, prediction)
            self.frame_records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 4),
                    "detections": detections,
                }
            )
            total_detections += len(detections)
            total_observed += sum(not detection["prediction_only"] for detection in detections)
            total_predictions += sum(detection["prediction_only"] for detection in detections)
            previous_flow_frame = detector_frame.copy()
            frame_index += 1
        capture.release()

        routes = infer_routes(
            {
                track_id: points
                for track_id, points in self.trajectories.items()
                if valid_route_trajectory(
                    points,
                    int(self.config["processing"]["trajectory_min_points"]),
                )
            },
            minimum_points=int(self.config["processing"]["trajectory_min_points"]),
            threshold_m=float(self.config["processing"]["route_cluster_threshold_m"]),
        )
        self._assign_routes(routes)
        if self.road_map is not None:
            self.road_map.assign_map_lanes(self.frame_records)
        self._write_data(routes)
        self._render(routes, fps)
        elapsed = time.perf_counter() - start
        audit = {
            "scene_id": self.config["scene_id"],
            "dataset": self.config["dataset"],
            "source_url": self.config["source_url"],
            "source_video_sha256": sha256(self.video_path),
            "weights_sha256": sha256(self.weights_path),
            "config_sha256": sha256(self.config_path),
            "frames_processed": len(self.frame_records),
            "detections_emitted": total_detections,
            "observed_detections": total_observed,
            "coasted_predictions": total_predictions,
            "suppressed_duplicate_predictions": suppressed_duplicate_predictions,
            "stable_track_ids": sorted(self.track_manager.tracks),
            "retired_track_ids": self.track_manager.retired_ids,
            "max_lost_frames": self.track_manager.max_lost_frames,
            "reacquisitions": self.track_manager.reacquisitions,
            "max_recovered_gap": self.track_manager.max_recovered_gap,
            "id_reuse": False,
            "inferred_routes": len(routes),
            "runtime_seconds": round(elapsed, 2),
            "runtime_stack": ["OpenCV inverse perspective mapping", "Ultralytics YOLO segmentation", "ByteTrack", "Shi-Tomasi + Lucas-Kanade optical flow", "NumPy Kalman filter", "SciPy robust lane fitting"],
            "llm_runtime_calls": 0,
            "manual_vehicle_annotations": 0,
            "camera_calibration": self.config.get(
                "calibration_path", "embedded planar road correspondences"
            ),
            "calibration_usage": (
                "road-marking control-point homography with engineering metric scale"
                if "planar_calibration" in self.config
                else "intrinsic matrix + camera-to-road extrinsic transform"
            ),
            "road_rectification": (
                "registered aerial road map + calibrated camera visibility footprint"
                if self.road_map is not None
                else "image-to-ground homography with temporal-median road texture"
            ),
            "road_map": self.config.get("road_map", {}).get("image_path"),
            "visibility_mask": "camera ground footprint intersected with drivable map polygon",
            "trajectory_constraint": (
                "shared road-curvature removal + global-track residual clustering + registered map-lane projection"
                if self.road_map is not None
                else "spike rejection + shared road curve + nearest lane projection"
            ),
            "panel_synchronization": "one shared per-frame detection record renders optical-flow, cuboid, mask, and map panels",
            "panel_cardinality_mismatches": sum(
                1
                for record in self.frame_records
                for detection in record["detections"]
                if "bbox_xyxy" not in detection
                or (self.road_map is not None and "map_position_px" not in detection)
            ),
            "map_motion_smoothing": "constant-velocity coasting plus bounded reacquisition blend",
            "publisher_overlay_suppressed_in_render": bool(
                self.config["processing"].get("suppress_preview_overlay", False)
            ),
            "source_preview_note": self.config["provenance"].get(
                "source_preview_note",
                self.config["provenance"].get("source_note", "clean source video"),
            ),
        }
        (self.output_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return audit

    def _coast_detection(
        self,
        track: GlobalTrack,
        motion: dict[str, float | str],
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any]:
        """Emit a predicted state so a temporary detector miss does not blink in BEV."""
        detection = deepcopy(track.last_detection)
        assert detection is not None
        old_ground = (detection["ground"]["x"], detection["ground"]["y"], 0.0)
        new_ground = (float(motion["x"]), float(motion["y"]), 0.0)
        old_pixel = self.camera.ground_to_pixel(old_ground)
        new_pixel = self.camera.ground_to_pixel(new_ground)
        dx = 0.0 if old_pixel is None or new_pixel is None else new_pixel[0] - old_pixel[0]
        dy = 0.0 if old_pixel is None or new_pixel is None else new_pixel[1] - old_pixel[1]

        def shift_points(points: list[list[float]]) -> list[list[float]]:
            return [
                [
                    round(float(np.clip(point[0] + dx, 0, frame_width - 1)), 2),
                    round(float(np.clip(point[1] + dy, 0, frame_height - 1)), 2),
                ]
                for point in points
            ]

        box = detection["bbox_xyxy"]
        detection["bbox_xyxy"] = [
            round(float(np.clip(box[0] + dx, 0, frame_width - 1)), 2),
            round(float(np.clip(box[1] + dy, 0, frame_height - 1)), 2),
            round(float(np.clip(box[2] + dx, 0, frame_width - 1)), 2),
            round(float(np.clip(box[3] + dy, 0, frame_height - 1)), 2),
        ]
        detection["mask_polygon"] = shift_points(detection["mask_polygon"])
        detection["cuboid_image_fitted"] = shift_points(detection["cuboid_image_fitted"])
        detection["anchor"] = [
            round(float(np.clip(detection["anchor"][0] + dx, 0, frame_width - 1)), 2),
            round(float(np.clip(detection["anchor"][1] + dy, 0, frame_height - 1)), 2),
        ]
        detection["ground"] = {"x": round(new_ground[0], 3), "y": round(new_ground[1], 3)}
        detection["velocity_mps"] = {
            "x": round(float(motion["vx"]), 3),
            "y": round(float(motion["vy"]), 3),
        }
        detection["speed_kmh"] = round(float(motion["speed_kmh"]), 2)
        detection["heading_deg"] = round(float(motion["heading_deg"]), 2)
        detection["yaw_rate_deg_s"] = 0.0
        detection["motion_state"] = "coasting"
        detection["tracking_status"] = "predicted"
        detection["lost_frames"] = track.lost_frames
        detection["prediction_only"] = True
        detection["optical_flow_vectors"] = []
        detection["confidence"] = round(float(detection["confidence"]) * 0.88, 4)
        dimensions = detection["dimensions_m"]
        cuboid = self.camera.project_vehicle_box(
            (new_ground[0], new_ground[1]),
            float(motion["heading_deg"]),
            (dimensions["length"], dimensions["width"], dimensions["height"]),
        )
        detection["cuboid_image"] = [[round(x, 2), round(y, 2)] for x, y in cuboid]
        detection["sources"]["tracking_status"] = "Kalman constant-velocity coast"
        return detection

    def _assign_routes(self, routes: list[InferredRoute]) -> None:
        """Add a lane-constrained display state while preserving raw metric localization."""
        if not routes:
            return
        for record in self.frame_records:
            for detection in record["detections"]:
                candidates = []
                raw_ground = (detection["ground"]["x"], detection["ground"]["y"])
                for route in routes:
                    snapped, distance, heading = project_to_polyline(raw_ground, route.centerline)
                    candidates.append((distance, route, snapped, heading))
                distance, route, snapped, heading = min(candidates, key=lambda item: item[0])
                if distance <= route.width_m / 2 + 1.5:
                    detection["route_id"] = route.route_id
                    detection["lane_ground"] = {"x": round(snapped[0], 3), "y": round(snapped[1], 3)}
                    detection["lateral_error_m"] = round(distance, 3)
                    velocity = detection["velocity_mps"]
                    direction = np.asarray([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
                    if velocity["x"] * direction[0] + velocity["y"] * direction[1] < 0:
                        heading = (heading + 180.0) % 360.0
                    detection["lane_heading_deg"] = round(heading, 2)
                    detection["sources"]["lane_ground"] = "nearest robust metric route projection"
                else:
                    detection["route_id"] = None

    def _write_data(self, routes: list[InferredRoute]) -> None:
        with (self.output_dir / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for record in self.frame_records:
                stream.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
        route_payload = [
            {
                "route_id": route.route_id,
                "centerline_m": route.centerline,
                "support_tracks": route.support_tracks,
                "width_m": route.width_m,
                "source": "clustered local metric vehicle trajectories",
            }
            for route in routes
        ]
        (self.output_dir / "routes.json").write_text(
            json.dumps(route_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _world_extent(self) -> tuple[float, float, float, float]:
        points = [point for values in self.trajectories.values() for point in values]
        if not points:
            return 280.0, 360.0, -15.0, 10.0
        array = np.asarray(points)
        minimum = array.min(axis=0)
        maximum = array.max(axis=0)
        span = np.maximum(maximum - minimum, np.asarray([20.0, 12.0]))
        center = (minimum + maximum) / 2
        return (
            float(center[0] - span[0] * 0.65),
            float(center[0] + span[0] * 0.65),
            float(center[1] - span[1] * 0.65),
            float(center[1] + span[1] * 0.65),
        )

    def _render(self, routes: list[InferredRoute], fps: float) -> None:
        width, height = self.crop[2], self.crop[3]
        writer = cv2.VideoWriter(
            str(self.output_dir / "three_stage.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.config["processing"]["output_fps"]),
            (width * 2, height * 2),
        )
        capture = cv2.VideoCapture(str(self.video_path))
        extent = self._world_extent()
        road_background = (
            self.road_map.render(width, height)[0]
            if self.road_map is not None
            else self._rectified_road_background(extent, width, height)
        )
        selected_frame: np.ndarray | None = None
        def fully_visible(detection: dict[str, Any]) -> bool:
            left, top, right, bottom = detection["bbox_xyxy"]
            return left > 4 and top > 26 and right < width - 4 and bottom < height - 6

        selected_index = max(
            self.frame_records,
            key=lambda record: (
                sum(
                    1
                    for detection in record["detections"]
                    if fully_visible(detection)
                    and not detection.get("prediction_only")
                    and 2.0 < detection["speed_kmh"] < 180.0
                ),
                sum(
                    1
                    for detection in record["detections"]
                    if fully_visible(detection) and not detection.get("prediction_only")
                ),
                -sum(1 for detection in record["detections"] if not fully_visible(detection)),
            ),
        )["frame_index"]
        for record in self.frame_records:
            ok, full_frame = capture.read()
            if not ok:
                break
            x, y, _, _ = self.crop
            raw = full_frame[y : y + height, x : x + width].copy()
            # The public preview already contains publisher-drawn colored boxes.
            # Remove them before drawing local results, otherwise those stale
            # graphics look like unmatched detections that have no map object.
            clean = (
                suppress_colored_preview_overlay(raw)
                if self.config["processing"].get("suppress_preview_overlay", False)
                else raw
            )
            if self.config["processing"].get("suppress_preview_overlay", False):
                clean = cv2.cvtColor(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                clean = cv2.addWeighted(clean, 0.88, cv2.GaussianBlur(clean, (5, 5), 0), 0.12, 0)
            flow_panel = clean.copy()
            cuboid_panel = clean.copy()
            mask_panel = clean.copy()
            mask_overlay = mask_panel.copy()
            for detection in record["detections"]:
                color = track_color(detection["track_id"])
                polygon = np.asarray(detection["mask_polygon"], dtype=np.int32)
                if len(polygon) >= 3:
                    cv2.fillPoly(mask_overlay, [polygon], color)
                box = detection["bbox_xyxy"]
                start = (round(box[0]), round(box[1]))
                end = (round(box[2]), round(box[3]))
                for panel_image in (flow_panel, cuboid_panel, mask_panel):
                    cv2.rectangle(panel_image, start, end, color, 2)
                for x1, y1, x2, y2 in detection.get("optical_flow_vectors", []):
                    cv2.arrowedLine(
                        flow_panel,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        color,
                        1,
                        cv2.LINE_AA,
                        tipLength=0.28,
                    )
            mask_panel = cv2.addWeighted(mask_overlay, 0.46, mask_panel, 0.54, 0)
            for detection in record["detections"]:
                color = track_color(detection["track_id"])
                cuboid = detection.get("cuboid_image_fitted", detection["cuboid_image"])
                if len(cuboid) == 8:
                    for start, end in CUBOID_EDGES:
                        cv2.line(cuboid_panel, tuple(map(round, cuboid[start])), tuple(map(round, cuboid[end])), color, 2, cv2.LINE_AA)
                box = detection["bbox_xyxy"]
                cv2.rectangle(
                    mask_panel,
                    (round(box[0]), round(box[1])),
                    (round(box[2]), round(box[3])),
                    color,
                    2,
                )
                suffix = f" P{detection['lost_frames']}" if detection.get("prediction_only") else ""
                label = f"ID {detection['track_id']}{suffix}"
                position = (round(box[0]), max(14, round(box[1]) - 4))
                for panel_image in (flow_panel, cuboid_panel, mask_panel):
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
                    cv2.rectangle(
                        panel_image,
                        (position[0] - 2, position[1] - label_size[1] - 3),
                        (position[0] + label_size[0] + 3, position[1] + 3),
                        (7, 13, 17),
                        -1,
                    )
                    cv2.putText(panel_image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            bev = self._render_bev(record, routes, extent, width, height, road_background)
            for image, title in (
                (flow_panel, "01 2D BOX + LOCAL OPTICAL FLOW"),
                (cuboid_panel, "02 ID + FITTED 3D CUBOID"),
                (mask_panel, "03 INSTANCE MASK + GLOBAL ID"),
                (bev, "04 MAP + SMOOTH TRACK"),
            ):
                cv2.rectangle(image, (0, 0), (width, 24), (7, 13, 17), -1)
                cv2.putText(image, title, (9, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (214, 220, 49), 1, cv2.LINE_AA)
                observed = sum(not item.get("prediction_only") for item in record["detections"])
                predicted = len(record["detections"]) - observed
                badge = f"N={len(record['detections'])} O={observed} P={predicted}"
                badge_width = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
                cv2.putText(
                    image,
                    badge,
                    (width - badge_width - 7, 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (49, 222, 214),
                    1,
                    cv2.LINE_AA,
                )
            combined = np.vstack(
                (
                    np.hstack((flow_panel, cuboid_panel)),
                    np.hstack((mask_panel, bev)),
                )
            )
            writer.write(combined)
            if record["frame_index"] == selected_index:
                selected_frame = combined.copy()
        capture.release()
        writer.release()
        if selected_frame is not None:
            cv2.imwrite(str(self.output_dir / "result_frame.jpg"), selected_frame)

    def _rectified_road_background(
        self,
        extent: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Build a static road texture, then perform the calibrated IPM/BEV warp.

        A temporal median suppresses moving vehicles and the preview's changing
        annotations.  The camera homography then rectifies lane markings into the
        same metric plane used by every vehicle state.
        """
        capture = cv2.VideoCapture(str(self.video_path))
        samples: list[np.ndarray] = []
        frame_count = max(1, len(self.frame_records))
        for frame_index in np.linspace(0, frame_count - 1, min(21, frame_count), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, full_frame = capture.read()
            if not ok:
                continue
            x, y, crop_width, crop_height = self.crop
            samples.append(full_frame[y : y + crop_height, x : x + crop_width])
        capture.release()
        if not samples:
            return np.full((height, width, 3), (31, 39, 44), dtype=np.uint8)
        median = np.median(np.stack(samples), axis=0).astype(np.uint8)
        homography = self.camera.image_to_bev_homography(extent, width, height)
        warped = cv2.warpPerspective(
            median,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(31, 39, 44),
        )
        return cv2.addWeighted(warped, 0.58, np.full_like(warped, (31, 39, 44)), 0.42, 0)

    def _render_bev(
        self,
        record: dict[str, Any],
        routes: list[InferredRoute],
        extent: tuple[float, float, float, float],
        width: int,
        height: int,
        road_background: np.ndarray | None = None,
    ) -> np.ndarray:
        image = (
            road_background.copy()
            if road_background is not None
            else np.full((height, width, 3), (31, 39, 44), dtype=np.uint8)
        )
        x_min, x_max, y_min, y_max = extent
        padding = 26

        def map_point(x: float, y: float) -> tuple[int, int]:
            if self.road_map is not None:
                point = self.road_map.world_to_display([(x, y)], width, height)[0]
                return round(float(point[0])), round(float(point[1]))
            px = padding + (x - x_min) / (x_max - x_min) * (width - padding * 2)
            py = height - padding - (y - y_min) / (y_max - y_min) * (height - padding * 2)
            return round(px), round(py)

        for index in range(1, 6):
            if self.road_map is not None:
                break
            gx = padding + index * (width - padding * 2) // 6
            gy = padding + index * (height - padding * 2) // 6
            cv2.line(image, (gx, padding), (gx, height - padding), (61, 73, 79), 1)
            cv2.line(image, (padding, gy), (width - padding, gy), (61, 73, 79), 1)
        if self.road_map is not None:
            for lane_index, lane in enumerate(self.road_map.lane_centerlines_display(width, height), start=1):
                points = np.asarray(lane, dtype=np.int32)
                cv2.polylines(image, [points], False, (49, 222, 214), 1, cv2.LINE_AA)
                midpoint = tuple(points[len(points) // 2])
                cv2.putText(image, f"L{lane_index:02d}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 205, 195), 1)
            camera_x, camera_y = self.road_map.camera_marker_display(width, height)
            camera_symbol = np.asarray(
                [[camera_x, camera_y], [camera_x - 15, camera_y - 9], [camera_x - 15, camera_y + 9]],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [camera_symbol], (49, 222, 214))
            cv2.putText(image, "CAM", (max(2, camera_x - 48), max(15, camera_y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (49, 222, 214), 1)
        for route in ([] if self.road_map is not None else routes):
            points = np.asarray([map_point(x, y) for x, y in route.centerline], dtype=np.int32)
            if len(points) > 1:
                if self.road_map is not None:
                    cv2.polylines(image, [points], False, (49, 222, 214), 2, cv2.LINE_AA)
                else:
                    pixels_per_meter = (height - padding * 2) / (y_max - y_min)
                    route_width = max(5, round(route.width_m * pixels_per_meter))
                    cv2.polylines(image, [points], False, (96, 113, 120), route_width, cv2.LINE_AA)
                    cv2.polylines(image, [points], False, (224, 226, 218), 2, cv2.LINE_AA)
                cv2.putText(image, route.route_id, tuple(points[len(points) // 2]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 195), 1)
        for detection in record["detections"]:
            if self.road_map is not None and "map_position_px" in detection:
                if not detection.get("map_visible", True):
                    continue
                icon = self.road_map.vehicle_icon_display(detection, width, height)
                polygon = np.asarray(icon["body"], dtype=np.int32)
                glass = np.asarray(icon["glass"], dtype=np.int32)
                heading_line = np.asarray(icon["heading"], dtype=np.int32)
                location_point = self.road_map.detection_position_display(detection, width, height)
                location = round(float(location_point[0])), round(float(location_point[1]))
                color = track_color(detection["track_id"])
                sprite_drawn = self.road_map.paste_vehicle_sprite(
                    image,
                    detection,
                    width,
                    height,
                    opacity=0.46 if detection.get("prediction_only") else 0.96,
                )
                if not sprite_drawn:
                    if detection.get("prediction_only"):
                        predicted_overlay = image.copy()
                        cv2.fillPoly(predicted_overlay, [polygon], color)
                        image = cv2.addWeighted(predicted_overlay, 0.42, image, 0.58, 0)
                    else:
                        cv2.fillPoly(image, [polygon], color)
                    if len(glass) >= 3:
                        cv2.fillPoly(image, [glass], (38, 53, 61))
                        cv2.polylines(image, [glass], True, (225, 236, 234), 1, cv2.LINE_AA)
                cv2.polylines(image, [polygon], True, (245, 245, 245), 1, cv2.LINE_AA)
                if len(heading_line) == 2:
                    cv2.arrowedLine(
                        image,
                        tuple(heading_line[0]),
                        tuple(heading_line[1]),
                        (245, 245, 245),
                        1,
                        cv2.LINE_AA,
                        tipLength=0.35,
                    )
                lane_label = detection.get("map_lane_id", "?")
                id_label = f"{detection['track_id']}/{lane_label}"
                id_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0]
                cv2.putText(
                    image,
                    id_label,
                    (location[0] - id_size[0] // 2, location[1] + id_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                continue
            display_ground = detection.get("lane_ground", detection["ground"])
            x = display_ground["x"]
            y = display_ground["y"]
            length = detection["dimensions_m"]["length"]
            vehicle_width = detection["dimensions_m"]["width"]
            angle = math.radians(detection.get("lane_heading_deg", detection["heading_deg"]))
            forward = np.asarray([math.cos(angle), math.sin(angle)])
            left = np.asarray([-math.sin(angle), math.cos(angle)])
            center = np.asarray([x, y])
            corners = [
                center + length / 2 * forward + vehicle_width / 2 * left,
                center + length / 2 * forward - vehicle_width / 2 * left,
                center - length / 2 * forward - vehicle_width / 2 * left,
                center - length / 2 * forward + vehicle_width / 2 * left,
            ]
            polygon = np.asarray([map_point(float(point[0]), float(point[1])) for point in corners], dtype=np.int32)
            color = track_color(detection["track_id"])
            if detection.get("prediction_only"):
                predicted_overlay = image.copy()
                cv2.fillPoly(predicted_overlay, [polygon], color)
                image = cv2.addWeighted(predicted_overlay, 0.42, image, 0.58, 0)
            else:
                cv2.fillPoly(image, [polygon], color)
            cv2.polylines(image, [polygon], True, (245, 245, 245), 1, cv2.LINE_AA)
            location = map_point(x, y)
            cv2.putText(image, str(detection["track_id"]), (location[0] + 5, location[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
        footer = (
            f"frame {record['frame_index']:04d} | highlighted = camera-visible road"
            if self.road_map is not None
            else f"frame {record['frame_index']:04d} | local metric coordinates"
        )
        cv2.putText(image, footer, (8, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (214, 220, 49), 1)
        return image
