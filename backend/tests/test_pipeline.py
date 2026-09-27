import json
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.calibration import MetricCamera
from app.pipeline.motion import MotionRegistry
from app.pipeline.optical_flow import sparse_vehicle_flow
from app.pipeline.road_map import RegisteredRoadMap
from app.pipeline.routes import infer_routes
from app.pipeline.tracking import GlobalTrackManager


def test_metric_camera_round_trip_on_ground_plane() -> None:
    intrinsic = np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])
    world_to_camera = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 10.0], [0.0, 0.0, 0.0, 1.0]]
    )
    camera = MetricCamera(intrinsic, np.zeros(5), world_to_camera, 640, 480, 640, 480)
    pixel = camera.ground_to_pixel((1.0, 2.0, 0.0))
    assert pixel is not None
    assert np.allclose(camera.pixel_to_ground(*pixel), (1.0, 2.0))
    homogeneous = camera.image_to_ground_homography @ np.asarray([*pixel, 1.0])
    assert np.allclose(homogeneous[:2] / homogeneous[2], (1.0, 2.0))


def test_planar_correspondence_camera_reprojects_control_points() -> None:
    image = [[220.0, 120.0], [420.0, 120.0], [80.0, 460.0], [560.0, 460.0]]
    ground = [[-5.0, 40.0], [5.0, 40.0], [-10.0, 0.0], [10.0, 0.0]]
    camera = MetricCamera.from_planar_correspondences(
        image, ground, frame_width=640, frame_height=480, focal_length_px=700.0
    )
    recovered = [camera.pixel_to_ground(*point) for point in image]
    assert np.allclose(recovered, ground, atol=1e-5)
    projected = [camera.ground_to_pixel((x, y, 0.0)) for x, y in ground]
    assert np.allclose(projected, image, atol=1e-5)


def test_motion_registry_derives_speed_from_metric_positions() -> None:
    registry = MotionRegistry()
    state = {}
    for index in range(8):
        state = registry.update(7, index, float(index), 0.0, 1.0)
    assert state["speed_kmh"] > 2.0
    assert state["motion_state"] == "moving"
    predicted = registry.coast(7, 8, 1.0)
    assert predicted["x"] > state["x"]
    assert predicted["motion_state"] == "coasting"


def test_route_inference_clusters_parallel_trajectories() -> None:
    trajectories = {
        1: [(float(index), 0.0) for index in range(12)],
        2: [(float(index), 0.4) for index in range(12)],
        3: [(float(index), 8.0) for index in range(12)],
    }
    routes = infer_routes(trajectories, minimum_points=8, threshold_m=2.0)
    assert len(routes) == 2
    assert sorted(len(route.support_tracks) for route in routes) == [1, 2]
    assert all(len(route.centerline) == 40 for route in routes)
    assert any(max(abs(y) for x, y in route.centerline) < 0.5 for route in routes)


def test_route_inference_removes_single_frame_lateral_spike() -> None:
    points = [(float(index), 0.1 * np.sin(index)) for index in range(15)]
    points[7] = (7.0, 9.0)
    routes = infer_routes({11: points}, minimum_points=8, threshold_m=2.0)
    assert len(routes) == 1
    assert max(abs(y) for x, y in routes[0].centerline) < 1.0


def test_global_track_manager_keeps_id_during_detection_gap() -> None:
    manager = GlobalTrackManager(max_lost_frames=3, max_match_distance_m=4.0)
    observation = {
        "detector_track_id": 91,
        "ground_measurement": (10.0, 2.0),
        "class_name": "car",
    }
    assigned = manager.assign(0, [observation], fps=10.0)
    global_id = assigned[0][0]
    manager.tracks[global_id].last_detection = {"track_id": global_id}
    manager.assign(1, [], fps=10.0)
    manager.assign(2, [], fps=10.0)
    assert manager.coast_tracks()[0].global_id == global_id

    reacquired = dict(observation, detector_track_id=402, ground_measurement=(10.6, 2.0))
    assert manager.assign(3, [reacquired], fps=10.0)[0][0] == global_id
    assert manager.tracks[global_id].lost_frames == 0
    assert manager.reacquisitions == 1
    assert manager.max_recovered_gap == 2


def test_global_track_manager_retires_after_maximum_gap() -> None:
    manager = GlobalTrackManager(max_lost_frames=2)
    global_id = manager.assign(
        0,
        [{"detector_track_id": 1, "ground_measurement": (0.0, 0.0), "class_name": "car"}],
        fps=10.0,
    )[0][0]
    manager.assign(1, [], fps=10.0)
    manager.assign(2, [], fps=10.0)
    manager.assign(3, [], fps=10.0)
    assert manager.tracks[global_id].retired


def test_registered_road_map_matches_control_points_and_limits_visibility() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/tumtraf_r01_s01_s040_50mm.json").read_text())
    crop = config["crop"]
    camera = MetricCamera.from_tumtraf(
        root / config["calibration_path"],
        frame_width=crop["width"],
        frame_height=crop["height"],
        transform_key=config["ground_transform_key"],
    )
    road_map = RegisteredRoadMap(root, config["road_map"], camera)

    mapped = road_map.world_to_original(config["road_map"]["world_control_points_m"])
    expected = np.asarray(config["road_map"]["image_control_points_px"], dtype=float)
    assert np.allclose(mapped, expected, atol=0.1)

    mask = road_map.visibility_mask(640, 180)
    visible_fraction = float(np.mean(mask > 0))
    assert 0.01 < visible_fraction < 0.95
    assert np.count_nonzero(mask[:, 128]) > np.count_nonzero(mask[:, 576])
    camera_x, _ = road_map.camera_marker_display(640, 180)
    assert camera_x > 560

    records = [
        {
            "detections": [
                {
                    "track_id": 7,
                    "prediction_only": False,
                    "ground": {"x": x, "y": y},
                    "sources": {},
                }
            ]
        }
        for x, y in ((250.0, 0.0), (270.0, 0.8), (290.0, -0.6))
    ]
    road_map.assign_map_lanes(records)
    lane_ids = {record["detections"][0]["map_lane_id"] for record in records}
    map_x = {record["detections"][0]["map_position_px"]["x"] for record in records}
    assert len(lane_ids) == 1
    assert len(map_x) == 1

    synthetic_records = []
    lane_tracks = [(-4.0, (28, 29)), (-1.0, (30, 31)), (1.0, (32, 33)), (3.2, (34, 36))]
    for frame_index in range(10):
        detections = []
        for lateral_offset, track_ids in lane_tracks:
            for pair_index, track_id in enumerate(track_ids):
                x = 245.0 + frame_index * 5.0 + pair_index * 22.0
                y = 0.0015 * (x - 290.0) ** 2 + lateral_offset
                detections.append(
                    {
                        "track_id": track_id,
                        "prediction_only": False,
                        "ground": {"x": x, "y": y},
                        "sources": {},
                    }
                )
        synthetic_records.append({"frame_index": frame_index, "detections": detections})
    road_map.assign_map_lanes(synthetic_records)
    assigned = {
        detection["track_id"]: detection["map_lane_id"]
        for detection in synthetic_records[-1]["detections"]
    }
    assert assigned[28] == assigned[29]
    assert assigned[34] == assigned[36]
    assert assigned[28] != assigned[34]
    assert len(set(assigned.values())) == 4


def test_registered_road_map_renders_vehicle_sprite() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/user_bridge_traffic.json").read_text())
    camera = MetricCamera.from_config(root, config)
    road_map = RegisteredRoadMap(root, config["road_map"], camera)
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    detection = {
        "track_id": 1,
        "class_name": "car",
        "map_lane_id": "L01",
        "map_lane_segment": 0,
        "map_position_px": {"x": 216.0, "y": 520.0},
        "ground": {"x": -12.0, "y": 50.0},
        "velocity_mps": {"x": 0.0, "y": 8.0},
        "dimensions_m": {"length": 4.55, "width": 1.82, "height": 1.48},
    }
    assert road_map.paste_vehicle_sprite(image, detection, 640, 360)
    assert np.count_nonzero(image) > 30
    corners = road_map.vehicle_polygon_display(detection, 640, 360)
    displayed_length = np.linalg.norm((corners[0] + corners[1]) / 2 - (corners[2] + corners[3]) / 2)
    displayed_width = np.linalg.norm(corners[0] - corners[1])
    assert np.isclose(displayed_length / displayed_width, 4.55 / 1.82, rtol=0.03)


def test_bidirectional_lane_assignment_uses_all_configured_lanes_and_is_monotonic() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/user_bridge_traffic.json").read_text())
    road_map = RegisteredRoadMap(root, config["road_map"], MetricCamera.from_config(root, config))
    lane_x = [-12.25, -8.75, -5.25, -1.75, 1.75, 5.25, 8.75, 12.25]
    records = []
    for frame_index in range(12):
        detections = []
        for lane_index, x in enumerate(lane_x):
            direction = 1 if lane_index < 4 else -1
            y = 70.0 - direction * frame_index * 0.8
            detections.append(
                {
                    "track_id": lane_index + 1,
                    "class_name": "car",
                    "prediction_only": False,
                    "ground": {"x": x, "y": y},
                    "velocity_mps": {"x": 0.0, "y": -direction * 8.0},
                    "dimensions_m": {"length": 4.55, "width": 1.82, "height": 1.48},
                    "sources": {},
                }
            )
        records.append({"frame_index": frame_index, "detections": detections})
    road_map.assign_map_lanes(records)
    assert {item["map_lane_id"] for item in records[-1]["detections"]} == {
        f"L{index:02d}" for index in range(1, 9)
    }
    for lane_index in range(8):
        progress = [frame["detections"][lane_index]["map_lane_progress_px"] for frame in records]
        signed_steps = np.diff(progress) * config["road_map"]["lane_directions"][lane_index]
        assert np.all(signed_steps >= -0.002)


def test_sparse_vehicle_flow_tracks_translated_roi_features() -> None:
    previous = np.zeros((120, 160, 3), dtype=np.uint8)
    for x in range(45, 105, 12):
        for y in range(35, 85, 12):
            cv2.circle(previous, (x, y), 2, (255, 255, 255), -1)
    transform = np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    current = cv2.warpAffine(previous, transform, (160, 120))
    vectors = sparse_vehicle_flow(previous, current, np.asarray([35, 25, 115, 95]))
    assert len(vectors) >= 6
    displacement = np.asarray([[x2 - x1, y2 - y1] for x1, y1, x2, y2 in vectors])
    assert np.allclose(np.median(displacement, axis=0), (4.0, 2.0), atol=0.4)


def test_map_endpoint_extension_does_not_collapse_distant_vehicles() -> None:
    line = np.asarray([[0., 0.], [0., 100.]], dtype=np.float32)
    for value in [-50., 140., 175.]:
        point, _, _, progress = RegisteredRoadMap._project_to_polyline_progress(np.asarray([1., value]), line)
        restored, _ = RegisteredRoadMap._point_at_progress(line, progress)
        assert np.allclose(point, [0, value])
        assert np.allclose(restored, point)


def test_mapping_is_idempotent_preserves_raw_and_reports_collisions() -> None:
    from copy import deepcopy
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/user_bridge_traffic.json").read_text())
    road = RegisteredRoadMap(root, config["road_map"], MetricCamera.from_config(root, config))
    records = [{"detections": [{
        "track_id": tid, "ground": {"x": -12.25, "y": 150.0 + tid * .1},
        "prediction_only": False, "sources": {},
        "dimensions_m": {"length": 4.55, "width": 1.82},
    } for tid in [1, 2]]}]
    raw = deepcopy([d['ground'] for d in records[0]['detections']])
    road.assign_map_lanes(records)
    first = deepcopy(records)
    road.assign_map_lanes(records)
    assert records == first
    assert [d['ground'] for d in records[0]['detections']] == raw
    assert all(d['map_visible'] for d in records[0]['detections'])
    assert all(d['map_overlap_with'] for d in records[0]['detections'])
    assert records[0]['detections'][0]['map_position_px'] != records[0]['detections'][1]['map_position_px']


def test_lane_change_requires_persistent_geometric_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/user_bridge_traffic.json").read_text())
    road = RegisteredRoadMap(root, config["road_map"], MetricCamera.from_config(root, config))
    records = [{"detections": [{"track_id": 1, "ground": {"x": x, "y": 80 - i}, "sources": {}, "prediction_only": False}]} for i, x in enumerate([-12.25] * 5 + [-8.75] * 10)]
    road.assign_map_lanes(records)
    assert records[0]['detections'][0]['map_lane_id'] == 'L01'
    assert records[7]['detections'][0]['map_tracking_lane_id'] == 'L01'
    assert records[7]['detections'][0]['map_lane_id'] == 'L02'
    assert records[7]['detections'][0]['map_lane_status'] == 'association_pending'
    assert records[-1]['detections'][0]['map_lane_id'] == 'L02'
