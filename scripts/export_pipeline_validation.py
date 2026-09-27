from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.calibration import MetricCamera  # noqa: E402
from app.pipeline.road_map import RegisteredRoadMap  # noqa: E402
WIDTH = 1800
HEIGHT = 1200
MARGIN = 56
GAP = 20
HEADER = 116
TOP_HEIGHT = 390
BOTTOM_TOP = 540
BOTTOM_HEIGHT = 604
COLORS = ["#2ED6C4", "#FFD04A", "#35CFE3", "#60D7A9", "#F255C5", "#8B7CFF", "#FF8C42"]
CUBOID_EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def color_for(track_id: int) -> str:
    return COLORS[track_id % len(COLORS)]


def rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[index : index + 2], 16) for index in (1, 3, 5))


def panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rectangle(box, fill="#121A20", outline="#65747B", width=3)
    draw.rectangle((left, top, right, top + 50), fill="#080D11")
    draw.rectangle((left, top, left + 9, top + 50), fill="#31D6C4")
    draw.text((left + 24, top + 10), title, fill="#F2F5F3", font=font(24, True))
    draw.text((right - 17, top + 16), subtitle, fill="#31D6C4", font=font(13), anchor="ra")
    return left + 3, top + 53, right - 3, bottom - 3


def fit(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int], float]:
    left, top, right, bottom = box
    scale = min((right - left) / image.width, (bottom - top) / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    offset = (
        left + (right - left - resized.width) // 2,
        top + (bottom - top - resized.height) // 2,
    )
    return resized, offset, scale


def best_record(records: list[dict], frame_width: int, frame_height: int) -> dict:
    def fully_visible(detection: dict) -> bool:
        left, top, right, bottom = detection["bbox_xyxy"]
        return left > 4 and top > 26 and right < frame_width - 4 and bottom < frame_height - 6

    return max(
        records,
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
    )


def decode_frame(config: dict, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(ROOT / config["video_path"]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"frame {frame_index} could not be decoded")
    crop = config["crop"]
    return frame[
        crop["y"] : crop["y"] + crop["height"],
        crop["x"] : crop["x"] + crop["width"],
    ]


def clean_preview_overlay(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 120)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def rectified_road_texture(
    config: dict,
    extent: tuple[float, float, float, float],
    width: int,
    height: int,
) -> Image.Image:
    crop = config["crop"]
    camera = MetricCamera.from_config(ROOT, config)
    capture = cv2.VideoCapture(str(ROOT / config["video_path"]))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    for frame_index in np.linspace(0, max(0, frame_count - 1), 21, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, full_frame = capture.read()
        if ok:
            samples.append(
                full_frame[
                    crop["y"] : crop["y"] + crop["height"],
                    crop["x"] : crop["x"] + crop["width"],
                ]
            )
    capture.release()
    median = np.median(np.stack(samples), axis=0).astype(np.uint8)
    warped = cv2.warpPerspective(
        median,
        camera.image_to_bev_homography(extent, width, height),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(40, 49, 55),
    )
    warped = cv2.addWeighted(warped, 0.62, np.full_like(warped, (40, 49, 55)), 0.38, 0)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def draw_instance_panel(
    base: Image.Image,
    content: tuple[int, int, int, int],
    record: dict,
) -> Image.Image:
    resized, offset, scale = fit(base, content)
    annotated = resized.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for detection in record["detections"]:
        color = color_for(detection["track_id"])
        fill = (*rgb(color), 70 if detection.get("prediction_only") else 135)
        polygon = [
            (point[0] * scale, point[1] * scale)
            for point in detection["mask_polygon"]
        ]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=fill, outline=(*rgb(color), 255), width=3)
        cuboid = [
            (point[0] * scale, point[1] * scale)
            for point in detection.get("cuboid_image_fitted", detection["cuboid_image"])
        ]
        if len(cuboid) == 8:
            for start, end in CUBOID_EDGES:
                draw.line((cuboid[start], cuboid[end]), fill=(*rgb(color), 255), width=3)
        x1, y1, _, _ = detection["bbox_xyxy"]
        coast = f" · P{detection['lost_frames']}" if detection.get("prediction_only") else ""
        label = f"ID {detection['track_id']} · {detection['class_name'].upper()}{coast}"
        label_width = draw.textbbox((0, 0), label, font=font(11, True))[2] + 12
        label_x = max(2, min(annotated.width - label_width - 2, x1 * scale))
        label_y = max(2, y1 * scale - 24)
        draw.rounded_rectangle(
            (label_x, label_y, label_x + label_width, label_y + 21),
            radius=3,
            fill=(*rgb(color), 235),
        )
        draw.text((label_x + 6, label_y + 4), label, fill="#071014", font=font(11, True))
    return Image.alpha_composite(annotated, overlay).convert("RGB"), offset


def vehicle_polygon(detection: dict) -> list[tuple[float, float]]:
    display_ground = detection.get("lane_ground", detection["ground"])
    x = display_ground["x"]
    y = display_ground["y"]
    length = detection["dimensions_m"]["length"]
    width = detection["dimensions_m"]["width"]
    angle = math.radians(detection.get("lane_heading_deg", detection["heading_deg"]))
    forward = np.asarray([math.cos(angle), math.sin(angle)])
    left = np.asarray([-math.sin(angle), math.cos(angle)])
    center = np.asarray([x, y])
    return [
        tuple(center + length / 2 * forward + width / 2 * left),
        tuple(center + length / 2 * forward - width / 2 * left),
        tuple(center - length / 2 * forward - width / 2 * left),
        tuple(center - length / 2 * forward + width / 2 * left),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "ua_detrac_clean_intersection.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "ua-detrac-clean-intersection",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    crop = config["crop"]
    metric_camera = MetricCamera.from_config(ROOT, config)
    road_map = RegisteredRoadMap(ROOT, config["road_map"], metric_camera) if "road_map" in config else None
    audit = json.loads((args.input / "audit.json").read_text(encoding="utf-8"))
    routes = json.loads((args.input / "routes.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.input / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    record = best_record(records, int(crop["width"]), int(crop["height"]))
    frame_bgr = decode_frame(config, record["frame_index"])
    frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inference_bgr = (
        clean_preview_overlay(frame_bgr)
        if config["processing"].get("suppress_preview_overlay", False)
        else frame_bgr
    )
    inference_frame = Image.fromarray(cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB))

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#050708")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (WIDTH // 2, 25),
        "CAROM V1 — LOCAL MONOCULAR TRAFFIC RECONSTRUCTION",
        fill="#F4F5F2",
        font=font(37, True),
        anchor="ma",
    )
    draw.text(
        (WIDTH // 2, 73),
        f"LOCAL CV VALIDATION · FRAME #{record['frame_index']} · 3D SCENE STAGE OMITTED",
        fill="#A3B0B5",
        font=font(18),
        anchor="ma",
    )

    column_width = (WIDTH - MARGIN * 2 - GAP) // 2
    raw_box = (MARGIN, HEADER, MARGIN + column_width, HEADER + TOP_HEIGHT)
    instance_box = (MARGIN + column_width + GAP, HEADER, WIDTH - MARGIN, HEADER + TOP_HEIGHT)
    map_box = (MARGIN, BOTTOM_TOP, WIDTH - MARGIN, BOTTOM_TOP + BOTTOM_HEIGHT)
    raw_content = panel(draw, raw_box, "01 · RAW FIXED-CAMERA FRAME", config["camera_id"].upper())
    instance_content = panel(draw, instance_box, "02 · INSTANCE / ID / ESTIMATED BOX", "LOCAL YOLO-SEG + TRACKING")
    map_content = panel(
        draw,
        map_box,
        (
            "03 · METRIC LANE MODEL + ESTIMATED VEHICLE STATES"
            if road_map is not None
            else "03 · RECTIFIED ROAD PLANE + VEHICLE STATES"
        ),
        (
            "8-LANE SCHEMATIC / NOMINAL SCALE"
            if road_map is not None
            else "PLANAR HOMOGRAPHY + LOCAL TRACK STATES"
        ),
    )

    raw_image, raw_offset, _ = fit(frame, raw_content)
    canvas.paste(raw_image, raw_offset)
    input_note = config.get("display", {}).get(
        "input_note", "CLEAN FIXED-CAMERA SOURCE · NO BURNED-IN LABELS"
    )
    note_width = draw.textbbox((0, 0), input_note, font=font(13, True))[2] + 22
    draw.rectangle(
        (raw_content[0] + 10, raw_content[3] - 34, raw_content[0] + note_width, raw_content[3] - 8),
        fill="#071014",
    )
    draw.text((raw_content[0] + 18, raw_content[3] - 29), input_note, fill="#FFB469", font=font(13, True))

    instance_image, instance_offset = draw_instance_panel(inference_frame, instance_content, record)
    canvas.paste(instance_image, instance_offset)

    trajectories: dict[int, list[tuple[float, float]]] = defaultdict(list)
    map_trajectories: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for item in records:
        for detection in item["detections"]:
            display_ground = detection.get("lane_ground", detection["ground"])
            trajectories[detection["track_id"]].append(
                (display_ground["x"], display_ground["y"])
            )
            if "map_position_px" in detection:
                map_position = detection["map_position_px"]
                map_trajectories[detection["track_id"]].append(
                    (map_position["x"], map_position["y"])
                )
    all_points = [
        point
        for trajectory in trajectories.values()
        for point in trajectory
    ] + [
        tuple(point)
        for route in routes
        for point in route["centerline_m"]
    ]
    values = np.asarray(all_points, dtype=float)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    span = np.maximum(maximum - minimum, np.asarray([20.0, 12.0]))
    center = (minimum + maximum) / 2
    x_min, x_max = center[0] - span[0] * 0.62, center[0] + span[0] * 0.62
    y_min, y_max = center[1] - span[1] * 0.70, center[1] + span[1] * 0.70
    ml, mt, mr, mb = map_content
    plot = (ml + 46, mt + 24, mr - 46, mb - 66)
    pl, pt, pr, pb = plot

    def world(point: tuple[float, float] | list[float]) -> tuple[float, float]:
        x, y = point
        if road_map is not None:
            mapped = road_map.world_to_display([(x, y)], pr - pl, pb - pt)[0]
            return pl + float(mapped[0]), pt + float(mapped[1])
        px = pl + (x - x_min) / (x_max - x_min) * (pr - pl)
        py = pb - (y - y_min) / (y_max - y_min) * (pb - pt)
        return px, py

    if road_map is not None:
        map_image, _ = road_map.render(pr - pl, pb - pt)
        texture = Image.fromarray(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
    else:
        texture = rectified_road_texture(
            config,
            (x_min, x_max, y_min, y_max),
            pr - pl,
            pb - pt,
        )
    canvas.paste(texture, (pl, pt))
    draw.rectangle(plot, outline="#7A878C", width=3)
    if road_map is not None:
        camera_x, camera_y = road_map.camera_marker_display(pr - pl, pb - pt)
        camera_x += pl
        camera_y += pt
        draw.polygon(
            [(camera_x, camera_y), (camera_x - 22, camera_y - 13), (camera_x - 22, camera_y + 13)],
            fill="#31DED6",
            outline="#FFFFFF",
        )
        draw.text((camera_x - 28, camera_y - 32), "CAMERA", fill="#31DED6", font=font(13, True), anchor="ra")
    for index in range(1, 8):
        if road_map is not None:
            break
        gx = pl + index * (pr - pl) / 8
        gy = pt + index * (pb - pt) / 8
        draw.line((gx, pt, gx, pb), fill="#46545B", width=1)
        draw.line((pl, gy, pr, gy), fill="#46545B", width=1)

    if road_map is not None:
        for lane_index, lane in enumerate(
            road_map.lane_centerlines_display(pr - pl, pb - pt),
            start=1,
        ):
            points = [(pl + float(point[0]), pt + float(point[1])) for point in lane]
            draw.line(points, fill="#D6DE31", width=2)
            draw.text(points[len(points) // 2], f"L{lane_index:02d}", fill="#F0F2EC", font=font(12, True), anchor="mm")
    for route in ([] if road_map is not None else routes):
        points = [world(point) for point in route["centerline_m"]]
        if road_map is not None:
            draw.line(points, fill="#D6DE31", width=3)
        else:
            route_width_px = max(10, int(route["width_m"] / (y_max - y_min) * (pb - pt)))
            draw.line(points, fill="#59686E", width=route_width_px)
            draw.line(points, fill="#D9DDD6", width=2)
        midpoint = points[len(points) // 2]
        draw.text(midpoint, route["route_id"], fill="#F0F2EC", font=font(13, True), anchor="mm")

    active_ids = {detection["track_id"] for detection in record["detections"]}
    for track_id in active_ids:
        if road_map is not None and map_trajectories[track_id]:
            displayed = road_map.original_to_display(
                map_trajectories[track_id][-18:],
                pr - pl,
                pb - pt,
            )
            points = [(pl + float(point[0]), pt + float(point[1])) for point in displayed]
        else:
            points = [world(point) for point in trajectories[track_id][-18:]]
        if len(points) > 1:
            draw.line(points, fill=color_for(track_id), width=3)

    sprite_ids: set[int] = set()
    if road_map is not None and road_map.vehicle_sprites:
        map_region = cv2.cvtColor(np.asarray(canvas.crop(plot)), cv2.COLOR_RGB2BGR)
        for detection in record["detections"]:
            if "map_position_px" not in detection:
                continue
            if not detection.get("map_visible", True):
                continue
            if road_map.paste_vehicle_sprite(
                map_region,
                detection,
                pr - pl,
                pb - pt,
                opacity=0.46 if detection.get("prediction_only") else 0.96,
            ):
                sprite_ids.add(int(detection["track_id"]))
        canvas.paste(Image.fromarray(cv2.cvtColor(map_region, cv2.COLOR_BGR2RGB)), (pl, pt))
        draw = ImageDraw.Draw(canvas)

    occupied_labels: list[tuple[float, float, float, float]] = []
    for detection in record["detections"]:
        if road_map is not None and not detection.get("map_visible", True):
            continue
        color = color_for(detection["track_id"])
        if road_map is not None and "map_position_px" in detection:
            icon = road_map.vehicle_icon_display(
                detection,
                pr - pl,
                pb - pt,
            )
            polygon = [(pl + float(point[0]), pt + float(point[1])) for point in icon["body"]]
            glass = [(pl + float(point[0]), pt + float(point[1])) for point in icon["glass"]]
            heading = [(pl + float(point[0]), pt + float(point[1])) for point in icon["heading"]]
            displayed_location = road_map.detection_position_display(
                detection,
                pr - pl,
                pb - pt,
            )
            location = pl + float(displayed_location[0]), pt + float(displayed_location[1])
        else:
            polygon = [world(point) for point in vehicle_polygon(detection)]
            glass = []
            heading = []
            display_ground = detection.get("lane_ground", detection["ground"])
            location = world((display_ground["x"], display_ground["y"]))
        if int(detection["track_id"]) not in sprite_ids:
            draw.polygon(polygon, fill=color, outline="#FFFFFF", width=2)
            if glass:
                draw.polygon(glass, fill="#26353D", outline="#EDF5F3", width=1)
            if len(heading) == 2:
                draw.line(heading, fill="#FFFFFF", width=2)
        coast = f" · P{detection['lost_frames']}" if detection.get("prediction_only") else ""
        lane = detection.get("map_lane_id", "lane ?")
        conflict = " !" if detection.get("map_overlap_with") else ""
        if detection.get("map_lane_status") in {"association_pending", "outside_road_model"}:
            conflict += " ?"
        label = f"{detection['track_id']}/{lane}{coast}{conflict}"
        label_width = draw.textbbox((0, 0), label, font=font(13, True))[2] + 16
        tx = min(pr - label_width - 4, location[0] + 10)
        ty = max(pt + 4, location[1] - 37)
        for dy in [-37, 12, -66, 41, -95, 70, -124, 99]:
            candidate_y = max(pt + 4, min(pb - 31, location[1] + dy))
            rectangle = (tx, candidate_y, tx + label_width, candidate_y + 27)
            if all(rectangle[2] < box[0] or rectangle[0] > box[2] or rectangle[3] < box[1] or rectangle[1] > box[3] for box in occupied_labels):
                ty = candidate_y
                break
        occupied_labels.append((tx, ty, tx + label_width, ty + 27))
        draw.line((location, (tx, ty + 13)), fill=color, width=1)
        draw.rounded_rectangle((tx, ty, tx + label_width, ty + 27), radius=4, fill="#071014", outline=color, width=2)
        draw.text((tx + 8, ty + 6), label, fill="#F3F5F2", font=font(13, True))

    validation = (
        f"FRAMES {audit['frames_processed']}  |  DETECTIONS {audit['detections_emitted']}  |  "
        f"TRACK IDS {len(audit['stable_track_ids'])}  |  LANES {len(road_map.lane_centerlines) if road_map else audit['inferred_routes']}  |  "
        f"LLM CALLS {audit['llm_runtime_calls']}  |  MANUAL VEHICLE FIELDS {audit['manual_vehicle_annotations']}"
    )
    draw.rectangle((ml + 18, mb - 48, mr - 18, mb - 12), fill="#071014", outline="#31434B")
    draw.text((ml + 32, mb - 39), validation, fill="#31D6C4", font=font(14, True))
    draw.text(
        (mr - 20, mt + 9),
        "ESTIMATED SCALE | ! = FOOTPRINT CONFLICT | P = PREDICTION",
        fill="#FFB469",
        font=font(12, True),
        anchor="ra",
    )
    if road_map is not None:
        draw.rounded_rectangle((pl + 14, pt + 12, pl + 335, pt + 43), radius=5, fill="#071014")
        draw.text(
            (pl + 27, pt + 20),
            "BRIGHT: CAMERA VISIBLE  ·  DARK: OUTSIDE FOV",
            fill="#D6DE31",
            font=font(12, True),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
