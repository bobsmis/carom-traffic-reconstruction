from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.geometry import project_points, solve_homography, vehicle_footprint  # noqa: E402


CANVAS_W = 1800
CANVAS_H = 1200
MARGIN = 58
GAP = 22
TITLE_H = 116
TOP_H = 380
BOTTOM_Y = TITLE_H + TOP_H + 50
BOTTOM_H = CANVAS_H - BOTTOM_Y - 54


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def fit_image(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    left, top, right, bottom = box
    target_w, target_h = right - left, bottom - top
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return resized, (left + (target_w - resized.width) // 2, top + (target_h - resized.height) // 2)


def scale_point(point: list[float], source_size: tuple[int, int], image_size: tuple[int, int], offset: tuple[int, int]) -> tuple[float, float]:
    sx = image_size[0] / source_size[0]
    sy = image_size[1] / source_size[1]
    return offset[0] + point[0] * sx, offset[1] + point[1] * sy


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, subtitle: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rectangle(box, fill="#111820", outline="#58656b", width=3)
    draw.rectangle((left, top, right, top + 50), fill="#080d11")
    draw.rectangle((left, top, left + 9, top + 50), fill="#31d6c4")
    draw.text((left + 24, top + 10), title, fill="#f4f6f5", font=font(24, True))
    draw.text((right - 18, top + 15), subtitle, fill="#31d6c4", font=font(14), anchor="ra")
    return left + 3, top + 53, right - 3, bottom - 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    scene = json.loads((ROOT / "backend" / "data" / "scene.json").read_text(encoding="utf-8"))
    frame = Image.open(ROOT / "backend" / "data" / "frame.jpg").convert("RGB")
    homography = solve_homography(
        [tuple(point) for point in scene["calibration"]["image_points"]],
        [tuple(point) for point in scene["calibration"]["ground_points"]],
    )
    positions = project_points(homography, [tuple(item["anchor"]) for item in scene["detections"]])

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "#050708")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (CANVAS_W // 2, 26),
        "CAROM — VEHICLE LOCALIZATION FROM A FIXED MONOCULAR CAMERA",
        fill="#f4f5f2",
        font=font(37, True),
        anchor="ma",
    )
    draw.text(
        (CANVAS_W // 2, 74),
        "OFFLINE RECONSTRUCTION · FRAME #2901 · FIRST THREE STAGES",
        fill="#a3b0b5",
        font=font(19),
        anchor="ma",
    )

    top_width = (CANVAS_W - MARGIN * 2 - GAP) // 2
    raw_box = (MARGIN, TITLE_H, MARGIN + top_width, TITLE_H + TOP_H)
    mask_box = (MARGIN + top_width + GAP, TITLE_H, CANVAS_W - MARGIN, TITLE_H + TOP_H)
    map_box = (MARGIN, BOTTOM_Y, CANVAS_W - MARGIN, BOTTOM_Y + BOTTOM_H)

    raw_content = panel(draw, raw_box, "01 · RAW CAMERA FRAME", "MONOCULAR RGB")
    raw_scaled, raw_offset = fit_image(frame, raw_content)
    canvas.paste(raw_scaled, raw_offset)
    draw.rectangle(
        (raw_content[0] + 12, raw_content[3] - 36, raw_content[0] + 302, raw_content[3] - 10),
        fill="#071014",
    )
    draw.text(
        (raw_content[0] + 22, raw_content[3] - 31),
        "2020-03-16 13:07:02 · FIXED CAM 01",
        fill="#eaf0ed",
        font=font(14),
    )

    mask_content = panel(draw, mask_box, "02 · INSTANCE MASKS + VOLUME BOXES", "5 TRACKS")
    processed = raw_scaled.convert("RGBA")
    overlay = Image.new("RGBA", processed.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    source_size = (scene["frame"]["width"], scene["frame"]["height"])
    local_offset = (0, 0)
    for item in scene["detections"]:
        mask = [scale_point(point, source_size, processed.size, local_offset) for point in item["mask_polygon"]]
        rgb = tuple(int(item["color"][i : i + 2], 16) for i in (1, 3, 5))
        odraw.polygon(mask, fill=(*rgb, 150), outline=(*rgb, 255), width=3)
        volume = [scale_point(point, source_size, processed.size, local_offset) for point in item["volume_box"]]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for start, end in edges:
            odraw.line((volume[start], volume[end]), fill=(*rgb, 255), width=3)
        lx, ly = mask[0]
        label = f"ID {item['track_id']} · {item['class_name'].upper()}"
        label_w = odraw.textbbox((0, 0), label, font=font(13, True))[2] + 16
        label_y = max(4, ly - 28)
        odraw.rounded_rectangle((lx, label_y, lx + label_w, label_y + 23), radius=3, fill=(*rgb, 235))
        odraw.text((lx + 8, label_y + 4), label, fill="#071014", font=font(13, True))
    processed = Image.alpha_composite(processed, overlay).convert("RGB")
    mask_offset = (
        mask_content[0] + (mask_content[2] - mask_content[0] - processed.width) // 2,
        mask_content[1] + (mask_content[3] - mask_content[1] - processed.height) // 2,
    )
    canvas.paste(processed, mask_offset)

    map_content = panel(draw, map_box, "03 · METRIC ROAD PLANE + VEHICLE STATES", "LOCAL WORLD FRAME / m")
    ml, mt, mr, mb = map_content
    plot = (ml + 44, mt + 26, mr - 44, mb - 34)
    pl, pt, pr, pb = plot
    extent = scene["ground_extent"]

    def world_to_map(x: float, y: float) -> tuple[float, float]:
        px = pl + (y - extent["y_min"]) / (extent["y_max"] - extent["y_min"]) * (pr - pl)
        py = pb - (x - extent["x_min"]) / (extent["x_max"] - extent["x_min"]) * (pb - pt)
        return px, py

    road = [world_to_map(x, y) for x, y in scene["road_model"]["surface_polygon_m"]]
    draw.polygon(road, fill="#283137", outline="#8b989c", width=3)

    for boundary in scene["road_model"]["lane_boundaries"]:
        points = [world_to_map(x, y) for x, y in boundary["points_m"]]
        if boundary["kind"] == "solid":
            draw.line(points, fill="#e6e3d4", width=3)
        else:
            (x1, y1), (x2, y2) = points
            length = math.dist(points[0], points[1])
            steps = max(1, int(length / 34))
            for index in range(steps):
                a = index / steps
                b = min(1.0, a + 0.5 / steps)
                draw.line((x1 + (x2-x1)*a, y1 + (y2-y1)*a, x1 + (x2-x1)*b, y1 + (y2-y1)*b), fill="#bfc5c3", width=2)

    for stop_line in scene["road_model"]["stop_lines"]:
        draw.line([world_to_map(x, y) for x, y in stop_line["points_m"]], fill="#ffffff", width=7)

    for zone in scene["road_model"]["zones"]:
        polygon = [world_to_map(x, y) for x, y in zone["polygon_m"]]
        draw.polygon(polygon, outline="#aeb8bb", width=2)
        x_values = [p[0] for p in polygon]
        for x in range(round(min(x_values)), round(max(x_values)), 18):
            draw.line((x, min(p[1] for p in polygon), x + 38, max(p[1] for p in polygon)), fill="#65747a", width=2)

    for lane in scene["road_model"]["lanes"]:
        lx, ly = world_to_map(lane["centerline_m"][0][0], 3.0)
        draw.text((lx, ly), lane["id"], fill="#819198", font=font(14, True), anchor="mm")

    for meter in range(0, 73, 12):
        x, _ = world_to_map(0, meter)
        draw.line((x, pt, x, pb), fill="#536169", width=1)
        draw.text((x, pb + 8), f"{meter}m", fill="#8ea0a7", font=font(13), anchor="ma")

    for item, (x, y) in zip(scene["detections"], positions, strict=True):
        rgb = item["color"]
        trajectory = [world_to_map(tx, ty) for tx, ty in item["trajectory_m"]]
        if len(trajectory) > 1:
            draw.line(trajectory, fill=rgb, width=4)
        footprint = vehicle_footprint(
            (x, y),
            item["dimensions_m"]["length"],
            item["dimensions_m"]["width"],
            item["heading_deg"],
        )
        vehicle = [world_to_map(px, py) for px, py in footprint]
        draw.polygon(vehicle, fill=rgb, outline="#ffffff")
        center = world_to_map(x, y)
        angle = math.radians(item["heading_deg"])
        arrow = (center[0] + math.cos(angle) * 30, center[1] - math.sin(angle) * 30)
        draw.line((center, arrow), fill="#ffffff", width=3)
        label = f"ID {item['track_id']} · {item['lane_id']}  |  {item['speed_kmh']:.1f} km/h"
        tx = min(pr - 220, center[0] + 12)
        ty = max(pt + 8, center[1] - 44)
        draw.rounded_rectangle((tx, ty, tx + 210, ty + 32), radius=4, fill="#071014", outline=rgb, width=2)
        draw.text((tx + 8, ty + 7), label, fill="#eef2ef", font=font(14, True))

    draw.line((pl + 18, pt + 22, pl + 98, pt + 22), fill="#ff9e3d", width=4)
    draw.polygon(((pl + 106, pt + 22), (pl + 92, pt + 14), (pl + 92, pt + 30)), fill="#ff9e3d")
    draw.text((pl + 18, pt - 2), "+Y / ROAD DIRECTION", fill="#ffb469", font=font(13, True))
    draw.text(
        (pr - 8, pb + 10),
        "8 CONTROL POINTS · PLANAR HOMOGRAPHY · VEHICLE-SCALE FOOTPRINTS",
        fill="#31d6c4",
        font=font(13),
        anchor="ra",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
