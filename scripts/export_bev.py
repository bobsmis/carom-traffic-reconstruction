from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.geometry import project_points, solve_homography  # noqa: E402


WIDTH = 1200
HEIGHT = 1600
PADDING_X = 150
PADDING_Y = 150


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    scene_path = ROOT / "backend" / "data" / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    extent = scene["ground_extent"]
    image_points = [tuple(point) for point in scene["calibration"]["image_points"]]
    ground_points = [tuple(point) for point in scene["calibration"]["ground_points"]]
    homography = solve_homography(image_points, ground_points)
    anchors = [tuple(item["anchor"]) for item in scene["detections"]]
    positions = project_points(homography, anchors)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#10171d")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, WIDTH, 110), fill="#172129")
    draw.text((44, 28), "CAROM // PROCESSED WORLD PLANE", fill="#31d6c4", font=font(26, True))
    draw.text((44, 66), "Fixed monocular camera · planar homography · unit: meter", fill="#91a4ae", font=font(18))

    plot_left = PADDING_X
    plot_right = WIDTH - PADDING_X
    plot_top = PADDING_Y
    plot_bottom = HEIGHT - 130

    for meter in range(0, 73, 10):
        y = plot_bottom - (meter - extent["y_min"]) / (extent["y_max"] - extent["y_min"]) * (plot_bottom - plot_top)
        draw.line((80, y, WIDTH - 80, y), fill="#31404a", width=2)
        draw.text((85, y - 28), f"{meter} m", fill="#80919a", font=font(17))

    road_top_half = 230
    road_bottom_half = 410
    center = WIDTH / 2
    road = [
        (center - road_top_half, plot_top),
        (center + road_top_half, plot_top),
        (center + road_bottom_half, plot_bottom),
        (center - road_bottom_half, plot_bottom),
    ]
    draw.polygon(road, fill="#253038", outline="#65737b", width=4)
    draw.line((center, plot_top, center, plot_bottom), fill="#ded9b6", width=5)
    for y in range(plot_top, plot_bottom, 48):
        draw.line((center, y, center, min(y + 23, plot_bottom)), fill="#10171d", width=7)
    draw.line((center - 120, plot_top, center - 260, plot_bottom), fill="#879399", width=3)
    draw.line((center + 120, plot_top, center + 260, plot_bottom), fill="#879399", width=3)

    def map_point(x: float, y: float) -> tuple[float, float]:
        px = plot_left + (x - extent["x_min"]) / (extent["x_max"] - extent["x_min"]) * (plot_right - plot_left)
        py = plot_bottom - (y - extent["y_min"]) / (extent["y_max"] - extent["y_min"]) * (plot_bottom - plot_top)
        return px, py

    for item, (x, y) in zip(scene["detections"], positions, strict=True):
        px, py = map_point(x, y)
        color = item["color"]
        draw.rounded_rectangle((px - 36, py - 18, px + 36, py + 18), radius=7, fill=color, outline="#f2f6f4", width=3)
        label_x = px + 50 if x < 8 else px - 250
        draw.text((label_x, py - 39), f"TRACK {item['track_id']}", fill=color, font=font(21, True))
        draw.text((label_x, py - 10), f"X {x:6.2f} m  ·  Y {y:6.2f} m", fill="#e8ece8", font=font(19))
        draw.text((label_x, py + 18), f"{item['speed_kmh']:.1f} km/h", fill="#91a4ae", font=font(17))

    draw.line((WIDTH - 96, 170, WIDTH - 96, 250), fill="#ff9e3d", width=5)
    draw.polygon([(WIDTH - 96, 150), (WIDTH - 108, 175), (WIDTH - 84, 175)], fill="#ff9e3d")
    draw.text((WIDTH - 108, 118), "N", fill="#f0f4f2", font=font(25, True))
    draw.text((44, HEIGHT - 72), f"Frame #2901  |  Vehicles {len(scene['detections'])}  |  Reprojection error {scene['calibration']['reprojection_error_px']:.2f}px", fill="#91a4ae", font=font(19))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
