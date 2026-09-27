from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.geometry import project_points, solve_homography  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    scene = json.loads((ROOT / "backend" / "data" / "scene.json").read_text(encoding="utf-8"))
    image_points = [tuple(point) for point in scene["calibration"]["image_points"]]
    ground_points = [tuple(point) for point in scene["calibration"]["ground_points"]]
    homography = solve_homography(image_points, ground_points)
    positions = project_points(homography, [tuple(item["anchor"]) for item in scene["detections"]])

    objects: list[dict] = []
    for item, (x, y) in zip(scene["detections"], positions, strict=True):
        objects.append(
            {
                "track_id": item["track_id"],
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "bbox_xywh_px": item["bbox"],
                "ground_anchor_uv_px": item["anchor"],
                "ground_position_xy_m": [round(x, 6), round(y, 6)],
                "speed_kmh": item["speed_kmh"],
                "heading_deg": item["heading_deg"],
            }
        )

    output = {
        "schema_version": "carom-rebuild.frame-result.v1",
        "scene_id": scene["scene_id"],
        "frame_index": 2901,
        "timestamp": scene["timestamp"],
        "source": scene["source"],
        "provenance": {
            "frame": "official CAROM demo thumbnail crop",
            "bbox_and_anchor": "manually initialized MVP annotations",
            "ground_position": "computed from anchor pixels with the recorded homography",
            "speed_and_heading": "MVP demonstration attributes; not measured from a temporal sequence",
        },
        "frame": scene["frame"],
        "coordinate_system": {
            "image": "origin top-left; u right; v down; unit pixel",
            "ground": "origin road-near center; x right; y forward; unit meter",
        },
        "calibration": {
            **scene["calibration"],
            "homography_image_to_ground": [[round(float(value), 12) for value in row] for row in homography],
        },
        "ground_extent_m": scene["ground_extent"],
        "objects": objects,
        "summary": {
            "vehicle_count": len(objects),
            "mean_speed_kmh": round(sum(item["speed_kmh"] for item in objects) / len(objects), 3),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "frame_2901_result.json"
    csv_path = args.output_dir / "frame_2901_vehicles.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = [
            "track_id",
            "class_name",
            "confidence",
            "bbox_x_px",
            "bbox_y_px",
            "bbox_w_px",
            "bbox_h_px",
            "anchor_u_px",
            "anchor_v_px",
            "ground_x_m",
            "ground_y_m",
            "speed_kmh",
            "heading_deg",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in objects:
            bbox = item["bbox_xywh_px"]
            anchor = item["ground_anchor_uv_px"]
            ground = item["ground_position_xy_m"]
            writer.writerow(
                {
                    "track_id": item["track_id"],
                    "class_name": item["class_name"],
                    "confidence": item["confidence"],
                    "bbox_x_px": bbox[0],
                    "bbox_y_px": bbox[1],
                    "bbox_w_px": bbox[2],
                    "bbox_h_px": bbox[3],
                    "anchor_u_px": anchor[0],
                    "anchor_v_px": anchor[1],
                    "ground_x_m": ground[0],
                    "ground_y_m": ground[1],
                    "speed_kmh": item["speed_kmh"],
                    "heading_deg": item["heading_deg"],
                }
            )

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
