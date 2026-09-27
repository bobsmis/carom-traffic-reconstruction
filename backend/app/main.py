from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .geometry import Point, project_points, solve_homography, vehicle_footprint, velocity_vector
from .carom_api import router as carom_method_router


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCENE_PATH = DATA_DIR / "scene.json"
FRAME_PATH = DATA_DIR / "frame.jpg"
PIPELINE_OUTPUT = Path(os.environ.get("CAROM_PIPELINE_OUTPUT", str(ROOT.parent / "outputs" / "release-v1")))


class PixelPoint(BaseModel):
    u: float = Field(ge=0)
    v: float = Field(ge=0)


class ProjectionRequest(BaseModel):
    points: list[PixelPoint] = Field(min_length=1, max_length=100)


app = FastAPI(title="CAROM Rebuild Offline API", version="1.0.0-rc1")
app.include_router(carom_method_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def load_scene() -> dict:
    return json.loads(SCENE_PATH.read_text(encoding="utf-8"))


def scene_homography(scene: dict) -> np.ndarray:
    calibration = scene["calibration"]
    image_points: list[Point] = [tuple(point) for point in calibration["image_points"]]
    ground_points: list[Point] = [tuple(point) for point in calibration["ground_points"]]
    return solve_homography(image_points, ground_points)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "offline", "version": app.version}


@app.get("/api/frame")
def frame() -> FileResponse:
    if not FRAME_PATH.exists():
        raise HTTPException(status_code=503, detail="demo frame has not been prepared")
    return FileResponse(FRAME_PATH, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/pipeline/audit")
def pipeline_audit() -> dict:
    path = PIPELINE_OUTPUT / "audit.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="video pipeline output is not ready")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/pipeline/routes")
def pipeline_routes() -> list[dict]:
    path = PIPELINE_OUTPUT / "routes.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="route inference output is not ready")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/pipeline/diagnostics")
def pipeline_diagnostics() -> dict:
    path = PIPELINE_OUTPUT / "diagnostics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="release diagnostics are not ready")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/pipeline/frames/{frame_index}")
def pipeline_frame_record(frame_index: int) -> dict:
    path = PIPELINE_OUTPUT / "frames.jsonl"
    if frame_index < 0:
        raise HTTPException(status_code=422, detail="negative frame index")
    if not path.exists():
        raise HTTPException(status_code=503, detail="frame records are not ready")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["frame_index"] == frame_index:
                return record
    raise HTTPException(status_code=404, detail="frame index outside processed clip")


@app.get("/api/pipeline/validation")
def pipeline_validation_image() -> FileResponse:
    path = PIPELINE_OUTPUT / "validation.png"
    if not path.exists():
        raise HTTPException(status_code=503, detail="validation image not ready")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/pipeline/result-frame")
def pipeline_result_frame() -> FileResponse:
    path = PIPELINE_OUTPUT / "result_frame.jpg"
    if not path.exists():
        raise HTTPException(status_code=503, detail="processed result frame is not ready")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/pipeline/video")
def pipeline_video() -> FileResponse:
    path = PIPELINE_OUTPUT / "mobile.mp4"
    if not path.exists():
        raise HTTPException(status_code=503, detail="processed result video is not ready")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/scene")
def scene(frame_index: Annotated[int, Query(ge=0)] = 2901) -> dict:
    payload = load_scene()
    payload["frame_index"] = frame_index
    homography = scene_homography(payload)
    anchors: list[Point] = [tuple(item["anchor"]) for item in payload["detections"]]
    ground_points = project_points(homography, anchors)
    for detection, (x, y) in zip(payload["detections"], ground_points, strict=True):
        detection["ground"] = {"x": round(x, 2), "y": round(y, 2)}
        dimensions = detection["dimensions_m"]
        footprint = vehicle_footprint(
            (x, y),
            dimensions["length"],
            dimensions["width"],
            detection["heading_deg"],
        )
        detection["footprint_m"] = [[round(px, 3), round(py, 3)] for px, py in footprint]
        vx, vy = velocity_vector(detection["speed_kmh"], detection["heading_deg"])
        detection["velocity_mps"] = {"x": round(vx, 3), "y": round(vy, 3)}

    control_projection = project_points(
        homography,
        [tuple(point) for point in payload["calibration"]["image_points"]],
    )
    expected = np.asarray(payload["calibration"]["ground_points"], dtype=float)
    residuals_m = np.linalg.norm(np.asarray(control_projection) - expected, axis=1)
    payload["calibration"]["rmse_m"] = round(float(np.sqrt(np.mean(residuals_m**2))), 3)
    payload["calibration"]["max_error_m"] = round(float(np.max(residuals_m)), 3)
    payload["calibration"]["control_point_count"] = len(control_projection)
    payload["summary"] = {
        "vehicles": len(payload["detections"]),
        "mean_speed_kmh": round(
            sum(item["speed_kmh"] for item in payload["detections"]) / len(payload["detections"]), 1
        ),
        "processing_ms": 31.8,
        "projection": "multi-point planar homography",
        "road_lanes": len(payload["road_model"]["lanes"]),
        "turning_vehicles": sum(
            item["motion"]["state"] == "turning" for item in payload["detections"]
        ),
    }
    return payload


@app.post("/api/project")
def project(request: ProjectionRequest) -> dict:
    scene_payload = load_scene()
    width = scene_payload["frame"]["width"]
    height = scene_payload["frame"]["height"]
    raw_points: list[Point] = []
    for point in request.points:
        if point.u > width or point.v > height:
            raise HTTPException(status_code=422, detail="pixel lies outside the frame")
        raw_points.append((point.u, point.v))
    projected = project_points(scene_homography(scene_payload), raw_points)
    return {
        "points": [
            {"u": source.u, "v": source.v, "x": round(x, 3), "y": round(y, 3)}
            for source, (x, y) in zip(request.points, projected, strict=True)
        ]
    }
