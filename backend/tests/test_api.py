from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "offline"


def test_scene_contains_projected_vehicles() -> None:
    response = client.get("/api/scene")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["vehicles"] == 5
    assert all("ground" in detection for detection in payload["detections"])
    assert payload["calibration"]["control_point_count"] >= 6
    assert payload["summary"]["road_lanes"] == 5
    assert all(len(detection["footprint_m"]) == 4 for detection in payload["detections"])
    assert all(len(detection["mask_polygon"]) >= 6 for detection in payload["detections"])


def test_project_rejects_out_of_frame_point() -> None:
    response = client.post("/api/project", json={"points": [{"u": 9000, "v": 20}]})
    assert response.status_code == 422


def test_pipeline_endpoints_are_declared() -> None:
    paths = {route.path for route in app.routes}
    assert "/api/pipeline/audit" in paths
    assert "/api/pipeline/routes" in paths
    assert "/api/pipeline/result-frame" in paths
    assert "/api/pipeline/video" in paths


def test_release_api_reads_frame_and_rejects_invalid_indices(tmp_path, monkeypatch) -> None:
    import json
    from app import main
    monkeypatch.setattr(main, "PIPELINE_OUTPUT", tmp_path)
    assert client.get('/api/pipeline/diagnostics').status_code == 503
    (tmp_path / 'diagnostics.json').write_text(json.dumps({'frames': 1}))
    (tmp_path / 'frames.jsonl').write_text(json.dumps({'frame_index': 0, 'detections': []}) + '\n')
    assert client.get('/api/pipeline/diagnostics').json()['frames'] == 1
    assert client.get('/api/pipeline/frames/0').json()['detections'] == []
    assert client.get('/api/pipeline/frames/1').status_code == 404
    assert client.get('/api/pipeline/frames/-1').status_code == 422
