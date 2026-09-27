"""Reproducible offline mapping release from saved local CV observations.

No detector results are synthesized. Use run_video_pipeline.py for fresh inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline.processor import VideoRoadProcessor
from app.pipeline.routes import InferredRoute


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/user_bridge_traffic.json")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    src, out = args.input.resolve(), args.output.resolve()
    if src == out:
        raise ValueError("Use a separate release output to preserve source records")
    out.mkdir(parents=True, exist_ok=True)
    processor = VideoRoadProcessor(ROOT, args.config.resolve(), out)
    processor.validate_inputs()
    source_audit = json.loads((src / 'audit.json').read_text())
    if source_audit['source_video_sha256'] != processor.config['source_sha256']:
        raise ValueError('Cached observations belong to a different source video')
    if source_audit['weights_sha256'] != digest(processor.weights_path):
        raise ValueError('Cached observations use different detector weights')
    records = [json.loads(line) for line in (src / "frames.jsonl").open()]
    before = [[(d['track_id'], d['ground'].copy(), d['bbox_xyxy'][:]) for d in r['detections']] for r in records]
    processor.road_map.assign_map_lanes(records)
    after = [[(d['track_id'], d['ground'], d['bbox_xyxy']) for d in r['detections']] for r in records]
    assert before == after, "Mapping must preserve raw IDs, measurements and detections"
    processor.frame_records = records
    tracks, occupancy = defaultdict(list), Counter()
    lane_status = Counter()
    overlaps = hidden = missing = outside = conflict_frames = 0
    corrections = []
    lane_changes = 0
    for record in records:
        conflict_frames += any(d.get('map_overlap_with') for d in record['detections'])
        for detection in record['detections']:
            tid = detection['track_id']
            lane = detection.get('map_lane_id')
            missing += lane is None
            hidden += not detection.get('map_visible', True)
            occupancy[lane] += 1
            lane_status[detection.get('map_lane_status', 'not_applicable')] += 1
            overlaps += len(detection.get('map_overlap_with', []))
            if lane:
                point = processor.road_map.detection_position_display(detection, 1000, 500)
                outside += not (0 <= point[0] < 1000 and 0 <= point[1] < 500)
                tracks[tid].append(detection)
                corrections.append(abs(detection['map_smoothing']['correction_px']) / 10.0)
    reversals = 0
    for values in tracks.values():
        for a, b in zip(values, values[1:]):
            if a['map_lane_id'] != b['map_lane_id']:
                lane_changes += 1
                continue
            reversals += a['map_lane_direction'] * (b['map_lane_progress_px'] - a['map_lane_progress_px']) < -0.002
    diagnostics = {
        'version': '1.0', 'frames': len(records),
        'lane_observations': dict(sorted(occupancy.items())),
        'lanes_with_vehicles': len(occupancy), 'lane_transitions': lane_changes,
        'lane_status_observations': dict(lane_status),
        'missing_map_objects': missing, 'hidden_map_objects': hidden,
        'reverse_steps_in_modeled_lane_progress': int(reversals),
        'overlapping_footprint_pairs': overlaps // 2,
        'frames_with_footprint_conflicts': conflict_frames,
        'projected_centers_outside_viewport': int(outside),
        'raw_measurements_preserved': before == after,
        'longitudinal_model_correction_p95_m': float(np.percentile(corrections, 95)),
        'metric_accuracy': 'unvalidated: manual correspondences and nominal width scale',
        'overlap_policy': 'report conflict; preserve location, ID and visibility',
        'direction_policy': 'configured direction prior; not proof of measured direction',
        'dimension_policy': 'bounded segmentation contact-width estimate + class prior',
        'release_status': 'candidate; external accuracy and occlusion identity require validation',
    }
    (out / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    with (out / 'frames.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    shutil.copy2(src / 'routes.json', out / 'routes.json')
    old_audit = json.loads((src / 'audit.json').read_text())
    old_audit.update({
        'version': '1.0-candidate', 'config_sha256': digest(args.config),
        'source_inference_config_sha256': source_audit['config_sha256'],
        'source_observation_file': str(src / 'frames.jsonl'),
        'source_observation_sha256': digest(src / 'frames.jsonl'),
        'inference_reused': True,
        'runtime_seconds_note': 'original inference run; excludes release remapping',
        'trajectory_constraint': 'geometric lane association with hysteresis; extrapolated progress; direction prior; conflicts retained',
        'panel_cardinality_mismatches': missing + hidden,
        'map_motion_smoothing': 'direction-constrained lane progress with raw correction recorded',
        'road_map_note': 'illustrative registered road texture, not a surveyed aerial map of this scene',
        'llm_runtime_calls': 0,
    })
    (out / 'audit.json').write_text(json.dumps(old_audit, indent=2))
    shutil.copy2(args.config, out / 'scene-config.json')
    print(json.dumps(diagnostics, indent=2), flush=True)
    if args.render:
        routes = [InferredRoute(x['route_id'], x['centerline_m'], x['support_tracks'], x['width_m']) for x in json.loads((out / 'routes.json').read_text())]
        capture = cv2.VideoCapture(str(processor.video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        processor._render(routes, fps)
        capture = cv2.VideoCapture(str(out / 'three_stage.mp4'))
        writer = cv2.VideoWriter(str(out / 'mobile.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), processor.config['processing']['output_fps'], (1280, 720))
        if not writer.isOpened():
            raise RuntimeError('Video writer failed')
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Preserve the 16:9 aspect ratio of the four-panel output.
            writer.write(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA))
            count += 1
        writer.release()
        capture.release()
        assert count == len(records), (count, len(records))
        import imageio_ffmpeg
        browser_video = out / 'mobile-browser.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(out / 'mobile.mp4'), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '21', '-movflags', '+faststart', str(browser_video)], check=True, capture_output=True)
        browser_video.replace(out / 'mobile.mp4')


if __name__ == '__main__':
    main()
