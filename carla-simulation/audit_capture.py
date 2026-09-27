"""Validate capture integrity, not reconstruction accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def audit(root):
    root = Path(root)
    read = lambda p: json.loads((root / p).read_text(encoding='utf-8'))
    rows = lambda p: [json.loads(s) for s in (root / p).read_text(encoding='utf-8').splitlines()]
    manifest = read('input/manifest.json')
    frames, truth = rows('input/frames.jsonl'), rows('truth/actors.jsonl')
    n, fps = manifest['frames'], manifest['fps']
    checks = {}
    checks['record_counts'] = len(frames) == len(truth) == n
    checks['mask_count'] = len(list((root / 'truth/masks').glob('*.png'))) == n
    checks['sensor_alignment'] = all(a['index'] == b['index'] == i and
        a['carla_frame'] == b['carla_frame'] and
        all(x == a['carla_frame'] for x in a['sensor_frames'])
        for i, (a, b) in enumerate(zip(frames, truth)))
    checks['frame_continuity'] = all(b['carla_frame'] - a['carla_frame'] == 1
        for a, b in zip(frames, frames[1:]))
    checks['timestamp_step'] = bool(np.allclose(np.diff([r['timestamp_s'] for r in frames]), 1/fps, atol=1e-5))
    checks['rgb_hash'] = hashlib.sha256((root / 'input/rgb.mp4').read_bytes()).hexdigest() == manifest['sha256']
    cap = cv2.VideoCapture(str(root / 'input/rgb.mp4'))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    checks['decoded_frames'] = decoded == n
    ids = {o['actor_id'] for o in truth[0]['objects']}
    checks['stable_actor_inventory'] = len(ids) == 24 and all({o['actor_id'] for o in r['objects']} == ids for r in truth)
    lane_counts = {}
    for o in truth[0]['objects']:
        lane = o['lane_id_by_actor_origin']
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
    checks['eight_lanes_three_each'] = len(lane_counts) == 8 and set(lane_counts.values()) == {3}
    cal = read('input/calibration.json')
    e = np.asarray(cal['world_to_camera_cv'])
    checks['proper_rotation'] = bool(np.allclose(e[:3,:3] @ e[:3,:3].T, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(e[:3,:3]), 1, atol=1e-5))
    report = {'passed': all(checks.values()), 'checks': checks, 'frames': n, 'fps': fps,
              'initial_lane_counts': lane_counts, 'actors': len(ids),
              'scope': 'Capture integrity only; no detector recall, ID or metric-accuracy claim.',
              'occlusion_warning': 'Actor frustum flags are not visibility labels. Instance-to-actor identity mapping requires separate validation.'}
    (root / 'capture-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report['passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run')
    raise SystemExit(0 if audit(p.parse_args().run) else 1)
