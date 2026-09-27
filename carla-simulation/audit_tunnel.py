"""Scenario verification, evaluator-only; not algorithm precision metrics."""
import argparse
import collections
import json
from pathlib import Path
import cv2

def audit(root):
    root=Path(root);meta=json.loads((root/'input/manifest.json').read_text())
    frames=[json.loads(s) for s in (root/'input/frames.jsonl').read_text().splitlines()]
    truth=[json.loads(s) for s in (root/'truth/actors.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (root/'truth/events.jsonl').read_text().splitlines()]
    cap=cv2.VideoCapture(str(root/'input/rgb.mp4'));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    previous={};transitions=[];ids=set()
    for row in truth:
        for obj in row['objects']:
            aid=obj['actor_id'];ids.add(aid);lane=obj['lane_id_by_actor_origin']
            old=previous.get(aid)
            if lane is not None:
                if old is not None and old!=lane:transitions.append({'actor_id':aid,'from':old,'to':lane,'frame':row['carla_frame']})
                previous[aid]=lane
    counts=collections.Counter(e['type'] for e in events)
    collision_pairs=sorted({tuple(sorted((e['actor_id'],e['other_actor_id']))) for e in events if e['type']=='collision_sensor'})
    checks={'frame_counts':decoded==len(frames)==len(truth)==meta['frames'],
            'sensor_sync':all(r['sensor_frames']==[r['carla_frame']]*2 for r in frames),
            'truth_sync':all(a['carla_frame']==b['carla_frame'] for a,b in zip(frames,truth)),
            'sequential_frames':all(b['carla_frame']==a['carla_frame']+1 for a,b in zip(frames,frames[1:]))}
    report={'integrity_passed':all(checks.values()),'checks':checks,'frames':decoded,
            'unique_simulated_vehicles':len(ids),'events':dict(counts),
            'lane_transitions':transitions,'collision_confirmed':counts['collision_sensor']>0,
            'unique_contact_pairs':collision_pairs,
            'note':'Scenario truth only. Lane transitions are actor-origin lane changes, not an algorithm score. Collision callbacks may contain repeated contact events.'}
    (root/'scenario-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return all(checks.values())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');raise SystemExit(0 if audit(p.parse_args().run) else 1)
