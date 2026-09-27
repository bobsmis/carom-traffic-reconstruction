"""Evaluator-only image-space association diagnostic, NOT an official benchmark.

Never imports the algorithm or modifies its output. Matching uses 2D boxes,
not predicted metric positions. Occluded projected boxes are not visibility GT.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    intersection=np.maximum(hi-lo,0).prod()
    union=np.maximum(np.asarray(a[2:])-a[:2],0).prod()+np.maximum(np.asarray(b[2:])-b[:2],0).prod()-intersection
    return float(intersection/union) if union>0 else 0.


def associate(pred, truth, width, height, threshold=.3, margin=.1):
    boxes=[];valid=[]
    for obj in truth:
        if not obj.get('in_camera_frustum') or obj.get('projected_cuboid_px') is None: continue
        uv=np.asarray(obj['projected_cuboid_px'])
        box=np.r_[np.maximum(uv.min(axis=0),[0,0]),np.minimum(uv.max(axis=0),[width,height])]
        if min(box[2:]-box[:2])<12:continue
        valid.append(obj);boxes.append(box)
    if not pred or not valid:return [],len(valid)
    scores=np.array([[iou(p['bbox_xyxy'],b) for b in boxes] for p in pred])
    # Dummy assignments let unmatched detections remain unmatched.
    cost=np.full((len(pred),len(valid)+len(pred)),1-threshold)
    cost[:,:len(valid)]=1-scores
    rows,cols=linear_sum_assignment(cost)
    matched=[]
    for r,c in zip(rows,cols):
        if c>=len(valid) or scores[r,c]<threshold:continue
        alternatives=np.r_[np.delete(scores[r],c),np.delete(scores[:,c],r),0.]
        if scores[r,c]-alternatives.max()<margin:continue
        matched.append((pred[r],valid[c],float(scores[r,c])))
    return matched,len(valid)


def summary(values):
    if not values:return {'count':0,'median':None,'p95':None}
    return {'count':len(values),'median':float(np.median(values)),'p95':float(np.percentile(values,95))}


def evaluate(run, predictions, output):
    read=lambda p: [json.loads(s) for s in p.read_text().splitlines()]
    truth=read(run/'truth/actors.jsonl'); frames=read(run/'input/frames.jsonl'); pred=read(predictions)
    camera=json.loads((run/'input/calibration.json').read_text())
    if len(truth)!=len(frames) or len(pred)!=len(frames):raise ValueError('frame count mismatch')
    errors={k:[] for k in ('lateral_abs_m','longitudinal_abs_m','position_m','heading_abs_deg','relative_size_error')}
    matched_count=eligible_count=observed_count=localized_count=0
    previous={};switches=0;details=[]
    for index,(gt,meta,p) in enumerate(zip(truth,frames,pred)):
        if gt['carla_frame']!=meta['carla_frame'] or p['frame_index']!=index or abs(p['timestamp_s']-meta['video_time_s'])>1e-6:
            raise ValueError('frame alignment mismatch')
        observed=[o for o in p['objects'] if o['observed'] and 'bbox_xyxy' in o]
        matches,eligible=associate(observed,gt['objects'],camera['width'],camera['height'])
        eligible_count+=eligible;observed_count+=len(observed);matched_count+=len(matches)
        for obj,target,score in matches:
            actor=target['actor_id']; last=previous.get(actor)
            if last and last[0]==index-1 and last[1]!=obj['id']:switches+=1
            previous[actor]=(index,obj['id'])
            if obj.get('ground_xy') is None:continue
            localized_count+=1
            delta=np.asarray(obj['ground_xy'])-target['ground_center_road_xy']
            errors['lateral_abs_m'].append(abs(float(delta[0])))
            errors['longitudinal_abs_m'].append(abs(float(delta[1])))
            errors['position_m'].append(float(np.linalg.norm(delta)))
            angle=obj['heading_rad']-target['heading_rad']
            errors['heading_abs_deg'].append(float(abs(np.degrees(np.arctan2(np.sin(angle),np.cos(angle))))))
            errors['relative_size_error'].extend((abs(np.asarray(obj['dimensions_lwh'])/target['dimensions_lwh']-1)).tolist())
            details.append({'frame':index,'track_id':obj['id'],'actor_id':actor,'image_iou':score,
                            'error_xy_m':delta.tolist(),'geometry_status':obj['geometry_status']})
    report={'acceptance_passed':None,'evaluation_type':'image_box_association_proxy_only',
            'limitations':['Occlusion visibility and instance-to-actor identity are not validated.',
                          'Ambiguous image matches excluded and counts retained; selection bias remains.',
                          'Counts are object-frames, not unique vehicles. Not recall/MOTA/IDF1.',
                          'Metric errors are conditional on proxy matching, not certified accuracy.'],
            'association':{'iou_min':.3,'uniqueness_margin':.1,'projected_min_extent_px':12},
            'frames':len(pred),'eligible_projected_object_frames':eligible_count,
            'observed_confirmed_object_frames':observed_count,'proxy_matched_object_frames':matched_count,
            'matched_localized_object_frames':localized_count,
            'unmatched_observed_object_frames':observed_count-matched_count,
            'unmatched_projected_object_frames':eligible_count-matched_count,
            'consecutive_proxy_id_changes':switches,
            'errors':{k:summary(v) for k,v in errors.items()},
            'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
            'truth_sha256':hashlib.sha256((run/'truth/actors.jsonl').read_bytes()).hexdigest()}
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (output/'matches.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in details))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();evaluate(args.run,args.predictions,args.output)
