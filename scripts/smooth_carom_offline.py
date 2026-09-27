"""Replay a traceable RTS estimate; input evidence is immutable and hashed."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.pipeline.offline_smoothing import smooth_positions
from app.pipeline.carom_geometry import RoadCamera,cuboid_vertices
from app.pipeline.measurement_region import MeasurementRegion
from run_carom_method import EvidenceRenderer,semantic_lane


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segments(records):
    by_id=defaultdict(list)
    for frame in records:
        for row in frame['objects']:
            if row.get('ground_xy') is not None:
                by_id[row['id']].append((frame,row))
    for tid,items in by_id.items():
        chunk=[]
        for item in items:
            if chunk and (item[0]['frame_index']!=chunk[-1][0]['frame_index']+1 or
                          item[1].get('geometry_epoch',0)!=chunk[-1][1].get('geometry_epoch',0)):
                yield tid,chunk; chunk=[]
            chunk.append(item)
        if chunk:
            yield tid,chunk


def temporal_report(records):
    accelerations=[]; headings=[]; births=Counter(); seen=set()
    for frame in records:
        for r in frame['objects']:
            if r['id'] not in seen:
                births[str(frame['frame_index'])]+=1; seen.add(r['id'])
    for _,chunk in segments(records):
        if len(chunk)<3: continue
        t=np.array([f['timestamp_s'] for f,r in chunk])
        p=np.array([r['ground_xy'] for f,r in chunk])
        dt=np.diff(t)
        v=np.diff(p,axis=0)/dt[:,None]
        a=np.diff(v,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
        accelerations.extend(np.linalg.norm(a,axis=1).tolist())
        angles=np.unwrap([r['heading_rad'] for f,r in chunk])
        headings.extend(np.abs(np.diff(angles)).tolist())
    return {'position_second_difference_mps2_p95':float(np.percentile(accelerations,95)) if accelerations else None,
            'heading_step_deg_p95':float(np.rad2deg(np.percentile(headings,95))) if headings else None,
            'published_unique_ids':len(seen),'first_publication_by_frame':dict(births),
            'localized_object_frames':sum(f['localized_count'] for f in records),
            'overlap_pair_frames':sum(f['footprint_overlap_pairs'] for f in records),
            'meaning':'Smoothness proxies only, not real acceleration, position accuracy or IDF1.'}


def calibration_readiness(config):
    c=config['planar_calibration']; reasons=[]
    if 'intrinsic_matrix' not in c: reasons.append('measured_intrinsic_matrix_missing')
    if not c.get('surveyed'): reasons.append('same_scene_surveyed_world_coordinates_missing')
    if len(c.get('ground_points_m',[]))<6: reasons.append('fewer_than_six_control_pairs')
    if len(c.get('check_points',{}).get('ground_points_m',[]))<2: reasons.append('independent_holdout_pairs_missing')
    try:
        _,audit=RoadCamera.from_scene(config)
        passed=True
    except ValueError as error:
        passed=False; reasons.append(str(error)); _,audit=RoadCamera.from_scene(config,experimental=True)
    return {'calibration_consistency_passed':passed,'metric_accuracy_validated':False,
            'lane_truth_validated':False,'issues':reasons,'calibration':audit,
            'required_evidence':['K measured for this video resolution/crop; lens distortion corrected',
                                 'At least 6 surveyed image/world pairs spanning all lanes and near/far range',
                                 'At least 2 disjoint held-out check pairs; do not derive from fitted H',
                                 'Independently measured lane boundaries and per-frame vehicle ground truth'],
            'note':'A low reprojection residual is consistency, not absolute metric accuracy.'}


def process(source,output):
    if source.resolve()==output.resolve(): raise ValueError('output must differ from evidence source')
    if output.exists() and any(output.iterdir()): raise ValueError('use an empty output directory; preserve previous evidence')
    source_hash=digest(source/'frames.jsonl')
    implementation={str(p.relative_to(ROOT)):p.read_bytes() for p in
                    [Path(__file__),ROOT/'backend/app/pipeline/offline_smoothing.py']}
    records=[json.loads(s) for s in (source/'frames.jsonl').read_text().splitlines()]
    original=deepcopy(records)
    config=json.loads((source/'scene-config.json').read_text())
    audit=json.loads((source/'audit.json').read_text())
    camera,calibration=RoadCamera.from_scene(config,experimental=True)
    region=MeasurementRegion(camera,config)
    counts=Counter(); displacements=[]; segments_report=[]
    for tid,chunk in segments(records):
        start=next((i for i,(_,r) in enumerate(chunk) if r['geometry_status']=='measured'),None)
        if start is None: continue
        chunk=chunk[start:]
        observations=[r['fit_quality']['ground_xy'] if r['geometry_status']=='measured' else None for _,r in chunk]
        if sum(z is not None for z in observations)<3:
            counts['insufficient_evidence_segments']+=1; continue
        times=[f['timestamp_s'] for f,r in chunk]
        variances=[.25+(max(0,z[1])/80)**2 if z is not None else 1. for z in observations]
        xs,ps,accepted=smooth_positions(times,observations,variances)
        # Reject whole segment rather than clip/snap a smooth estimate to ROI.
        if not all(region.ground_check(x[:2])['eligible'] for x in xs):
            counts['roi_rejected_segments']+=1; continue
        for (f,r),x,p,used in zip(chunk,xs,ps,accepted):
            before=np.asarray(r['ground_xy'])
            heading=r['heading_rad']
            speed=float(np.linalg.norm(x[2:]))
            # Body direction inferred from temporally consistent motion only
            # when velocity dominates uncertainty; retain evidence at rest.
            direction_reliable=speed>max(1.,2*np.sqrt(np.trace(p[2:,2:])))
            if direction_reliable:
                heading=math.atan2(x[3],x[2])
            displacement=float(np.linalg.norm(x[:2]-before)); displacements.append(displacement)
            r['online_estimate']={k:deepcopy(r.get(k)) for k in ('ground_xy','velocity_xy_mps','heading_rad','covariance_xy','lane','geometry_status','reason')}
            source_status=r['geometry_status']
            measurement_status='accepted' if used else 'rejected' if source_status=='measured' else 'missing'
            r['geometry_status']='smoothed'
            r['measurement_status']=measurement_status
            r['reason']='offline_'+measurement_status+'_measurement'
            r['offline_estimate']={'method':'RTS_position_evidence','measurement_used':bool(used),
                                   'motion_heading_reliable':bool(direction_reliable),
                                   'online_displacement_m':displacement,'uses_future_frames':True,
                                   'metric_accuracy_validated':False}
            r.update(ground_xy=x[:2].tolist(),velocity_xy_mps=x[2:].tolist(),covariance_xy=p[:2,:2].tolist(),
                     heading_rad=heading,lane=semantic_lane(*x[:2],config))
            try: r['cuboid_image']=camera.project(cuboid_vertices(x[:2],heading,r['dimensions_lwh'])).tolist()
            except ValueError: r['cuboid_image']=None
            counts['smoothed_object_frames']+=1
        counts['smoothed_segments']+=1
        segments_report.append({'id':tid,'first_frame':chunk[0][0]['frame_index'],'last_frame':chunk[-1][0]['frame_index'],
                                'geometry_epoch':chunk[0][1].get('geometry_epoch',0),
                                'position_observations':sum(z is not None for z in observations),
                                'innovation_rejections':sum(z is not None and not used for z,used in zip(observations,accepted))})
    for f in records:
        localized=[r for r in f['objects'] if r.get('ground_xy') is not None]; overlap=0
        for i,a in enumerate(localized):
            for b in localized[i+1:]:
                pa=cuboid_vertices(a['ground_xy'],a['heading_rad'],a['dimensions_lwh'])[:4,:2].astype(np.float32)
                pb=cuboid_vertices(b['ground_xy'],b['heading_rad'],b['dimensions_lwh'])[:4,:2].astype(np.float32)
                area,_=cv2.intersectConvexConvex(pa,pb)
                overlap+=int(area>.1)
        f['footprint_overlap_pairs']=overlap
    # Identity, visibility and observations must never be altered to improve metrics.
    assert [[r['id'] for r in f['objects']] for f in records]==[[r['id'] for r in f['objects']] for f in original]
    output.mkdir(parents=True,exist_ok=True)
    report={'before':temporal_report(original),'after':temporal_report(records),'counts':dict(counts),
            'segments':segments_report,'evidence_frames_sha256':source_hash,
            'same_ids_same_frame_membership':True,'new_detections_synthesized':0,'lane_snapping':False,
            'collision_relocation':False,'source':str(source.resolve()),
            'online_displacement_m_p95':float(np.percentile(displacements,95)) if displacements else None,
            'accuracy_validated':False,'paper_exact_reproduction':False}
    (output/'smoothing-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (output/'calibration-readiness.json').write_text(json.dumps(calibration_readiness(config),indent=2))
    (output/'scene-config.json').write_text(json.dumps(config,indent=2))
    with (output/'frames.jsonl').open('w') as handle:
        for r in records: handle.write(json.dumps(r,allow_nan=False)+'\n')
    audit['offline_smoothing']={'input_path':str(source.resolve()),'input_sha256':source_hash,
                                'process_noise_spectral_density':2.25,'method':'constant_velocity_KF_RTS',
                                'position_evidence':'accepted silhouette support-plane position; not filtered position',
                                'velocity_evidence':'inferred jointly from positions; flow velocity not treated as independent'}
    for relative,content in implementation.items():
        audit['implementation_sha256'][relative]=hashlib.sha256(content).hexdigest()
        snapshot=output/'implementation'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
    audit['counts']['footprint_overlap_pairs']=report['after']['overlap_pair_frames']
    audit['source_geometry_state_counts']={k:audit['counts'].pop(k,0) for k in ('measured','predicted','pending')}
    audit['counts'].update(Counter(r['geometry_status'] for f in records for r in f['objects']))
    audit['limits'].append('Offline RTS uses future frames, changes posterior poses, and is not a calibration/accuracy repair.')
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    renderer=EvidenceRenderer(camera,config,ROOT/config['video_path'])
    cap=cv2.VideoCapture(str(ROOT/config['video_path'])); fps=cap.get(cv2.CAP_PROP_FPS)
    writer=cv2.VideoWriter(str(output/'method-preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
    if not writer.isOpened(): raise RuntimeError('video writer failed')
    try:
        for f in records:
            ok,frame=cap.read()
            if not ok: raise ValueError('video shorter than evidence')
            c=config['crop']; frame=frame[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']]
            rendered=renderer.render(frame,f,calibration)
            cv2.putText(rendered,'OFFLINE RTS | future-frame estimate',(1250,612),0,.5,(90,245,160),1,cv2.LINE_AA)
            writer.write(rendered)
            if f['frame_index']==audit['preview_frame']:
                cv2.imwrite(str(output/'validation.png'),rendered)
    finally:
        cap.release(); writer.release()
    assert digest(source/'frames.jsonl')==source_hash
    print(json.dumps({k:v for k,v in report.items() if k!='segments'},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args(); process(args.source,args.output)
