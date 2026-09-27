"""Independent CAROM-method experiment. Legacy inferred states are never read.

Default: local YOLO inference on source frames. --detection-cache reuses ONLY
class/confidence/bbox/mask observations; cache predictions and old IDs, metric
coordinates, headings, dimensions, lane assignments are explicitly discarded.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import os
import time

# Adopt a separately verified GPU runtime only at a camera/process boundary.
# Already running workers and the CPU environment are left untouched.
_root=Path(__file__).resolve().parents[1]
_gpu=_root/'.venv-cuda/bin/python'
if (__name__=='__main__' and os.environ.get('CAROM_RUNTIME','auto')=='auto'
        and (_root/'.venv-cuda/VERIFIED.json').exists()
        and Path(sys.prefix).resolve()!=(_root/'.venv-cuda').resolve()):
    os.execv(str(_gpu),[str(_gpu),str(Path(__file__).resolve()),*sys.argv[1:]])

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.pipeline.carom_geometry import RoadCamera, flow_heading, fit_cuboid, flow_velocity, cuboid_vertices
from app.pipeline.carom_tracking import FlowMaskTracker, MetricState, deduplicate_observations
from app.pipeline.measurement_region import MeasurementRegion
from app.pipeline.geometry_visibility import GeometryVisibility
from app.pipeline.vehicle_size import VehicleSize
from app.pipeline.static_lanes import lane_membership

PRIORS={'car':[4.55,1.82,1.48],'bus':[12.,2.55,3.25],'truck':[8.2,2.5,3.2], 'motorcycle':[2.2,.8,1.35]}
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def raw_observations(record):
    keys=('class_name','confidence','bbox_xyxy','mask_polygon')
    return [{**{k:d[k] for k in keys},'detector_crop_truncated':d.get('detector_crop_truncated',False)} for d in record['detections']
            if not d.get('prediction_only',False) and len(d.get('mask_polygon',[]))>=3]


def is_occluded(d, others, shape):
    x1,y1,x2,y2=d['bbox_xyxy']; h,w=shape[:2]
    poly=np.asarray(d['mask_polygon'])
    if (x1<=2 or y1<=2 or x2>=w-3 or y2>=h-3
            or poly[:,0].min()<=1 or poly[:,0].max()>=w-2):
        return 'image_boundary_truncation'
    area=max((x2-x1)*(y2-y1),1)
    for other in others:
        if other is d:
            continue
        a,b,c,e=other['bbox_xyxy']
        intersection=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,e)-max(y1,b))
        # Conservative depth ordering heuristic, explicitly not oracle occlusion.
        if e>y2 and intersection/area>.18:
            return 'foreground_overlap_gate'
    return None


def semantic_lane(x,y,config):
    """Annotation only: never moves a position or determines its heading."""
    if config['road_map'].get('lanes'):
        return lane_membership(x,y,config['road_map']['lanes'],config['road_map']['longitudinal_bounds_m'])
    width=config['road_map']['lane_width_m']
    count=config['road_map']['visible_lane_count']
    edge=-count*width/2
    lane=math.floor((x-edge)/width)
    if not 0<=lane<count:
        return {'lane_id':None,'status':'outside_nominal_lane_model'}
    center=edge+(lane+.5)*width
    return {'lane_id':f'L{lane+1:02}', 'lateral_offset_m':float(x-center),
            'status':'nominal_geometry_unverified', 'position_was_snapped':False}


def color(tid):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[tid*37%180,190,255]]]),cv2.COLOR_HSV2BGR)[0,0])


class EvidenceRenderer:
    """Metric-uniform BEV on rectified source background, no foreign road map."""
    def __init__(self,camera,config,video):
        self.camera=camera
        self.region=MeasurementRegion(camera,config)
        # world y points left; world x points down. Both use exactly 10 px/m.
        self.T=np.array([[0.,-10.,1480.],[10.,0.,220.],[0,0,1.]])
        self.M=self.T@np.linalg.inv(camera.H)
        cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames=[]
        for index in np.linspace(0,max(0,n-1),15,dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index)); ok,f=cap.read()
            if ok:
                crop=config['crop']; f=f[crop['y']:crop['y']+crop['height'],crop['x']:crop['x']+crop['width']]
                frames.append(cv2.warpPerspective(f,self.M,(1540,440)))
        cap.release()
        self.background=np.median(frames,axis=0).astype(np.uint8) if frames else np.zeros((440,1540,3),np.uint8)
        crop=config['crop']; valid=cv2.warpPerspective(np.full((crop['height'],crop['width']),255,np.uint8),self.M,(1540,440))
        self.background[valid==0]=(22,24,26)

    def point(self,xy):
        return tuple(np.round((self.T@np.r_[xy,1.])[:2]).astype(int))

    def render(self,frame,record,audit):
        canvas=np.full((1080,1920,3),(15,18,22),np.uint8)
        raw=cv2.resize(frame,(960,540)); overlay=frame.copy()
        if self.region.enabled:
            roi=np.round(self.region.image_polygon()).astype(np.int32)
            mask=np.zeros(frame.shape[:2],np.uint8); cv2.fillConvexPoly(mask,roi,255)
            overlay[mask==0]=(overlay[mask==0]*.38).astype(np.uint8)
            cv2.polylines(overlay,[roi],True,(90,245,160),3,cv2.LINE_AA)
            cv2.putText(overlay,'MEASUREMENT ROI | outside excluded',(24,frame.shape[0]-24),0,.8,(90,245,160),2)
        for row in record['objects']:
            c=color(row['id']); p=np.asarray(row['mask_polygon'],np.int32)
            if row['observed'] and len(p)>=3:
                paint=overlay.copy(); cv2.fillPoly(paint,[p],c); overlay=cv2.addWeighted(paint,.28,overlay,.72,0)
            if len(p)>=3:
                cv2.polylines(overlay,[p],True,c,2)
                anchor=p.min(axis=0)
                tag=f"#{row['id']} {row['geometry_status']}" + ('/coast' if not row['observed'] else '')
                cv2.putText(overlay,tag,tuple(anchor),(cv2.FONT_HERSHEY_SIMPLEX),.55,c,2,cv2.LINE_AA)
            if row.get('cuboid_image'):
                corners=np.round(row['cuboid_image']).astype(int)
                for a,b in EDGES:
                    cv2.line(overlay,tuple(corners[a]),tuple(corners[b]),c,2,cv2.LINE_AA)
        canvas[44:584,:960]=raw; canvas[44:584,960:]=cv2.resize(overlay,(960,540))
        cv2.putText(canvas,'1  ORIGINAL | untouched source pixels',(20,30),0,.75,(230,230,230),2)
        cv2.putText(canvas,'2  MASK / GLOBAL ID / GEOMETRY',(980,30),0,.75,(230,230,230),2)
        bev=self.background.copy()
        if self.region.enabled:
            roi=np.array([self.point(p) for p in self.region.polygon],np.int32)
            mask=np.zeros(bev.shape[:2],np.uint8); cv2.fillConvexPoly(mask,roi,255)
            bev[mask==0]=(bev[mask==0]*.28).astype(np.uint8)
            cv2.polylines(bev,[roi],True,(90,245,160),2,cv2.LINE_AA)
        # Thin nominal-lane guides; geometry labels explicitly separate from evidence.
        for x in np.linspace(-14,14,9):
            cv2.line(bev,self.point([x,0]),self.point([x,140]),(130,130,130),1,cv2.LINE_AA)
        for y in (0,25,50,75,100,125):
            cv2.putText(bev,f'{y}m',self.point([-18,y]),0,.4,(230,230,230),1)
        pending=[]; outside=[]
        for row in record['objects']:
            if row.get('ground_xy') is None:
                pending.append(row['id']); continue
            xy=row['ground_xy']; center=self.point(xy)
            if not (0<=center[0]<1540 and 0<=center[1]<440):
                outside.append(row['id']); continue
            c=color(row['id']); theta=row['heading_rad']; dims=row['dimensions_lwh']
            corners=cuboid_vertices(xy,theta,dims)[:4,:2]
            pts=np.array([self.point(p) for p in corners],np.int32)
            paint=bev.copy(); cv2.fillConvexPoly(paint,pts,c)
            bev=cv2.addWeighted(paint,.55,bev,.45,0)
            cv2.polylines(bev,[pts],True,c,1)
            front=np.array(xy)+np.array([math.cos(theta),math.sin(theta)])*dims[0]/2
            cv2.arrowedLine(bev,center,self.point(front),(255,255,255),1,tipLength=.4)
            cv2.putText(bev,str(row['id']),(center[0]+4,center[1]-4),0,.4,c,1,cv2.LINE_AA)
        canvas[630:1070,0:1540]=bev
        cv2.putText(canvas,'3  RECTIFIED SOURCE / METRIC FOOTPRINTS  |  NOMINAL CALIBRATION - NOT ACCURACY VALIDATED',(20,618),0,.6,(70,200,245),1)
        lines=[f"FRAME {record['frame_index']}",f"Observed: {record['observed_count']}",
               f"Coasting: {record['coasting_count']}",f"Localized: {record['localized_count']}",
               f"Pending: {len(pending)}",f"Tentative: {len(record.get('tentative_objects',[]))}",f"Outside view: {len(outside)}",
               'UNLOCATED IDS:']
        lines += [','.join(map(str,pending[i:i+9])) for i in range(0,len(pending),9)]
        if self.region.enabled:
            lines += [f"Excluded candidates: {len(record['rejected_candidates'])}",
                      f"Suppressed tracks: {len(record['excluded_objects'])}",
                      f"ROI y: {self.region.polygon[:,1].min():g}..{self.region.polygon[:,1].max():g}m (nominal)"]
        rmse=audit['control_reprojection_rmse_px']
        lines += ['No lane snapping','No direction clamp','No collision relocation',
                  f'Calib RMSE: {rmse:.1f}px' if rmse is not None else 'Exact simulator camera']
        for i,line in enumerate(lines[:20]):
            cv2.putText(canvas,line,(1555,650+i*21),0,.49,(225,225,225),1,cv2.LINE_AA)
        return canvas


def process(config_path,output,max_frames,experimental,cache_path=None,make_video=False):
    if max_frames<1:
        raise ValueError('max_frames must be positive')
    config=json.loads(config_path.read_text()); video=ROOT/config['video_path']
    camera,calibration=RoadCamera.from_scene(config,experimental=experimental)
    region=MeasurementRegion(camera,config)
    geometry_policy=GeometryVisibility(**config.get('geometry_prediction',{}))
    digest=hashlib.sha256(video.read_bytes()).hexdigest()
    if config.get('source_sha256') and digest!=config['source_sha256']:
        raise ValueError('source video checksum mismatch')
    output.mkdir(parents=True,exist_ok=True)
    implementation_paths=[Path(__file__),ROOT/'backend/app/pipeline/carom_geometry.py',
                          ROOT/'backend/app/pipeline/carom_tracking.py',ROOT/'backend/app/pipeline/measurement_region.py',
                          ROOT/'backend/app/pipeline/vehicle_size.py',ROOT/'backend/app/pipeline/geometry_visibility.py',
                          ROOT/'backend/app/pipeline/geometry_recovery.py']
    implementation_hashes={}
    for path in implementation_paths:
        relative=path.relative_to(ROOT); content=path.read_bytes()
        snapshot=output/'implementation'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True); snapshot.write_bytes(content)
        implementation_hashes[str(relative)]=hashlib.sha256(content).hexdigest()
    started=time.perf_counter();detector_seconds=0.;device_name='cache'
    records_cache={}
    if cache_path:
        records_cache={r['frame_index']:r for r in map(json.loads,cache_path.read_text().splitlines())}
        detector=None
    else:
        from ultralytics import YOLO
        import torch
        device_name='cuda:0' if torch.cuda.is_available() else 'cpu'
        detector=YOLO(str(ROOT/config['detector']['weights']))
        print(json.dumps({'event':'runtime','device':device_name,'torch':torch.__version__,'python':sys.executable}),flush=True)
    cap=cv2.VideoCapture(str(video)); fps=float(cap.get(cv2.CAP_PROP_FPS)); dt=1/fps
    tracker=FlowMaskTracker(config['processing']['global_tracking']['max_lost_frames'],
                            min_hits=config['processing']['global_tracking'].get('min_confirmation_hits',3))
    renderer=EvidenceRenderer(camera,config,video)
    writer=None
    if make_video:
        writer=cv2.VideoWriter(str(output/'method-preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
        if not writer.isOpened():
            raise RuntimeError('video writer failed')
    stats=Counter(); previous=None; best_score=-1; preview=None; max_lost=tracker.max_lost
    records=[]; detection_records=[]
    try:
        for index in range(max_frames):
            ok,frame=cap.read()
            if not ok: break
            crop=config['crop']; frame=frame[crop['y']:crop['y']+crop['height'],crop['x']:crop['x']+crop['width']]
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if cache_path:
                if index not in records_cache:
                    raise ValueError(f'detection cache missing source frame {index}')
                detections=raw_observations(records_cache[index])
            else:
                dcfg=config['detector']
                rx1,ry1,rx2,ry2=region.detector_crop(frame.shape)
                detector_frame=frame[ry1:ry2,rx1:rx2]
                detector_start=time.perf_counter()
                pred=detector.predict(detector_frame,imgsz=dcfg['image_size'],conf=dcfg['confidence'],classes=dcfg['classes'],device=device_name,half=False,verbose=False)[0]
                if device_name.startswith('cuda'):torch.cuda.synchronize()
                detector_seconds+=time.perf_counter()-detector_start
                detections=[]
                if pred.masks is not None:
                    # Transfer all boxes once rather than synchronizing the GPU
                    # separately for coordinates, class and confidence of every car.
                    box_rows=pred.boxes.data.cpu().numpy()
                    for box,mask in zip(box_rows,pred.masks.xy):
                        bx1,by1,bx2,by2=box[:4]
                        cut=((rx1>0 and bx1<2) or (ry1>0 and by1<2) or
                             (rx2<frame.shape[1] and bx2>rx2-rx1-2) or
                             (ry2<frame.shape[0] and by2>ry2-ry1-2))
                        detections.append({'class_name':pred.names[int(box[5])], 'confidence':float(box[4]),
                                           'detector_crop_truncated':bool(cut),
                                           'bbox_xyxy':(box[:4]+[rx1,ry1,rx1,ry1]).tolist(),
                                           'mask_polygon':(mask+[rx1,ry1]).tolist()})
            detection_records.append({'frame_index':index,'timestamp_s':index/fps,'detections':deepcopy(detections)})
            candidate_count=len(detections)
            detections,duplicates,original_indices=deduplicate_observations(detections)
            detections,rejected=region.split(detections)
            for item in rejected: item['candidate_index']=original_indices[item['candidate_index']]
            rejected+=duplicates
            stats['candidate_count']+=candidate_count
            stats['pretracking_excluded']+=len(rejected)
            for rejected_item in rejected: stats['excluded:'+rejected_item['reason']]+=1
            tracks,retired=tracker.update(detections,previous,gray,dt,current_bgr=frame)
            objects=[]; excluded_objects=[]; tentative_objects=[]
            for track in tracks:
                d=track.detection; fit=None; raw_fit=None; proposed_size=None; recovery_event=None
                reason='detector_miss' if track.lost else 'heading_pending'
                polygon=track.polygon
                current_observation={'bbox_xyxy':np.r_[polygon.min(axis=0),polygon.max(axis=0)].tolist(),
                                     'mask_polygon':polygon.tolist()} if track.lost else d
                measurement_gate=region.observation_check(current_observation)
                retained=geometry_policy.evaluate(track.metric,track.geometry_age,track.lost,dt,max_lost,region,accepted_positions=track.metric_observations)
                if not measurement_gate['eligible'] and not retained['publish']:
                    excluded_objects.append({'id':track.id,'observed':track.lost==0,**measurement_gate})
                    stats['track_excluded:'+measurement_gate['reason']]+=1
                    continue
                hresult=flow_heading(camera,track.flow) if not track.lost else None
                velocity=None
                # Use previous pose for features tracked FROM previous frame.
                if track.fit and track.geometry_age==1 and len(track.flow)>=5:
                    velocity=flow_velocity(camera,track.fit,track.flow,dt)
                if hresult:
                    theta=hresult['heading_rad']
                    if track.heading is not None:
                        delta=math.atan2(math.sin(theta-track.heading),math.cos(theta-track.heading))
                        # Reject one-frame 180-degree ambiguity; do not force road direction.
                        if abs(delta)<math.pi/2:
                            theta=track.heading+.25*delta
                        else:
                            theta=track.heading
                    track.heading=theta; track.heading_source=hresult['method']
                elif track.heading is None and not track.lost and track.age>=8:
                    try:
                        movement=np.diff(camera.ground([track.initial_center,track.polygon.mean(axis=0)]),axis=0)[0]
                        if np.linalg.norm(movement)>.8:
                            track.heading=math.atan2(movement[1],movement[0]); track.heading_source='accumulated_image_motion_ground_direction'
                    except ValueError:
                        pass
                if track.heading is not None and not track.lost and measurement_gate['eligible']:
                    reason=is_occluded(d,detections,frame.shape)
                    if reason is None:
                        prior=PRIORS.get(track.class_name,PRIORS['car'])
                        raw_fit=fit_cuboid(camera,track.polygon,track.heading,prior)
                        proposed_size=deepcopy(track.size)
                        proposed_size.observe(raw_fit)
                        fit=(fit_cuboid(camera,track.polygon,track.heading,prior,fixed_dimensions=proposed_size.value)
                             if proposed_size.value is not None else raw_fit)
                        reason=fit.get('reason')
                accepted=False
                raw_eligible=(raw_fit and raw_fit.get('status')=='fitted' and
                              region.ground_check(raw_fit['ground_xy'])['eligible'] and
                              not track.lost and d['class_name']==track.class_name)
                track.recovery.observe(raw_fit,track.class_name,index,track.association_score,bool(raw_eligible))
                bootstrap=track.recovery.candidate(fps,minimum=3) if track.metric is None else None
                initialization_ready=track.metric is not None or bootstrap is not None
                if track.metric is None:
                    if bootstrap is None:
                        reason='multiframe_geometry_initialization_pending'
                    else:
                        proposed_size=VehicleSize(value=np.asarray(bootstrap['dimensions_lwh']),
                            samples=[s['dimensions'] for s in track.recovery.samples[-3:]])
                        fit=fit_cuboid(camera,track.polygon,track.heading,PRIORS.get(track.class_name,PRIORS['car']),
                                      fixed_dimensions=proposed_size.value)
                if fit and fit['status']=='fitted':
                    measurement_gate=region.ground_check(fit['ground_xy'])
                if fit and fit['status']=='fitted' and measurement_gate['eligible'] and initialization_ready:
                    if track.metric is None:
                        track.metric=MetricState.at(fit['ground_xy']); accepted=True
                    else:
                        variance=.25+(max(0,fit['ground_xy'][1])/80)**2
                        accepted=track.metric.observe(fit['ground_xy'],variance=variance)
                    if accepted:
                        track.metric_observations+=1
                        track.size=proposed_size
                        if track.size_class is None: track.size_class=track.class_name
                        track.fit=fit; track.geometry_age=0
                    else:
                        reason='metric_innovation_gate'
                track.geometry_reject_streak=0 if accepted else track.geometry_reject_streak+1
                # A separate hypothesis may replace only provisional geometry,
                # after repeated independent evidence and improved image fit.
                changed_class=track.size_class is not None and track.size_class!=track.class_name
                candidate=track.recovery.candidate(fps)
                if (candidate and track.metric is not None and not track.size.locked
                        and (changed_class or track.geometry_reject_streak>=5)):
                    alternate=fit_cuboid(camera,track.polygon,track.heading,PRIORS.get(track.class_name,PRIORS['car']),
                                         fixed_dimensions=candidate['dimensions_lwh'])
                    old_iou=fit.get('silhouette_hull_iou',0) if fit and fit.get('status')=='fitted' else 0.
                    if (alternate.get('status')=='fitted' and alternate['silhouette_hull_iou']>=max(.8,old_iou+.05)
                            and alternate['support_rmse_m']<=.2 and region.ground_check(alternate['ground_xy'])['eligible']):
                        recovery_event={'reason':'provisional_class_change' if changed_class else 'repeated_metric_rejection',
                                        'old_geometry_epoch':track.geometry_epoch,'old_class':track.size_class,
                                        'old_dimensions_lwh':track.fit['dimensions_lwh'],
                                        'old_state_xy_vxy':track.metric.mean.tolist(),'candidate':candidate,
                                        'old_fit_iou':old_iou,'new_fit_iou':alternate['silhouette_hull_iou']}
                        track.geometry_epoch+=1;track.size=VehicleSize(value=np.asarray(candidate['dimensions_lwh']),
                            samples=[s['dimensions'] for s in track.recovery.samples[-6:]])
                        track.size_class=track.class_name
                        track.metric=MetricState.at(alternate['ground_xy'])
                        track.metric.mean[2:]=candidate['velocity_xy_mps']
                        track.metric.covariance=np.diag([1.,1.,9.,9.])
                        track.metric_observations=candidate['sample_count'];track.geometry_age=0
                        track.geometry_reject_streak=0;track.fit=alternate;fit=alternate;accepted=True
                        reason='geometry_reinitialized_from_multiframe_evidence';velocity=None
                        stats['geometry_recoveries']+=1
                if velocity and track.metric is not None and measurement_gate['eligible']:
                    velocity['accepted']=track.metric.observe(velocity['xy_mps'],velocity=True,
                                                             variance=max(1.,velocity['dispersion_mps']**2))
                visibility=geometry_policy.evaluate(track.metric,track.geometry_age,track.lost,dt,max_lost,region,accepted_positions=track.metric_observations)
                located=visibility['publish']
                if not measurement_gate['eligible'] and not located:
                    excluded_objects.append({'id':track.id,'observed':track.lost==0,**measurement_gate})
                    stats['track_excluded:'+measurement_gate['reason']]+=1
                    continue
                row={'id':track.id,'class_name':track.class_name,'instantaneous_detector_class':d['class_name'],
                     'lifecycle':{'state':'coasting' if track.lost and track.confirmed else 'confirmed' if track.confirmed else 'tentative',
                                  'hits':track.hits,'hit_streak':track.hit_streak,'association_score':track.association_score,
                                  'reacquired_after_frames':track.reacquired_after_frames},
                     'measurement_gate':measurement_gate,
                     'geometry_visibility':visibility,
                     'geometry_epoch':track.geometry_epoch,'geometry_recovery_event':recovery_event,
                     'recovery_evidence_samples':len(track.recovery.samples),'size_class':track.size_class,
                     'class_evidence':dict(track.class_evidence),'observed':track.lost==0,
                     'lost_frames':track.lost,'geometry_age_frames':track.geometry_age,
                     'mask_polygon':track.polygon.tolist(),'bbox_xyxy':d['bbox_xyxy'],
                     'geometry_status':'measured' if accepted and located else 'predicted' if located else 'pending',
                     'reason':reason,'heading_source':track.heading_source,'flow_inliers':hresult['inlier_count'] if hresult else 0,
                     'flow_velocity':velocity,'fit_quality':fit,'raw_size_fit':raw_fit,
                     'size_stabilization':track.size.report(),'ground_xy':None,'cuboid_image':None}
                if located:
                    xy=track.metric.mean[:2]; dims=track.fit['dimensions_lwh']
                    row.update(ground_xy=xy.tolist(),velocity_xy_mps=track.metric.mean[2:].tolist(),
                               heading_rad=track.heading,dimensions_lwh=dims,
                               covariance_xy=track.metric.covariance[:2,:2].tolist(),
                               lane=semantic_lane(*xy,config))
                    try: row['cuboid_image']=camera.project(cuboid_vertices(xy,track.heading,dims)).tolist()
                    except ValueError: row['cuboid_image']=None
                if not track.confirmed:
                    tentative_objects.append(row)
                    stats['tentative_object_frames']+=1
                    continue
                objects.append(row)
                stats[row['geometry_status']]+=1
                if reason: stats['reason:'+reason]+=1
                if hresult: stats['heading_flow']+=1
                if velocity and velocity.get('accepted'): stats['velocity_flow']+=1
            # Diagnostic only; intersections are not hidden, relocated or "fixed".
            localized=[r for r in objects if r['ground_xy'] is not None]
            overlap=0
            for i,a in enumerate(localized):
                for b in localized[i+1:]:
                    pa=cuboid_vertices(a['ground_xy'],a['heading_rad'],a['dimensions_lwh'])[:4,:2].astype(np.float32)
                    pb=cuboid_vertices(b['ground_xy'],b['heading_rad'],b['dimensions_lwh'])[:4,:2].astype(np.float32)
                    area,_=cv2.intersectConvexConvex(pa,pb)
                    if area>.1: overlap+=1
            stats['footprint_overlap_pairs']+=overlap
            record={'frame_index':index,'timestamp_s':index/fps,'objects':objects,'retired_ids':retired,
                    'candidate_count':candidate_count,'eligible_candidate_count':len(detections),
                    'rejected_candidates':rejected,'excluded_objects':excluded_objects,'tentative_objects':tentative_objects,
                    'observed_count':sum(o['observed'] for o in objects),'coasting_count':sum(not o['observed'] for o in objects),
                    'localized_count':len(localized),'footprint_overlap_pairs':overlap}
            assert record['observed_count']+sum(o['observed'] for o in excluded_objects+tentative_objects)==len(detections)
            records.append(record)
            score=sum(r['geometry_status']=='measured' for r in objects)
            if make_video or score>best_score:
                rendered=renderer.render(frame,record,calibration)
                if writer: writer.write(rendered)
                if score>best_score:
                    preview=rendered.copy(); best_score=score; preview_index=index
            previous=gray
            if index%30==0: print(json.dumps({'camera':config.get('scene_id'),'frame':index,'frames_requested':max_frames,'device':device_name,
                'elapsed_s':round(time.perf_counter()-started,2),'fps':round((index+1)/(time.perf_counter()-started),2),
                'localized':len(localized),'measured':score,'total':len(objects)}),flush=True)
    finally:
        cap.release()
        if writer: writer.release()
    with (output/'frames.jsonl').open('w') as handle:
        for r in records: handle.write(json.dumps(r,allow_nan=False)+'\n')
    with (output/'detections.jsonl').open('w') as handle:
        for r in detection_records: handle.write(json.dumps(r,allow_nan=False)+'\n')
    if preview is not None: cv2.imwrite(str(output/'validation.png'),preview)
    audit={'performance':{'device':device_name,'elapsed_s':time.perf_counter()-started,'detector_s':detector_seconds,'frames':len(records)},
           'method':'CAROM-2021-guided independent implementation','calibration':calibration,
           'release_version':'offline-v5','source_fps':fps,'geometry_prediction_policy':vars(geometry_policy),
           'geometry_initialization_policy':{'consecutive_shape_samples':3,'recovery_samples':6,
                                             'minimum_association_score':.35,'recovery_only_provisional':True},
           'frames':len(records),'global_ids_created':tracker.next_id-1,'counts':dict(stats),
           'source_sha256':digest,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
           'detector_weights_sha256':hashlib.sha256((ROOT/config['detector']['weights']).read_bytes()).hexdigest(),
           'implementation_sha256':implementation_hashes,
           'vehicle_size_policy':'per_id_robust_initialization_then_rigid_size; refit translation with fixed dimensions',
           'lifecycle_policy':{'min_confirmation_hits':tracker.min_hits,'max_lost_frames':tracker.max_lost,
                               'tentative_max_lost_frames':tracker.tentative_max_lost,'appearance_assisted_reassociation':True,
                               'tentative_evidence_preserved':True},
           'measurement_region':region.settings,
           'detector_region_mode':'cached_full_frame_observations_filtered_before_tracking' if cache_path else 'roi_bounding_crop_with_context_then_polygon_gate',
           'opencv_version':cv2.__version__,
           'detection_source':'local_cached_YOLO_masks' if cache_path else 'local_YOLO_segmentation',
           'cache_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path else None,
           'legacy_metric_states_used':False,'llm_runtime_calls':0,'lane_position_snapping':False,
           'direction_clamping':False,'collision_relocation':False,'preview_frame':preview_index if preview is not None else None,
           'detector_instance_id_conservation':True,
           'limits':['No surveyed calibration for current scene; metric accuracy unverified.',
                     'YOLO masks replace author-finetuned Mask R-CNN.',
                     'Size priors and support-plane least squares replace unpublished empirical rules.',
                     'Flow surface-depth velocity replaces paper accumulation window.',
                     'Pending geometry is shown explicitly, not invented on the map.',
                     'Nominal lane labels are not surveyed lane-ground-truth.',
                     'No GPS/drone/identity ground-truth evaluation yet.']}
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    (output/'scene-config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(audit,indent=2),flush=True)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'configs/user_bridge_traffic.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/carom-method-wide-stable')
    p.add_argument('--max-frames',type=int,default=90)
    p.add_argument('--experimental-calibration',action='store_true')
    p.add_argument('--detection-cache',type=Path)
    p.add_argument('--video',action='store_true')
    args=p.parse_args()
    process(args.config,args.output,args.max_frames,args.experimental_calibration,args.detection_cache,args.video)
