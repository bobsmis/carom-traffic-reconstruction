"""Independent invariant checks; these do NOT substitute for labelled accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.pipeline.carom_geometry import RoadCamera,cuboid_vertices
from app.pipeline.measurement_region import MeasurementRegion
from app.pipeline.carom_tracking import polygon_iou


def verify(output):
    audit=json.loads((output/'audit.json').read_text())
    cfg=json.loads((output/'scene-config.json').read_text())
    camera,_=RoadCamera.from_scene(cfg,experimental=True)
    region=MeasurementRegion(camera,cfg)
    counts=Counter(); reprojection=[]; ious=[]; posterior_ious=[]; speeds=[]; prior_bounds=0
    rows=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    expired=set(); locked_sizes={}; last_sizes={}; last_raw={}; raw_jumps=[]; output_jumps=[]
    epochs={}; recoveries=[]; last_any_size={}; revision_jumps=[]
    for frame in rows:
        objects=frame['objects']; ids=[o['id'] for o in objects]
        assert len(ids)==len(set(ids)), 'duplicate ID within frame'
        assert not expired.intersection(ids), 'retired ID reused'
        expired.update(frame['retired_ids'])
        assert sum(o['observed'] for o in objects)==frame['observed_count']
        assert sum(not o['observed'] for o in objects)==frame['coasting_count']
        assert sum(o['ground_xy'] is not None for o in objects)==frame['localized_count']
        if 'candidate_count' in frame:
            assert frame['candidate_count']==frame['eligible_candidate_count']+len(frame['rejected_candidates'])
            assert frame['observed_count']+sum(o['observed'] for o in frame['excluded_objects']+frame.get('tentative_objects',[]))==frame['eligible_candidate_count']
            assert not set(ids).intersection(o['id'] for o in frame['excluded_objects'])
            assert not set(ids).intersection(o['id'] for o in frame.get('tentative_objects',[]))
        for obj in objects:
            epoch=obj.get('geometry_epoch',0);tid=obj['id']
            if tid in epochs and epoch!=epochs[tid]:
                event=obj.get('geometry_recovery_event')
                assert event and epoch==epochs[tid]+1 and event['old_geometry_epoch']==epochs[tid]
                assert (tid,epochs[tid]) not in locked_sizes, 'locked shape automatically reset'
                assert event['candidate']['sample_count']>=6
                assert event['new_fit_iou']>=event['old_fit_iou']+.05
                recoveries.append({'frame':frame['frame_index'],'id':tid,'event':event})
            epochs[tid]=epoch
            assert 'cuboid_image_fitted' not in obj
            counts[obj['geometry_status']]+=1
            if obj['geometry_status']=='pending':
                assert obj['ground_xy'] is None
            if obj['ground_xy'] is None:
                continue
            assert region.ground_check(obj['ground_xy'])['eligible'], 'measurement outside valid region'
            if obj.get('size_stabilization'):
                dims=np.asarray(obj['dimensions_lwh'])
                prior=last_any_size.get(obj['id'])
                if prior is not None and prior[0]!=epoch:
                    revision_jumps.append(float(np.max(np.abs(dims/prior[1]-1))))
                last_any_size[obj['id']]=(epoch,dims.copy())
                tid=(obj['id'],obj.get('geometry_epoch',0)); policy=obj['size_stabilization']
                if tid in locked_sizes:
                    assert np.array_equal(dims,locked_sizes[tid]), 'locked identity changed physical size'
                if policy['state']=='locked':
                    locked_sizes[tid]=dims.copy()
                if tid in last_sizes:
                    before,frame_before=last_sizes[tid]
                    gap=frame['frame_index']-frame_before
                    change=float(np.max(np.abs(dims/before-1)))
                    assert change<=(1+policy['max_step_fraction'])**gap-1+1e-8
                    if gap==1: output_jumps.append(change)
                last_sizes[tid]=(dims.copy(),frame['frame_index'])
                raw=obj.get('raw_size_fit')
                if raw and raw['status']=='fitted':
                    raw_dims=np.asarray(raw['dimensions_lwh'])
                    if tid in last_raw and frame['frame_index']-last_raw[tid][1]==1:
                        raw_jumps.append(float(np.max(np.abs(raw_dims/last_raw[tid][0]-1))))
                    last_raw[tid]=(raw_dims,frame['frame_index'])
            if obj.get('geometry_visibility'):
                visibility=obj['geometry_visibility']
                assert visibility['publish']
                assert visibility['prediction_age_s']<=visibility['max_gap_seconds']
                assert visibility['position_sigma_m']<=visibility['max_position_sigma_m']
                assert obj['lost_frames']<=cfg['processing']['global_tracking']['max_lost_frames']
            else:
                assert obj['geometry_age_frames']<=cfg['processing']['global_tracking']['max_lost_frames']
            if obj.get('offline_estimate') and audit.get('release_version') in ('offline-v4','offline-v5'):
                assert obj['geometry_status']=='smoothed'
                assert obj['measurement_status'] in ('accepted','missing','rejected')
                assert (obj['measurement_status']=='accepted')==obj['offline_estimate']['measurement_used']
            assert obj['lane']['status'] in ('outside_nominal_lane_model','nominal_geometry_unverified')
            if obj['lane']['lane_id']:
                assert obj['lane']['position_was_snapped'] is False
            assert np.linalg.eigvalsh(obj['covariance_xy']).min()>=-1e-8
            vertices=cuboid_vertices(obj['ground_xy'],obj['heading_rad'],obj['dimensions_lwh'])
            assert np.allclose(vertices[:4,:2].mean(axis=0),obj['ground_xy'])
            if obj['cuboid_image']:
                error=np.max(np.linalg.norm(camera.project(vertices)-obj['cuboid_image'],axis=1))
                reprojection.append(float(error))
                assert error<1e-6
                if obj['observed']:
                    posterior_ious.append(polygon_iou(obj['cuboid_image'],obj['mask_polygon']))
            speeds.append(float(np.linalg.norm(obj['velocity_xy_mps'])*3.6))
            fit=obj.get('fit_quality')
            if fit and fit['status']=='fitted':
                ious.append(fit['silhouette_hull_iou'])
            counts['lane:'+str(obj['lane']['lane_id'])]+=1
    assert len(rows)==audit['frames']
    report={'invariants_passed':True,'frames':len(rows),'object_state_counts':dict(counts),
            'max_shared_projection_disagreement_px':max(reprojection,default=0),
            'fit_hull_iou_median':float(np.median(ious)) if ious else None,
            'posterior_projected_mask_hull_iou_median':float(np.median(posterior_ious)) if posterior_ious else None,
            'posterior_projected_mask_hull_iou_p10':float(np.percentile(posterior_ious,10)) if posterior_ious else None,
            'unvalidated_speed_kmh_p95':float(np.percentile(speeds,95)) if speeds else None,
            'footprint_overlap_pairs':audit['counts']['footprint_overlap_pairs'],
            'accuracy_validated':False,'identity_accuracy_validated':False,
            'measurement_region_enforced':region.enabled,
            'pretracking_excluded':audit['counts'].get('pretracking_excluded',0),
            'locked_size_track_count':len(locked_sizes),
            'locked_size_changes':0,
            'geometry_recovery_events':recoveries,
            'max_size_revision_fraction':max(revision_jumps,default=0),
            'raw_size_frame_change_p95':float(np.percentile(raw_jumps,95)) if raw_jumps else None,
            'output_size_frame_change_p95':float(np.percentile(output_jumps,95)) if output_jumps else None,
            'output_size_frame_change_max':max(output_jumps,default=0),
            'note':'Geometry algebra, bookkeeping and conservation only; not position, dimension, lane or ID accuracy.'}
    (output/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); args=p.parse_args(); verify(args.output)
