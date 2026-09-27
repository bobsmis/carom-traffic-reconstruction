"""Paired V4/V5 audit of this source fixture, not an accuracy benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.pipeline.carom_geometry import cuboid_vertices
from app.pipeline.carom_tracking import polygon_iou


def load(path):
    rows=[json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines()]
    return rows,{(f['frame_index'],r['id']):r for f in rows for r in f['objects']}


def overlaps(rows):
    result=[]
    for f in rows:
        objects=[r for r in f['objects'] if r['ground_xy'] is not None]
        for i,a in enumerate(objects):
            for b in objects[i+1:]:
                polygons=[cuboid_vertices(r['ground_xy'],r['heading_rad'],r['dimensions_lwh'])[:4,:2].astype(np.float32) for r in (a,b)]
                area,_=cv2.intersectConvexConvex(*polygons)
                if area>.1:result.append({'frame':f['frame_index'],'ids':[a['id'],b['id']],'area_nominal_m2':float(area)})
    return result


def check(previous,current):
    fa,a=load(previous);fb,b=load(current)
    audits=[json.loads((p/'audit.json').read_text()) for p in (previous,current)]
    assert audits[0]['source_sha256']==audits[1]['source_sha256']=='f6106e513f8f454097c1d7160ae495bb732597e110afaa6f39034358b2cfd4a8'
    assert len(fa)==len(fb)==240
    caches=[Path(x['offline_smoothing']['input_path'])/'detections.jsonl' for x in audits]
    cache_hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in caches]
    assert cache_hashes[0]==cache_hashes[1], 'paired comparison requires identical detections'
    common=set(a)&set(b)
    paired=sorted(k for k in common if all(d[k]['ground_xy'] is not None and d[k]['observed'] and d[k]['cuboid_image'] for d in (a,b)))
    fit={}
    for name,d in [('v4',a),('v5',b)]:
        values=[polygon_iou(d[k]['cuboid_image'],d[k]['mask_polygon']) for k in paired]
        fit[name]={'median':float(np.median(values)),'p10':float(np.percentile(values,10))}
    old_overlap,new_overlap=overlaps(fa),overlaps(fb)
    assert len(new_overlap)==0
    pair_common_frames=[i for i in range(240) if all((i,tid) in d and d[i,tid]['ground_xy'] is not None for tid in (24,26) for d in (a,b))]
    pair_first={}
    for tid in (24,26):
        localized=[(i,r) for (i,j),r in b.items() if j==tid and r['ground_xy'] is not None]
        assert localized and localized[-1][0]==239
        pair_first[tid]={'first_localized_frame':localized[0][0],'initial_dimensions_lwh':localized[0][1]['dimensions_lwh']}
    assert pair_first[24]['first_localized_frame']>=82
    assert pair_first[26]['first_localized_frame']>81
    assert all(b[i,26]['ground_xy'] is not None for i in range(96,240))
    report={'passed':True,'scope':'240-frame fixture only; not ground-truth accuracy',
            'same_source_and_detection_cache':True,'detections_sha256':cache_hashes[0],
            'unique_ids':[len({k[1] for k in d}) for d in (a,b)],
            'published_object_frames':[len(a),len(b)],
            'localized_object_frames':[sum(r['ground_xy'] is not None for r in d.values()) for d in (a,b)],
            'added_publications':sorted(set(b)-set(a)),'removed_publications':sorted(set(a)-set(b)),
            'common_newly_localized':sum(a[k]['ground_xy'] is None and b[k]['ground_xy'] is not None for k in common),
            'common_lost_localization':sum(a[k]['ground_xy'] is not None and b[k]['ground_xy'] is None for k in common),
            'common_observed_localized_samples':len(paired),'paired_projected_hull_iou':fit,
            'overlap_counts':[len(old_overlap),len(new_overlap)],'old_overlap_evidence':old_overlap,
            'ids24_26_common_localized_frames':len(pair_common_frames),
            'old_overlaps_on_pair_common_frames':sum(x['frame'] in pair_common_frames and set(x['ids'])=={24,26} for x in old_overlap),
            'ids24_26_v5_initialization':pair_first,
            'accuracy_validated':False,'note':'Overlap reduction includes delayed initialization and changed dimensions; neither is proof of physical accuracy.'}
    (current/'v5-regressions.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous',type=Path);p.add_argument('current',type=Path)
    args=p.parse_args();check(args.previous,args.current)
