"""Offline synchronized multi-camera reconstruction, input-only perception path.

No truth/ files, CARLA actor IDs, velocities, sizes or simulator APIs are read.
Global association is experimental, not a validated long-range identity result.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.pipeline.multicam import GlobalTracker
from import_carla_input import convert

def appearance(image,obj):
    mask=np.zeros(image.shape[:2],np.uint8)
    polygon=np.asarray(obj.get('mask_polygon',[]),np.int32)
    if len(polygon)>=3:cv2.fillPoly(mask,[polygon],255)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0,1],mask,[16,8],[0,180,0,256]).reshape(-1)
    return h/max(float(np.linalg.norm(h)),1e-6)

def sprites():
    result={}
    for kind in ('car','truck','bus'):
        p=cv2.imread(str(ROOT/'assets/vehicles'/f'{kind}.png'),cv2.IMREAD_UNCHANGED)
        if p is None:raise ValueError('missing provided vehicle sprite')
        if p.shape[2]==3:p=np.dstack([p,np.where(p.max(axis=2)>10,255,0).astype(np.uint8)])
        ys,xs=np.where(p[:,:,3]>10);p=p[ys.min():ys.max()+1,xs.min():xs.max()+1]
        # Supplied truck points left; others point up. Normalize all to right.
        p=cv2.rotate(p,cv2.ROTATE_180 if kind=='truck' else cv2.ROTATE_90_CLOCKWISE)
        result[kind]=p
    return result

SPRITES=None
def car_icon(canvas,xy,obj,scale):
    global SPRITES
    if SPRITES is None:SPRITES=sprites()
    p=SPRITES.get(obj['class_name'],SPRITES['car'])
    length,width,_=obj['dimensions']
    # Fit within metric footprint, keep original sprite aspect ratio (no stretching).
    factor=min(max(4,length*scale)/p.shape[1],max(3,width*scale)/p.shape[0])
    p=cv2.resize(p,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    angle=math.degrees(obj['heading']-math.pi/2)
    ph,pw=p.shape[:2];matrix=cv2.getRotationMatrix2D((pw/2,ph/2),angle,1.)
    nw=math.ceil(abs(matrix[0,0])*pw+abs(matrix[0,1])*ph)
    nh=math.ceil(abs(matrix[0,1])*pw+abs(matrix[0,0])*ph)
    matrix[0,2]+=nw/2-pw/2;matrix[1,2]+=nh/2-ph/2
    p=cv2.warpAffine(p,matrix,(nw,nh),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
    h,w=p.shape[:2];x=int(xy[0]-w/2);y=int(xy[1]-h/2)
    x0=max(0,x);y0=max(0,y);x1=min(canvas.shape[1],x+w);y1=min(canvas.shape[0],y+h)
    if x1<=x0 or y1<=y0:return
    q=p[y0-y:y1-y,x0-x:x1-x];alpha=q[:,:,3:4]/255.
    if obj['status']=='predicted':alpha*=.55
    canvas[y0:y1,x0:x1]=(q[:,:,:3]*alpha+canvas[y0:y1,x0:x1]*(1-alpha)).astype(np.uint8)
    cv2.putText(canvas,str(obj['id']),(x,max(12,y-3)),0,.3,(80,240,245),1,cv2.LINE_AA)

def map_view(objects,nodes,offset=0):
    canvas=np.full((540,960,3),(30,35,39),np.uint8)
    # One metric crop, isotropic scaling. No folded/stacked whole-tunnel map.
    bands=1
    span=60.
    scale=14.
    for band in range(bands):
        start=offset+15.
        top=65
        left=38
        cv2.rectangle(canvas,(left,top),(int(left+span*scale),int(top+28*scale)),(62,67,72),-1)
        for lane in range(9):
            y=int(top+lane*3.5*scale)
            cv2.line(canvas,(left,y),(int(left+span*scale),y),(65,190,210) if lane==4 else (125,130,135),1)
        cv2.putText(canvas,f'{start:.0f} - {start+span:.0f} m',(left,top-7),0,.35,(200,210,220),1)
        for node in nodes:
            a=max(start,node['global_offset_m']);b=min(start+span,node['global_offset_m']+150)
            if a<b:
                cv2.line(canvas,(int(left+(a-start)*scale),top),(int(left+(b-start)*scale),top),(150,180,100),2)
                cv2.putText(canvas,node['id'],(int(left+(a-start)*scale),top+12),0,.28,(160,200,140),1)
        for obj in objects:
            x,y=obj['xy']
            if start<=y<start+span:car_icon(canvas,(left+(y-start)*scale,top+(x+14)*scale),obj,scale)
    cv2.putText(canvas,'RGB-derived global IDs | experimental association | sprites are visual symbols',(22,535),0,.35,(170,190,200),1)
    return canvas

def encode(source,target):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(source),'-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)

def run(source,name):
    if not name.replace('-','').isalnum():raise ValueError('invalid name')
    registry=json.loads((source/'cameras.json').read_text());nodes=registry['cameras']
    ids=[node['id'] for node in nodes]
    if len(ids)!=len(set(ids)) or not all(re.fullmatch(r'cam\d{2}',cid) for cid in ids):raise ValueError('invalid camera registry')
    manifest=json.loads((source/'manifest.json').read_text());fps=manifest['fps'];n=manifest['frames']
    records={};caps={};outputs={};indexes={}
    for node in nodes:
        cid=node['id'];scene=name+'-'+cid;config=ROOT/'configs'/f'{scene}.json'
        if not config.exists():convert(source/cid,scene)
        output=ROOT/'outputs'/scene;outputs[cid]=output
        if not (output/'audit.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/run_carom_method.py'),'--config',str(config),'--output',str(output),'--max-frames',str(n)],check=True)
        records[cid]=[json.loads(s) for s in (output/'frames.jsonl').read_text().splitlines()]
        digest=hashlib.sha256((source/cid/'rgb.mp4').read_bytes()).hexdigest()
        declared=json.loads((source/cid/'manifest.json').read_text())['sha256']
        audited=json.loads((output/'audit.json').read_text())['source_sha256']
        if digest!=declared or digest!=audited:raise ValueError('camera input/cache hash mismatch')
        indexes[cid]=[json.loads(s) for s in (source/cid/'frames.jsonl').read_text().splitlines()]
        if len(records[cid])!=n or len(indexes[cid])!=n:raise ValueError('frame count mismatch')
        caps[cid]=cv2.VideoCapture(str(source/cid/'rgb.mp4'))
    web=source.parent/'web';web.mkdir(exist_ok=True);(web/'complete.json').unlink(missing_ok=True)
    temp=ROOT/'outputs'/name;temp.mkdir(exist_ok=True)
    tracker=GlobalTracker();rows=[];writers={}
    def writer(key):
        target=temp/(key.replace('/','-')+'.mp4')
        w=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(960,540))
        if not w.isOpened():raise RuntimeError('encoder failed')
        writers[key]=(w,target);return w
    anno={node['id']:writer(node['id']+'/annotation') for node in nodes};bev={node['id']:writer(node['id']+'/bev') for node in nodes}
    volume_audit=[]
    try:
        for index in range(n):
            images={};observations=[];reference=indexes[nodes[0]['id']][index]['carla_frame']
            for node in nodes:
                cid=node['id']
                if indexes[cid][index]['carla_frame']!=reference:raise ValueError('asynchronous camera frames')
                ok,image=caps[cid].read()
                if not ok:raise ValueError('RGB decode ended early')
                images[cid]=image
                for obj in records[cid][index]['objects']:
                    if obj.get('ground_xy') is None:continue
                    observations.append({'camera':cid,'local_id':obj['id'],
                        'xy':np.asarray(obj['ground_xy'])+[0,node['global_offset_m']],
                        'velocity':np.asarray(obj.get('velocity_xy_mps',[0,0])),
                        'appearance':appearance(image,obj),'heading':obj['heading_rad'],
                        'dimensions':obj['dimensions_lwh'],'class_name':obj['class_name'],
                        'variance':float(np.trace(obj.get('covariance_xy',np.eye(2)))), 'observed':obj['observed']})
            objects=tracker.update(observations,index/fps)
            row={'frame_index':index,'time_s':index/fps,'objects':objects};rows.append(row)
            identities={(s['camera'],s['local_id']):o['id'] for o in objects for s in o['sources']}
            for node in nodes:
                cid=node['id'];image=images[cid].copy()
                for obj in records[cid][index]['objects']:
                    gid=identities.get((cid,obj['id']));box=np.asarray(obj['bbox_xyxy'],int)
                    cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(70,215,160),1)
                    cv2.putText(image,f'G{gid}' if gid is not None else f'local {obj["id"]} / pending',tuple(box[:2]),0,.4,(80,240,245),1)
                    verdict=volume_verdict(obj)
                    volume_audit.append({'frame':index,'camera':cid,'local_id':obj['id'],**verdict})
                    corners=np.asarray(obj.get('cuboid_image') or [],dtype=float)
                    if corners.shape==(8,2) and np.isfinite(corners).all() and np.abs(corners).max()<100000:
                        color=(80,220,120) if verdict['accepted'] else (0,165,255)
                        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                            cv2.line(image,tuple(corners[a].astype(int)),tuple(corners[b].astype(int)),color,1,cv2.LINE_AA)
                        label='estimated LWH '+ 'x'.join(f'{v:.1f}' for v in obj['dimensions_lwh'])
                        cv2.putText(image,label,tuple(box[:2]+[0,14]),0,.32,color,1)
                anno[cid].write(cv2.resize(image,(960,540)))
                bev[cid].write(map_view([o for o in objects if cid in o['cameras']],nodes,node['global_offset_m']))
    finally:
        for w,_ in writers.values():w.release()
        for cap in caps.values():cap.release()
    for key,(_,path) in writers.items():
        target=web/(key+'.mp4');target.parent.mkdir(exist_ok=True,parents=True);encode(path,target)
    for node in nodes:encode(source/node['id']/'rgb.mp4',web/node['id']/'raw.mp4')
    (web/'global.json').write_text(json.dumps({'fps':fps,'frames':rows,'identity_accuracy_validated':False,
        'processing_mode':'offline','llm_runtime_calls':0,'oracle_vehicle_states_used':False},allow_nan=False))
    (web/'volume-audit.json').write_text(json.dumps(volume_audit,allow_nan=False))
    subprocess.run([sys.executable,str(ROOT/'scripts/sumo_replay.py'),'--web',str(web),'--rig',str(source/'cameras.json')],check=True)
    (web/'complete.json').write_text(json.dumps({'frames':n,'fps':fps,'cameras':len(nodes),'status':'completed'}))
    print('MULTICAM_OUTPUT='+str(web),flush=True)

def volume_verdict(obj):
    dims=np.asarray(obj.get('dimensions_lwh',[]),dtype=float)
    fit=obj.get('fit_quality') or {}
    reasons=[]
    if dims.shape!=(3,) or not np.isfinite(dims).all():reasons.append('missing_or_nonfinite_dimensions')
    elif not (1.5<=dims[0]<=20 and .8<=dims[1]<=3.2 and .8<=dims[2]<=4.8):reasons.append('size_outside_broad_vehicle_bounds')
    if fit.get('status')!='fitted' or fit.get('silhouette_hull_iou',0)<.6:reasons.append('weak_silhouette_fit')
    if (obj.get('size_stabilization') or {}).get('state')!='locked':reasons.append('size_not_locked')
    return {'accepted':not reasons,'reasons':reasons,'source':'rgb_silhouette_camera_geometry','absolute_accuracy_validated':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--name',required=True)
    a=p.parse_args();run(a.input,a.name)
