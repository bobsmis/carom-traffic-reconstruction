"""Replay unchanged computed states at a larger metric display scale."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.pipeline.carom_geometry import RoadCamera
from app.pipeline.metric_bev import MetricBevDetail


def export(output,pixels_per_meter=20.):
    data=output/'frames.jsonl'; before=hashlib.sha256(data.read_bytes()).hexdigest()
    cfg=json.loads((output/'scene-config.json').read_text())
    audit=json.loads((output/'audit.json').read_text())
    rows=[json.loads(line) for line in data.read_text().splitlines()]
    camera,_=RoadCamera.from_scene(cfg,experimental=True)
    renderer=MetricBevDetail(camera,cfg,ROOT/cfg['video_path'],pixels_per_meter)
    video=cv2.VideoCapture(str(ROOT/cfg['video_path'])); fps=video.get(cv2.CAP_PROP_FPS); video.release()
    first=renderer.render(rows[0]); height,width=first.shape[:2]
    out=cv2.VideoWriter(str(output/'bev-detail-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not out.isOpened(): raise RuntimeError('BEV video writer failed')
    preview=None
    try:
        for row in rows:
            rendered=renderer.render(row); out.write(rendered)
            if row['frame_index']==audit['preview_frame']:
                preview=rendered; cv2.imwrite(str(output/'bev-detail.png'),rendered)
    finally: out.release()
    if preview is None: raise ValueError('audit preview frame missing')
    # Separate full-size composite; neither stretch a single axis nor squeeze the BEV.
    original=cv2.imread(str(output/'validation.png'))
    if original is None or original.shape[:2]!=(1080,1920):
        raise ValueError('expected original 1920x1080 validation layout')
    top=original[:584]
    top=cv2.resize(top,(width,round(top.shape[0]*width/top.shape[1])),interpolation=cv2.INTER_AREA)
    import numpy as np
    cv2.imwrite(str(output/'validation-detail.png'),np.vstack((top,preview)))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
                    '-i',str(output/'bev-detail-raw.mp4'),'-c:v','libx264','-preset','fast','-crf','21',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(output/'bev-detail.mp4')],check=True)
    cap=cv2.VideoCapture(str(output/'bev-detail.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(width,height)
    cap.release()
    after=hashlib.sha256(data.read_bytes()).hexdigest(); assert before==after
    meta={'pixels_per_meter_x':pixels_per_meter,'pixels_per_meter_y':pixels_per_meter,
          'previous_native_pixels_per_meter':10,'world_coordinates_changed':False,
          'frames_sha256':before,'frames':len(rows),'width':width,'height':height,
          'fps':fps,'preview_frame':audit['preview_frame'],
          'world_to_native_bev':renderer.viewport.transform.tolist(),
          'note':'Native pixels only; fitted-to-window viewing rescales the image. 1m grid scales with it.',
          'renderer_sha256':hashlib.sha256((ROOT/'backend/app/pipeline/metric_bev.py').read_bytes()).hexdigest()}
    (output/'bev-detail.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--pixels-per-meter',type=float,default=20.)
    args=p.parse_args();export(args.output,args.pixels_per_meter)
