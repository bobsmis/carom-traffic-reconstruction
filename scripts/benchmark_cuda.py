"""Same-input FP32 CPU/CUDA detector benchmark and conservative activation gate.
Not an end-to-end accuracy benchmark; no truth data is read.
"""
import json
import time
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from evaluate_carla_proxy import iou

ROOT=Path(__file__).resolve().parents[1]
def main():
    assert torch.cuda.is_available(), 'CUDA not available'
    assert '+cpu' not in torch.__version__
    cfg=json.loads((ROOT/'configs/multi-20260927-102144-cam01.json').read_text())
    source=ROOT/cfg['video_path'];capture=cv2.VideoCapture(str(source));frames=[]
    for index in range(60,100,4):
        capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read();assert ok;frames.append(frame)
    capture.release();dcfg=cfg['detector'];results={};predictions={}
    for device in ['cpu','cuda:0']:
        model=YOLO(str(ROOT/dcfg['weights']))
        args=dict(imgsz=dcfg['image_size'],conf=dcfg['confidence'],classes=dcfg['classes'],device=device,half=False,verbose=False)
        model.predict(frames[0],**args)
        if device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=[];rows=[]
        for frame in frames:
            start=time.perf_counter();p=model.predict(frame,**args)[0]
            row=p.boxes.data.cpu().numpy().tolist()
            if device.startswith('cuda'):torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-start);rows.append(row)
        predictions[device]=rows;results[device]={'frames':len(frames),'seconds':sum(elapsed),'fps':len(frames)/sum(elapsed),'median_ms':float(np.median(elapsed))*1000}
        del model
    total=max_total=matched=0;min_iou=1.
    for cpu,gpu in zip(predictions['cpu'],predictions['cuda:0']):
        max_total+=max(len(cpu),len(gpu));total+=len(cpu)
        if not cpu or not gpu:continue
        scores=np.array([[iou(a[:4],b[:4]) if int(a[5])==int(b[5]) else 0 for b in gpu] for a in cpu])
        r,c=linear_sum_assignment(1-scores)
        for a,b in zip(r,c):
            if scores[a,b]>=.99:matched+=1;min_iou=min(min_iou,float(scores[a,b]))
    report={'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'precision':'FP32','source':str(source),
        'weights_sha256':hashlib.sha256((ROOT/dcfg['weights']).read_bytes()).hexdigest(),
        'benchmark':results,'speedup':results['cpu']['seconds']/results['cuda:0']['seconds'],
        'box_class_agreement_at_iou_099':matched/max_total if max_total else 1.,'cpu_boxes':total,
        'limitations':'10 identical full frames; warmup excluded; concurrent CPU worker/CARLA may affect timing; masks and end-to-end geometry not compared.'}
    out=ROOT/'outputs/cuda-benchmark.json';out.write_text(json.dumps(report,indent=2))
    # Explicit marker is required by the camera-boundary launcher. No silent CPU fallback activation.
    if report['box_class_agreement_at_iou_099']<.95 or report['speedup']<=1:
        raise RuntimeError('Runtime benchmark gate did not pass; automatic handoff stays disabled')
    marker=ROOT/'.venv-cuda/VERIFIED.json';temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(marker)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
