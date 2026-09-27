"""Portable local evidence viewer; no network requests or model-generated media."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]


def export(output,destination):
    cfg=json.loads((output/'scene-config.json').read_text())
    audit=json.loads((output/'audit.json').read_text())
    report=json.loads((output/'smoothing-report.json').read_text())
    original=Path(audit['offline_smoothing']['input_path'])
    destination.mkdir(parents=True,exist_ok=True)
    names=['validation.png','validation-detail.png','bev-detail.png','mobile.mp4','bev-detail.mp4',
           'audit.json','verification.json','smoothing-report.json','calibration-readiness.json','frames.jsonl','scene-config.json']
    for name in names: shutil.copy2(output/name,destination/name)
    shutil.copy2(original/'detections.jsonl',destination/'detections.jsonl')
    shutil.copy2(original/'frames.jsonl',destination/'online-frames.jsonl')
    version=audit.get('release_version','offline-v3')
    labels={'offline-v3':'V3','offline-v4':'V4','offline-v5':'V5'}
    label=labels[version]
    document=f'OFFLINE_{label}_IMPLEMENTATION.md'
    shutil.copy2(ROOT/'docs'/document,destination/'IMPLEMENTATION.md')
    if (output/'regressions.json').exists(): shutil.copy2(output/'regressions.json',destination/'regressions.json')
    if (output/'v5-regressions.json').exists(): shutil.copy2(output/'v5-regressions.json',destination/'v5-regressions.json')
    evidence=output/'id24-26-source-evidence.png'
    if evidence.exists(): shutil.copy2(evidence,destination/evidence.name)
    video=ROOT/cfg['video_path'];crop=cfg['crop']
    filters=f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},scale=1280:-2"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(video),
                    '-frames:v',str(audit['frames']),'-vf',filters,'-an','-c:v','libx264','-crf','20',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(destination/'original-clip.mp4')],check=True)
    cap=cv2.VideoCapture(str(destination/'original-clip.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==audit['frames'];cap.release()
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CAROM 离线 V3 核验</title>
<style>body{font:16px system-ui;background:#111820;color:#e8eef2;margin:0 auto;max-width:1500px;padding:16px}
h1{font-size:24px}h2{font-size:19px}a{color:#73d9cd}video,img{display:block;width:100%;height:auto;background:#000}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:#1c2732;padding:14px;border-radius:10px}
.warning{border-left:4px solid #edb85b;padding:12px;background:#342d23}button{padding:10px;margin:8px 8px 8px 0}
#plane{overflow:auto}#plane img{width:2160px;max-width:none}small{color:#b9c8d3}</style>
<h1>CAROM · 离线 V3 核验包</h1>
<p class="warning">实验结果，非完整论文复现或精度验收。当前标定使用假定参数；车道错位、尺度和残余重叠仍待实测核验。所有视频来自本地素材与算法处理。</p>
<p>__FRAMES__ 帧。__HITS__ 帧确认 ID；短时漏检保留 __LOST__ 帧；离线利用后续帧平滑，不是实时效果。逐帧 ID 集合保持不变。</p>
<div class="grid"><section class="card"><h2>原始素材</h2><video id="raw" controls playsinline preload="metadata" src="original-clip.mp4"></video></section>
<section class="card"><h2>分割、ID、体积框及俯视位置</h2><video id="result" controls playsinline preload="metadata" src="mobile.mp4"></video></section></div>
<button id="sync">从头对照播放</button><button id="pause">同时暂停</button><small>播放器同步仅供肉眼对照；精确对应请查看 frames.jsonl 的帧号。</small>
<section class="card"><h2>俯视图动态结果（横纵等比例）</h2><video controls playsinline preload="metadata" src="bev-detail.mp4"></video></section>
<h2>验证截图</h2><a href="validation-detail.png"><img src="validation-detail.png" alt="原始画面、检测及俯视位置"></a>
<h2>核验信息</h2><p>抖动代理指标 P95：__JITTER_BEFORE__ → __JITTER_AFTER__；朝向帧间变化 P95：__HEADING_BEFORE__° → __HEADING_AFTER__°。这些是平滑性，不是定位准确率。残余重叠：__OVERLAP__ 次车辆对—帧。</p>
<p><a href="frames.jsonl">离线逐帧状态</a> · <a href="online-frames.jsonl">平滑前逐帧状态</a> · <a href="detections.jsonl">原始检测掩膜</a> ·
<a href="smoothing-report.json">前后对比</a> · <a href="calibration-readiness.json">标定缺项</a> · <a href="verification.json">一致性测试</a> · <a href="IMPLEMENTATION.md">实现与局限</a></p>
<script>const raw=document.getElementById('raw'),result=document.getElementById('result');
document.getElementById('sync').onclick=async()=>{raw.currentTime=0;result.currentTime=0;try{await Promise.all([raw.play(),result.play()])}catch(e){}};
document.getElementById('pause').onclick=()=>{raw.pause();result.pause()};</script></html>'''
    html=html.replace('离线 V3', '离线 '+label)
    if version=='offline-v5':
        html=html.replace('<h2>核验信息</h2>', '<h2>核验信息</h2><p class="warning">V5 增加连续三帧几何初始化。当前片段重叠 19 → 0，但定位记录 3638 → 3560（减少 2.1%）；零重叠部分来自延迟发布，并非物理精度证明。ID 24/26 的估计尺寸仍不合理。六帧证据恢复分支在本片段未触发。</p><p><a href="v5-regressions.json">V4/V5 同素材同检测逐项对照</a></p>')
    values={'FRAMES':audit['frames'],'HITS':audit['lifecycle_policy']['min_confirmation_hits'],
            'LOST':audit['lifecycle_policy']['max_lost_frames'],'OVERLAP':report['after']['overlap_pair_frames']}
    for stage in ('before','after'):
        values['JITTER_'+stage.upper()]=round(report[stage]['position_second_difference_mps2_p95'],2)
        values['HEADING_'+stage.upper()]=round(report[stage]['heading_step_deg_p95'],2)
    for key,value in values.items(): html=html.replace('__'+key+'__',str(value))
    (destination/'index.html').write_text(html,encoding='utf-8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file() and p.name!='SHA256.json'}
    (destination/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'bundle':str(destination),'files':len(manifest),'frames':audit['frames']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('destination',type=Path)
    args=p.parse_args();export(args.output,args.destination)
