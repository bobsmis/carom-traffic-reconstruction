"""Read-only evidence API for the paper-method branch (separate from V1)."""
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

router=APIRouter(prefix='/api/carom-method',tags=['paper-method'])
OUTPUT=Path(os.environ.get('CAROM_METHOD_OUTPUT',str(Path(__file__).resolve().parents[2]/'outputs/carom-offline-v5-final')))


def require(name):
    path=OUTPUT/name
    if not path.is_file():
        raise HTTPException(503,'paper-method artifact has not been produced')
    return path


@router.get('/audit')
def audit():
    return json.loads(require('audit.json').read_text())


@router.get('/run-info')
def run_info():
    # Do not publish a partial artifact set or silently fall back to legacy V1.
    for name in ('frames.jsonl','validation.png','mobile.mp4','bev-detail.mp4','verification.json',
                 'smoothing-report.json','calibration-readiness.json'):
        require(name)
    a=json.loads(require('audit.json').read_text())
    return {'run_id':OUTPUT.name,'version':a.get('release_version','unversioned-experiment'),
            'frames':a['frames'],'fps':a.get('source_fps',30.),'preview_frame':a['preview_frame'],
            'source_sha256':a['source_sha256'],'calibration':a['calibration'],
            'geometry_prediction_policy':a.get('geometry_prediction_policy'),
            'api_namespace':'/api/carom-method','accuracy_validated':False}


@router.get('/verification')
def verification():
    return json.loads(require('verification.json').read_text())


@router.get('/validation')
def validation():
    return FileResponse(require('validation.png'),media_type='image/png',headers={'Cache-Control':'no-store'})


@router.get('/smoothing-report')
def smoothing_report():
    return json.loads(require('smoothing-report.json').read_text())


@router.get('/calibration-readiness')
def calibration_readiness():
    return json.loads(require('calibration-readiness.json').read_text())


@router.get('/video')
def video():
    return FileResponse(require('mobile.mp4'),media_type='video/mp4')


@router.get('/bev-detail')
def bev_detail():
    return FileResponse(require('bev-detail.png'),media_type='image/png',headers={'Cache-Control':'no-store'})


@router.get('/bev-detail-video')
def bev_detail_video():
    return FileResponse(require('bev-detail.mp4'),media_type='video/mp4')


@router.get('/bev-display')
def bev_display():
    return json.loads(require('bev-detail.json').read_text())


@router.get('/bev-view',response_class=HTMLResponse)
def bev_view():
    require('bev-detail.png')
    return HTMLResponse('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>米制俯视图 · 原比例查看</title>
<style>body{margin:0;background:#10191f;color:#ecf2f5;font:16px system-ui}header{padding:16px}
button,input{margin:5px}button{padding:8px}#plane{overflow:auto;max-height:78vh;border-top:1px solid #789}
img{display:block;max-width:none;height:auto}a{color:#63dfca}p{margin:8px 0}</style>
<header><strong>俯视图 · 每米显示长度</strong>
<p>默认原比例显示，可横向拖动/滚动查看全部道路。横纵等比例，坐标不变。</p>
<label>缩放 <input id="zoom" type="range" min="0.25" max="2" step="0.25" value="1"></label>
<button id="native">原比例</button><button id="fit">适应屏幕</button><span id="scale" role="status"></span>
<p>1 米网格；显示放大不等于标定精度提高。<a href="bev-detail-video">查看原分辨率视频</a></p></header>
<div id="plane"><img id="map" src="bev-detail" alt="等比例米制俯视图，支持滚动和缩放"></div>
<script>
const map=document.getElementById('map'), plane=document.getElementById('plane');
const slider=document.getElementById('zoom'), status=document.getElementById('scale');
let metadata=null, scale=1;
function apply(s){scale=s;if(!map.naturalWidth)return;map.style.width=(map.naturalWidth*s)+'px';
status.textContent=metadata ? '当前每米约 '+(metadata.pixels_per_meter_x*s).toFixed(1)+' CSS像素' : '加载比例数据…';}
map.onload=()=>apply(scale);slider.oninput=()=>apply(Number(slider.value));
document.getElementById('native').onclick=()=>{slider.value='1';apply(1)};
document.getElementById('fit').onclick=()=>{if(map.naturalWidth)apply(plane.clientWidth/map.naturalWidth)};
fetch('bev-display').then(r=>{if(!r.ok)throw Error('比例数据读取失败');return r.json()})
.then(data=>{metadata=data;apply(scale)}).catch(e=>status.textContent=e.message);
if(map.complete)apply(scale);
</script></html>''')


@router.get('/frames/{index}')
def frame(index:int):
    if index<0:
        raise HTTPException(422,'negative frame index')
    with require('frames.jsonl').open() as f:
        for line in f:
            record=json.loads(line)
            if record['frame_index']==index:
                return record
    raise HTTPException(404,'frame outside processed clip')
