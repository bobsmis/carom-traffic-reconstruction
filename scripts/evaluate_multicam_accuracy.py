"""Evaluator-only frozen-output audit. Never imported by perception/capture.

Uses instance-visible bounding boxes for image-space assignment, not metric
position matching. CARLA actor states are read here ONLY after inference.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import math
import csv
import html
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from evaluate_carla_proxy import iou

ROOT=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(a):
    a=np.asarray(a,float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,
        'median':float(np.median(a)) if len(a) else None,'p95':float(np.percentile(a,95)) if len(a) else None,
        'max':float(a.max()) if len(a) else None}
def lane(x):return int((x+14)//3.5) if -14<=x<14 else -1
def distance_bin(y):
    for a,b in [(0,30),(30,60),(60,100),(100,150)]:
        if a<=y<b:return f'{a}-{b} m'
    return 'outside 0-150 m'
def evaluate(run,output):
    output.mkdir(parents=True,exist_ok=False)
    rig=json.loads((run/'input/cameras.json').read_text());global_data=json.loads((run/'web/global.json').read_text())
    global_ids=[{(s['camera'],s['local_id']):o['id'] for o in f['objects'] for s in o['sources']} for f in global_data['frames']]
    totals=Counter();errors=defaultdict(list);details=[];camera_reports=[];hashes={str(run/'web/global.json'):digest(run/'web/global.json')};identity={};id_events=[]
    distance_counts=defaultdict(Counter)
    mapping_checks=[];global_truth=defaultdict(set);global_conflicts=0;crosscam_actors=defaultdict(set)
    for node in rig['cameras']:
        cid=node['id'];predpath=ROOT/'outputs'/('multi-'+run.name+'-'+cid)/'frames.jsonl'
        truthpath=run/'truth'/cid/'actors.jsonl';metapath=run/'input'/cid/'frames.jsonl'
        hashes[str(predpath)]=digest(predpath);hashes[str(truthpath)]=digest(truthpath);hashes[str(metapath)]=digest(metapath)
        pred=read(predpath);truth=read(truthpath);meta=read(metapath)
        if len(pred)!=len(truth) or len(pred)!=len(meta):raise ValueError('Frame counts disagree')
        c=Counter();ce=defaultdict(list);previous={}
        maskhash=hashlib.sha256()
        for index,(p,t,m) in enumerate(zip(pred,truth,meta)):
            if p['frame_index']!=index or t['carla_frame']!=m['carla_frame'] or abs(p['timestamp_s']-m['video_time_s'])>1e-6:raise ValueError('Frame alignment failed')
            maskpath=run/'truth'/cid/'masks'/f'{index:06}.png';raw=maskpath.read_bytes();maskhash.update(raw)
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            encoded=image[:,:,0].astype(np.uint16)*256+image[:,:,1].astype(np.uint16)
            values,counts=np.unique(encoded,return_counts=True);counts=dict(zip(values.tolist(),counts.tolist()))
            eligible=[];boxes=[]
            actor_ids=[o['actor_id']%65536 for o in t['objects']]
            if len(actor_ids)!=len(set(actor_ids)):raise ValueError('16-bit actor ID collision')
            for g in t['objects']:
                count=counts.get(g['actor_id']%65536,0)
                if count>0:c['visible_any_pixels']+=1
                if count<64:
                    if count>0:c['excluded_under_64_pixels']+=1
                    continue
                ys,xs=np.where(encoded==g['actor_id']%65536)
                box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],float)
                if min(box[2:]-box[:2])<4:c['excluded_under_4px_extent']+=1;continue
                # Check saved instance ID interpretation independently of predictions.
                uv=np.asarray(g.get('projected_cuboid_px'),dtype=float)
                if uv.shape!=(8,2) or not np.isfinite(uv).all():raise ValueError('Visible actor without valid projection')
                hull=cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
                footprint=np.zeros(encoded.shape,np.uint8);cv2.fillConvexPoly(footprint,hull,1)
                inside=float(footprint[ys,xs].mean());mapping_checks.append(inside)
                if inside<.80:raise ValueError(f'Instance ID/projection verification failed: {cid}/{index}/{g["actor_id"]}: {inside}')
                eligible.append(g);boxes.append(box)
                distance_counts[distance_bin(g['ground_center_road_xy'][1])]['eligible']+=1
            observed=[o for o in p['objects'] if o['observed']]
            c['eligible_visible_object_frames']+=len(eligible);c['observed_confirmed_object_frames']+=len(observed)
            c['frames']+=1
            if not observed or not eligible:continue
            scores=np.array([[iou(o['bbox_xyxy'],b) for b in boxes] for o in observed])
            cost=np.full((len(observed),len(eligible)+len(observed)),.7);cost[:,:len(eligible)]=1-scores
            rr,cc=linear_sum_assignment(cost)
            for r,col in zip(rr,cc):
                if col>=len(eligible) or scores[r,col]<.3:continue
                o=observed[r];g=eligible[col];actor=g['actor_id'];c['matched_object_frames']+=1
                distance_counts[distance_bin(g['ground_center_road_xy'][1])]['matched']+=1
                alternatives=np.r_[np.delete(scores[r],col),np.delete(scores[:,col],r),0]
                if scores[r,col]-alternatives.max()<.1:c['ambiguous_matches']+=1;continue
                c['strict_matches']+=1
                prev=identity.get((cid,actor));gid=global_ids[index].get((cid,o['id']))
                if prev and prev[1]!=o['id']:
                    c['consecutive_id_switches' if prev[0]==index-1 else 'reappearance_id_changes']+=1
                    id_events.append({'camera':cid,'frame':index,'actor':actor,'old_id':prev[1],'new_id':o['id'],'gap_frames':index-prev[0]-1})
                if prev and index-prev[0]>1:c['matched_track_gaps']+=1
                identity[cid,actor]=(index,o['id'])
                if gid is not None:
                    global_truth[gid].add(actor);crosscam_actors[actor].add(cid)
                if o.get('ground_xy') is None:c['matched_but_unlocalized']+=1;continue
                c['localized_strict_matches']+=1
                distance_counts[distance_bin(g['ground_center_road_xy'][1])]['localized']+=1
                delta=np.asarray(o['ground_xy'])-g['ground_center_road_xy'];size=np.asarray(o['dimensions_lwh']);gt_size=np.asarray(g['dimensions_lwh'])
                theta=o['heading_rad']-g['heading_rad'];angle=abs(math.degrees(math.atan2(math.sin(theta),math.cos(theta))))
                values={'lateral_abs_m':abs(delta[0]),'longitudinal_abs_m':abs(delta[1]),'position_m':float(np.linalg.norm(delta)),
                        'heading_abs_deg':angle,'length_abs_m':abs(size[0]-gt_size[0]),'width_abs_m':abs(size[1]-gt_size[1]),'height_abs_m':abs(size[2]-gt_size[2]),
                        'volume_relative_error_pct':abs(np.prod(size)/np.prod(gt_size)-1)*100}
                for k,v in values.items():ce[k].append(float(v));errors[k].append(float(v))
                true_lane=lane(g['ground_center_road_xy'][0]);pred_lane=lane(o['ground_xy'][0])
                c['lane_correct']+=int(true_lane==pred_lane and true_lane>=0);c['heading_reversed_over90']+=int(angle>90)
                c['position_within_1m']+=int(values['position_m']<=1);c['position_within_2m']+=int(values['position_m']<=2)
                last=previous.get(actor)
                if last and last[0]==index-1 and last[1]==o['id']:
                    step=float(np.linalg.norm(delta-last[2]));errors['error_step_m'].append(step);ce['error_step_m'].append(step)
                previous[actor]=(index,o['id'],delta)
                details.append({'camera':cid,'frame':index,'actor_id':actor,'track_id':o['id'],'global_id':gid,'image_iou':float(scores[r,col]),
                    'truth_longitudinal_m':g['ground_center_road_xy'][1],'true_lane_index':true_lane,'predicted_lane_index':pred_lane,
                    'error_lateral_m':float(delta[0]),'error_longitudinal_m':float(delta[1]),**values})
        c['unmatched_visible_object_frames']=c['eligible_visible_object_frames']-c['matched_object_frames']
        c['unmatched_observed_object_frames']=c['observed_confirmed_object_frames']-c['matched_object_frames']
        totals.update(c);camera_reports.append({'camera':cid,'counts':dict(c),'errors':{k:stats(v) for k,v in ce.items()},'instance_png_sequence_sha256':maskhash.hexdigest()})
    for path,value in hashes.items():
        if digest(Path(path))!=value:raise ValueError('Input changed during evaluation')
    c=totals;localized=c['localized_strict_matches'];eligible=c['eligible_visible_object_frames']
    ratios={'visible_matching_recall':c['matched_object_frames']/eligible,'confirmed_matching_precision':c['matched_object_frames']/c['observed_confirmed_object_frames'],
        'strict_localization_coverage':localized/eligible,'lane_accuracy_on_localized':c['lane_correct']/localized,
        'within_1m_on_localized':c['position_within_1m']/localized,'within_1m_coverage':c['position_within_1m']/eligible}
    report={'run':run.name,'duration_seconds':len(global_data['frames'])/global_data['fps'],'cameras':len(rig['cameras']),
        'protocol':{'min_visible_pixels':64,'min_visible_extent_px':4,'image_iou_min':.3,'strict_margin':.1,'predictions':'observed confirmed tracks only; not raw detector AP',
        'matching':'visible instance bbox, Hungarian; geometry errors only unambiguous matches','position_matching_used':False,'truth_used_for_inference':False,
        'instance_decoder':'B*256+G, verified against actor projected convex hull; no 16bit collisions','no_metric_outlier_removal':True},
        'counts':dict(c),'ratios':ratios,'errors':{k:stats(v) for k,v in errors.items()},'per_camera':camera_reports,
        'distance_bins':{k:dict(v) for k,v in distance_counts.items()},
        'instance_projection_containment':stats(mapping_checks),'global_ids_linked_to_multiple_truth_actors':{str(k):sorted(v) for k,v in global_truth.items() if len(v)>1},
        'truth_actors_matched_in_multiple_cameras':{str(k):sorted(v) for k,v in crosscam_actors.items() if len(v)>1},
        'source_hashes':hashes,'global_output_sha256':digest(run/'web/global.json'),'evaluator_sha256':digest(Path(__file__)),
        'limitations':['Same development scene, not held-out validation.','Only 6 seconds; no 3-minute/end-to-end identity acceptance.',
        'Instance association gates and small-object exclusion cause selection bias; counts retained.',
        'CARLA blueprint boxes are simulator geometric truth, not exact visible mesh volume.',
        'Predicted-only/coasting objects excluded from this accuracy protocol; no standard MOTA/IDF1/AP claim.',
        'No truth-fed inference; OS-level access isolation is not enforced.']}
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    (output/'id-events.json').write_text(json.dumps(id_events,indent=2))
    with (output/'matches.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    write_report(report,output)
    print(json.dumps({'counts':report['counts'],'ratios':ratios,'errors':report['errors']},indent=2))

def write_report(r,out):
    def pct(v):return f'{100*v:.2f}%'
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    labels={'lateral_abs_m':'横向绝对误差 (m)','longitudinal_abs_m':'纵向绝对误差 (m)','position_m':'平面位置误差 (m)',
    'heading_abs_deg':'朝向误差 (°)','length_abs_m':'长度误差 (m)','width_abs_m':'宽度误差 (m)','height_abs_m':'高度误差 (m)',
    'volume_relative_error_pct':'体积相对误差 (%)','error_step_m':'相邻帧定位误差变化 (m)'}
    lines=['# CAROM 多相机当前精度测量报告','',f'评估样本：{r["run"]}，10 路相机，每路 60 帧，10 FPS，6 秒。仅评估冻结的现有结果；三分钟新采集尚未执行。',
    '', '## 1. 评估边界与结论','',
    '**本报告是开发场景上的短片实测，不是正式高精度验收。SUMO 坐标回放的数值误差不等于视觉定位误差。**',
    '真值仅由独立评估脚本读取；没有反馈到检测、几何拟合或关联。所有输入哈希在评估结束时复核。',
    '', '## 2. 覆盖率（车辆×帧，非唯一车辆数）','',
    '| 指标 | 实测 |','|---|---:|',
    f'| 可见且达到像素阈值的真值对象帧 | {r["counts"]["eligible_visible_object_frames"]} |',
    f'| 小于 64 可见像素的对象帧（排除） | {r["counts"]["excluded_under_64_pixels"]} |',
    f'| 图像匹配召回率 | {pct(r["ratios"]["visible_matching_recall"])} |',
    f'| 已确认观测轨迹的图像匹配精确率 | {pct(r["ratios"]["confirmed_matching_precision"])} |',
    f'| 严格匹配且有定位的覆盖率 | {pct(r["ratios"]["strict_localization_coverage"])} |',
    f'| 成功定位样本中的 1 m 内比例 | {pct(r["ratios"]["within_1m_on_localized"])} |',
    f'| 对全部合格可见目标的 1 m 内覆盖率 | {pct(r["ratios"]["within_1m_coverage"])} |',
    f'| 定位样本车道正确率 | {pct(r["ratios"]["lane_accuracy_on_localized"])} |',
    '', '## 3. 几何精度（严格图像匹配且成功定位；保留全部离群值）','',
    '| 指标 | 样本数 | 均值 | 中位数 | P95 | 最大值 |','|---|---:|---:|---:|---:|---:|']
    for k,s in r['errors'].items():lines.append(f'| {labels[k]} | {s["n"]} | {fmt(s["mean"])} | {fmt(s["median"])} | {fmt(s["p95"])} | {fmt(s["max"])} |')
    lines+=['','相邻帧误差变化统计真实位置误差的一阶差分，只计连续同 ID 匹配；它混合测量噪声和几何偏差变化，不等同于单独的显示抖动指标。',
    '', '## 4. 每相机表现','', '| 相机 | 可见对象帧 | 严格定位帧 | 位置中位数 (m) | 位置 P95 (m) |','|---|---:|---:|---:|---:|']
    for cam in r['per_camera']:
        s=cam['errors'].get('position_m',{})
        lines.append(f'| {cam["camera"]} | {cam["counts"].get("eligible_visible_object_frames",0)} | {cam["counts"].get("localized_strict_matches",0)} | {fmt(s.get("median"))} | {fmt(s.get("p95"))} |')
    lines+=['','### 按相机局部道路纵向坐标分组（非相机斜距）','','| 纵向区段 | 可见对象帧 | 图像匹配帧 | 严格定位覆盖率 |','|---|---:|---:|---:|']
    for key in ['0-30 m','30-60 m','60-100 m','100-150 m','outside 0-150 m']:
        b=r['distance_bins'].get(key,{})
        if b.get('eligible'):lines.append(f'| {key} | {b["eligible"]} | {b.get("matched",0)} | {pct(b.get("localized",0)/b["eligible"])} |')
    lines+=['','## 5. 身份与跨镜','',f'- 连续帧局部 ID 变化：{r["counts"].get("consecutive_id_switches",0)} 次。',
    f'- 匹配中断后重新出现且 ID 变化：{r["counts"].get("reappearance_id_changes",0)} 次。',
    f'- 一个全局 ID 在本协议下对应多个真值车辆：{len(r["global_ids_linked_to_multiple_truth_actors"])} 个。',
    f'- 在多个相机中获得严格匹配的真值车辆：{len(r["truth_actors_matched_in_multiple_cameras"])} 辆。',
    '- 没有足够的长线交接样本；零次切换不表示跨镜跟踪成功。不报告未经完整协议验证的 IDF1/MOTA。',
    '', '## 6. 方法与限制','',
    '- 先从实例图取得可见像素框：至少 64 像素、边长至少 4 像素。匹配采用图像 IoU≥0.3 的匈牙利分配；几何误差再要求候选间隔≥0.1。匹配不使用预测的米制位置。',
    '- 实例 ID 编码由 B×256+G 读取，逐目标检查可见像素落在该 actor 投影凸包内至少 80%，且无 16 位 actor ID 冲突。该一致性校验增强可信度，但不是独立人工标注。',
    '- 车道按 8 条 3.5 m 车道的真值/估计中心所属区间比较；变道压线时属于中心车道定义，不代表违规判断。',
    '- 统计对象是已确认且本帧观测到的轨迹，不是原始检测器 AP；纯预测续航目标未纳入本轮精度。遮挡后重现、显示闪烁和长期误关联仍需长视频专项测试。',
    '- 匹配成功样本的误差存在选择偏差，所以同时报告覆盖率。当前数据为开发场景，不是独立测试集。',
    '- 体积基准为 CARLA 蓝图包围盒，其尺寸可能包含附件或与可见车身不同；尺寸质量门限通过也不等于尺寸精确。',
    '', '## 7. 后续验收','',
    '优先改善纵向定位与尺寸退化，再检验车道/朝向；冻结算法后录制 180 秒、覆盖入口至出口及多相机交接。另设未调参的随机种子、亮度和密度，统计速度、IDF1、遮挡恢复和预测续航误差。',
    '', '## 8. 复核资料','',
    '- report.json：完整统计、阈值与 SHA256。','- matches.csv：逐匹配对象几何误差。','- id-events.json：身份变化事件。',
    '- evaluator.py：本次评估脚本副本。',
    '- [CARLA 实例分割官方说明](https://carla.readthedocs.io/en/latest/tuto_G_instance_segmentation_sensor/)：实例信息位于 G/B 通道，语义信息位于 R 通道。']
    text='\n'.join(lines);(out/'报告.md').write_text(text,encoding='utf-8')
    body=[];table=False
    for line in lines:
        if line.startswith('|'):
            if not table:body.append('<table>');table=True
            if set(line.replace('|','').replace(':','').replace('-','').replace(' ','')):body.append('<tr>'+''.join('<td>'+html.escape(v.strip())+'</td>' for v in line.strip('|').split('|'))+'</tr>')
            continue
        if table:body.append('</table>');table=False
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));body.append(f'<h{level}>'+html.escape(line.lstrip('# '))+f'</h{level}>')
        elif line:body.append('<p>'+html.escape(line).replace('**','')+'</p>')
    if table:body.append('</table>')
    (out/'报告.html').write_text('<!doctype html><meta charset="utf-8"><title>CAROM精度报告</title><style>body{max-width:1100px;margin:40px auto;padding:24px;font:16px/1.65 system-ui;color:#243044;background:#fafafa}table{border-collapse:collapse;width:100%;font-size:14px}td{padding:8px;border:1px solid #cbd5e1}tr:first-child{background:#e2e8f0;font-weight:600}h1,h2{color:#16365b} @media print{body{margin:0;background:white}tr{break-inside:avoid}}</style>'+''.join(body),encoding='utf-8')
    (out/'evaluator.py').write_bytes(Path(__file__).read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();evaluate(a.run,a.output)
