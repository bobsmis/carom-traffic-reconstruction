import { useEffect, useMemo, useRef, useState } from 'react';
import './simulation.css';
type Run={id:string;schema?:string;completed:boolean;processed:boolean;frames:number};
type Status={api_version?:number;busy:boolean;action:string|null;error:string|null;run:string|null;runs:Run[];log:string};
type Node={id:string;station_m:number;global_offset_m:number};
type Track={id:number;cameras:string[];xy:number[];status:string;class_name:string};
type GlobalData={fps:number;frames:{time_s:number;objects:Track[]}[]};
const streams=[['raw','节点原始 RGB'],['annotation','全局 ID 标注'],['bev','局部放大平面 · 60 m'],['sumo','SUMO · 视觉轨迹回放']] as const;
const resumeKey='carom.multicam.resume.v1';
function readResume(){
 const defaults={run:'',camera:'cam01',live:false,time:0,rate:1};
 try{const x:unknown=JSON.parse(localStorage.getItem(resumeKey)||'null');
 if(!x||typeof x!=='object')return defaults;
 const p=x as Record<string,unknown>;
 return {run:typeof p.run==='string'&&/^\d{8}-\d{6}$/.test(p.run)?p.run:'',
 camera:typeof p.camera==='string'&&/^cam(0[1-9]|10)$/.test(p.camera)?p.camera:'cam01',
 live:typeof p.live==='boolean'?p.live:false,
 time:typeof p.time==='number'&&Number.isFinite(p.time)&&p.time>=0?p.time:0,
 rate:typeof p.rate==='number'&&[.5,1,2].includes(p.rate)?p.rate:1};
 }catch{return defaults;}
}
export default function MultiSimulation(){
 const [resume]=useState(readResume);
 const [state,setState]=useState<Status|null>(null),[error,setError]=useState(''),[selected,setSelected]=useState(resume.run);
 const [nodes,setNodes]=useState<Node[]>([]),[data,setData]=useState<GlobalData|null>(null),[camera,setCamera]=useState(resume.camera);
 const [live,setLive]=useState(resume.live),[tick,setTick]=useState(0),[seconds,setSeconds]=useState(180),[brightness,setBrightness]=useState(70);
 const [follow,setFollow]=useState<number|null>(null),[time,setTime]=useState(resume.time),[playing,setPlaying]=useState(false),[rate,setRate]=useState(resume.rate);
 const videos=useRef<Record<string,HTMLVideoElement|null>>({});
 const runs=state?.runs.filter(r=>r.schema==='multicam-v1'&&r.completed)||[];
 const upgraded=state?.api_version===3;
 const run=selected||runs[0]?.id||'';const ready=runs.find(r=>r.id===run)?.processed;
 const savedTime=Math.floor(time);
 useEffect(()=>{if(!run)return;try{localStorage.setItem(resumeKey,JSON.stringify({run,camera,live,time:savedTime,rate}));}catch{/* Browser storage can be unavailable; playback remains functional. */}},[run,camera,live,savedTime,rate]);
 useEffect(()=>{let active=true;async function poll(){try{const r=await fetch('/sim-api/status');if(!r.ok)throw new Error('控制服务未连接');const s:Status=await r.json();if(active){setState(s);setTick(t=>t+1);}}catch(e){if(active)setError(String(e));}}
 void poll();const timer=setInterval(()=>void poll(),2000);return()=>{active=false;clearInterval(timer);};},[]);
 useEffect(()=>{if(!run)return;let active=true;async function load(){try{
 const r=await fetch(`/sim-data/${run}/cameras.json`);if(!r.ok)throw new Error('相机标定目录未就绪');const rig:{cameras:Node[]}=await r.json();if(active)setNodes(rig.cameras);
 if(ready){const g=await fetch(`/sim-data/${run}/global.json`);if(!g.ok)throw new Error('全局关联结果尚未完成');const d:GlobalData=await g.json();if(active)setData(d);}else if(active)setData(null);
 }catch(e){if(active)setError(String(e));}}void load();return()=>{active=false;};},[run,ready]);
 const [rawDuration,setRawDuration]=useState(0);
 const duration=data?data.frames.length/data.fps:rawDuration;
 const tracks=data?.frames[Math.min(data.frames.length-1,Math.floor(time*data.fps))]?.objects||[];
 const ids=useMemo(()=>data?Array.from(new Set(data.frames.flatMap(f=>f.objects.map(o=>o.id)))).sort((a,b)=>a-b):[],[data]);
 function pause(){Object.values(videos.current).forEach(v=>v?.pause());setPlaying(false);}
 function seek(t:number){setTime(t);Object.values(videos.current).forEach(v=>{if(v&&Number.isFinite(v.duration))v.currentTime=Math.min(t,Math.max(0,v.duration-.01));});}
 async function play(){if(playing){pause();return;}const list=Object.values(videos.current).filter((v):v is HTMLVideoElement=>!!v&&v.readyState>=2);
 if(list.length!==(ready?4:1)){setError('等待视频加载完成后再播放');return;}if(time>=duration-.1)seek(0);
 const results=await Promise.allSettled(list.map(v=>{v.playbackRate=rate;return v.play();}));
 if(results.some(r=>r.status==='rejected')){pause();setError('部分视频播放失败，已暂停全部窗口');}else setPlaying(true);}
 function clock(v:HTMLVideoElement){const t=v.currentTime;setTime(t);Object.values(videos.current).forEach(other=>{if(other&&other!==v&&other.readyState>=2&&Math.abs(other.currentTime-t)>.15)other.currentTime=t;});
 if(follow!==null&&data){const obj=data.frames[Math.min(data.frames.length-1,Math.floor(t*data.fps))]?.objects.find(o=>o.id===follow);if(obj?.cameras.length&&!obj.cameras.includes(camera))setCamera(obj.cameras[0]);}}
 async function action(name:string){setError('');try{const r=await fetch(`/sim-api/${name}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seconds,seed:83,lane_changes:true,collision:false,brightness,run})});const d:{error?:string}=await r.json();if(!r.ok)throw new Error(d.error||'请求失败');if(!['brightness','stop'].includes(name))setState(s=>s?{...s,busy:true,action:name,error:null}:s);}catch(e){setError(String(e));}}
 return <div className="sim-shell"><main><header><span>CAROM / MULTI-CAMERA LAB</span><b>{state?.busy?`执行中 · ${state.action}`:'本地多相机控制台'}</b></header>
 <section className="sim-title"><small>1,240 M · 10 CHECKPOINTS · RGB ONLY</small><h1>长隧道联合监控</h1><p>180 秒入口生成 · 局部等比例放大 · SUMO 轨迹适配。跨镜 ID 为实验关联，尚未通过长线精度验收。</p></section>
 {state&&!upgraded&&<div className="sim-error" role="status">控制服务仍是旧版本；重启后启用 180 秒采集和 SUMO 视频通道。现有素材仍为 6 秒验证片。</div>}
 {(error||state?.error)&&<div className="sim-error" role="alert">{error||state?.error}<button onClick={()=>setError('')}>关闭提示</button></div>}
 <section className="sim-controls"><label>采集秒数<input type="number" min={3} max={180} value={seconds} onChange={e=>setSeconds(Number(e.target.value))}/></label><button disabled={!state||state.busy} onClick={()=>void action('launch')}>启动 CARLA</button><button disabled={!state||state.busy||!upgraded} className="sim-primary" onClick={()=>void action('capture-multi')}>搭建并采集 10 个节点</button><button disabled={!state?.busy||state.action!=='capture-multi'} onClick={()=>void action('stop')}>停止采集</button><label>环境亮度 {brightness}<input aria-label="环境亮度" type="range" min={0} max={100} value={brightness} onChange={e=>setBrightness(Number(e.target.value))}/></label><button onClick={()=>void action('brightness')}>应用亮度</button><small>控制 CARLA 环境光/天气；采集中生效。不是隧道灯具功率，也不修改已录视频。</small></section>
 <section className="sim-controls"><select aria-label="多相机采集记录" value={run} onChange={e=>{pause();setSelected(e.target.value);setTime(0);setFollow(null);}}><option value="">选择采集</option>{runs.map(r=><option key={r.id} value={r.id}>{r.id} · {r.frames} 帧/相机 · {r.processed?'已处理':'已采集'}</option>)}</select><button disabled={!run||state?.busy||ready} onClick={()=>void action('process-multi')}>{state?.busy&&state.action==='process-multi'?'正在处理全部节点…':'处理全部节点并关联'}</button><small>当前全局检测为离线处理，不伪装为实时推理。</small></section>
 <section className="sim-controls"><label>检查节点<select aria-label="检查节点" value={camera} onChange={e=>{setFollow(null);setCamera(e.target.value);}}>{(nodes.length?nodes:Array.from({length:10},(_,i)=>({id:`cam${String(i+1).padStart(2,'0')}`,station_m:i*124,global_offset_m:i*124}))).map(n=><option key={n.id} value={n.id}>{n.id} · {n.station_m} m</option>)}</select></label>
 <label>跟随全局 ID<select aria-label="跟随全局 ID" value={follow??''} disabled={!data||live} onChange={e=>setFollow(e.target.value?Number(e.target.value):null)}><option value="">不跟随</option>{ids.map(id=><option value={id} key={id}>G{id}</option>)}</select></label><span>{follow===null?`本帧 ${tracks.length} 个已定位对象`:tracks.some(o=>o.id===follow)?`G${follow} 已定位`:`G${follow} 本帧未定位，不补造位置`}</span></section>
 <div className="sim-grid">{streams.map(([id,title])=><section className="sim-video" key={id}><h2>{title}<small>{live?'抽帧预览 / 等待离线结果':`${camera} · ${time.toFixed(1)} s`}</small></h2>{live&&id==='raw'?<img src={`/sim-api/${camera}.jpg?t=${tick}`} alt="当前相机最近采集画面"/>:!live&&!!run&&(id==='raw'||(ready&&(id!=='sumo'||upgraded)))?<video key={`${run}-${id}-${camera}`} ref={el=>{videos.current[id]=el;}} src={`/sim-media/${run}/${camera+'/'+id}.mp4`} muted playsInline controls={!ready} preload="metadata" onLoadedMetadata={e=>{if(id==='raw')setRawDuration(e.currentTarget.duration);e.currentTarget.currentTime=Math.min(time,e.currentTarget.duration-.01);e.currentTarget.playbackRate=rate;}} onCanPlay={e=>{if(playing)void e.currentTarget.play().catch(()=>pause());}} onTimeUpdate={e=>{if(id==='raw')clock(e.currentTarget);}} onEnded={()=>pause()} onError={()=>{pause();setError(`${title} 加载失败`);}}/>:<div className="sim-placeholder">{live?'切换离线回放查看算法结果':!upgraded?'请重启控制服务以启用 SUMO 通道':state?.busy&&state.action==='process-multi'?'正在处理视频并生成此通道，请勿重复启动；进度见后台任务日志':'采集已完成，请先处理全部节点以生成此通道'}<small>不读取车辆真值生成结果</small></div>}</section>)}</div>
 <section className="sim-controls"><button disabled={live||!run||(!ready&&!rawDuration)||(ready&&!upgraded)} onClick={()=>void play()}>{playing?(ready?'同步暂停四窗':'暂停原始视频'):(ready?'同步播放四窗':'播放原始视频')}</button><button disabled={live||!run||(!ready&&!rawDuration)||(ready&&!upgraded)} onClick={()=>{pause();seek(0);}}>回到起点</button><label>{time.toFixed(1)} / {duration.toFixed(1)} s<input aria-label="同步回放时间轴" type="range" min={0} max={duration||1} step={.1} value={Math.min(time,duration||1)} disabled={live||!run||(!ready&&!rawDuration)||(ready&&!upgraded)} onChange={e=>seek(Number(e.target.value))}/></label><select aria-label="同步播放速度" value={rate} onChange={e=>{const r=Number(e.target.value);setRate(r);Object.values(videos.current).forEach(v=>{if(v)v.playbackRate=r;});}}><option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></section>
 <p>观看位置、素材、摄像头和倍速自动保存在此浏览器，刷新后恢复并暂停。处理结果已保存在本机，无需重新处理。</p><details className="sim-log"><summary>后台任务日志</summary><pre>{state?.log}</pre></details></main><aside><h2>CAROM</h2><p>长隧道工作区</p><button className={live?'active':''} onClick={()=>{pause();setLive(true);}}>Online / 节点预览</button><button className={!live?'active':''} onClick={()=>{pause();setLive(false);}}>Offline / 联动回放</button><a href="http://127.0.0.1:8767/?library">统一结果库 / 客户端 ↗</a><a href="/">单相机 / 事故记录</a><footer>SUMO 回放视觉估计，非真值<br/>Unity 尚未接入<br/>在线检测吞吐量待验证</footer></aside></div>;
}

