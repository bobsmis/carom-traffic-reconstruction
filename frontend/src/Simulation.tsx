import { useEffect, useState } from 'react';
import './simulation.css';
type Run={id:string;completed:boolean;frames:number;processed:boolean};
type Status={busy:boolean;action:string|null;error:string|null;run:string|null;runs:Run[];log:string};
type Event={type:string;actor_id?:number;other_actor_id?:number;carla_frame?:number;impulse?:number};
export default function Simulation(){
 const [state,setState]=useState<Status|null>(null),[error,setError]=useState('');
 const [seconds,setSeconds]=useState(20),[seed,setSeed]=useState(73);
 const [changes,setChanges]=useState(true),[collision,setCollision]=useState(true);
 const [selected,setSelected]=useState(''),[tab,setTab]=useState('online'),[tick,setTick]=useState(0);
 const [events,setEvents]=useState<Event[]>([]);
 useEffect(()=>{let active=true;async function refresh(){try{
  const r=await fetch('/sim-api/status');if(!r.ok)throw new Error(`服务 HTTP ${r.status}`);
  const s:Status=await r.json();if(active){setState(s);setTick(t=>t+1);}
  const e=await fetch('/sim-api/events');if(e.ok&&active)setEvents(await e.json() as Event[]);
 }catch(e){if(active)setError(e instanceof Error?e.message:'连接失败');}}
 void refresh();const timer=setInterval(()=>void refresh(),2000);return()=>{active=false;clearInterval(timer);};},[]);
 const run=selected||state?.run||'',ready=state?.runs.find(r=>r.id===run)?.processed;
 async function action(name:string){setError('');try{
  const r=await fetch(`/sim-api/${name}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seconds,seed,lane_changes:changes,collision,run})});
  const data:{error?:string}=await r.json();if(!r.ok)throw new Error(data.error||'任务失败');
  if(name!=='stop')setState(s=>s?{...s,busy:true,action:name,error:null}:s);
 }catch(e){setError(e instanceof Error?e.message:'请求失败');}}
 return <div className="sim-shell"><main><header><span>CAROM / SIMULATION LAB</span><b>{state?.busy?`执行中 · ${state.action}`:'本地控制台'}</b></header>
 <section className="sim-title"><small>FIXED CAMERA · TUNNEL BENCHMARK</small><h1>隧道交通仿真实验室</h1><p>真实仿真画面 → 本地感知 → 平面重建。动态真值仅供评估。</p><a href="/?multi">进入 1,240 米多相机长隧道工作台 →</a></section>
 {(error||state?.error)&&<div role="alert" className="sim-error">{error||state?.error}</div>}
 <section className="sim-controls"><label>时长 / 秒<input type="number" min={3} max={120} value={seconds} onChange={e=>setSeconds(Number(e.target.value))}/></label><label>种子<input type="number" min={0} value={seed} onChange={e=>setSeed(Number(e.target.value))}/></label><label><input type="checkbox" checked={changes} onChange={e=>setChanges(e.target.checked)}/>变道</label><label><input type="checkbox" checked={collision} onChange={e=>setCollision(e.target.checked)}/>碰撞测试</label>
 <button disabled={!state||state.busy} onClick={()=>void action('launch')}>启动 CARLA</button><button className="sim-primary" disabled={!state||state.busy} onClick={()=>void action('capture')}>搭建并采集</button><button disabled={!state?.busy||state.action!=='capture'} onClick={()=>void action('stop')}>停止采集</button></section>
 <section className="sim-controls"><select aria-label="采集记录" value={run} onChange={e=>setSelected(e.target.value)}><option value="">选择采集记录</option>{state?.runs.map(r=><option key={r.id} value={r.id}>{r.id} · {r.frames} 帧 · {r.processed?'已处理':r.completed?'已采集':'未完成'}</option>)}</select><button disabled={!run||state?.busy||ready} onClick={()=>void action('process')}>运行感知与重建</button><small>只读取 input/，不使用车辆真值补漏检。</small></section>
 {tab==='events'?<section className="sim-events"><h2>最近采集的仿真事件</h2><p>请求不代表发生；collision_sensor 才表示物理接触。</p><table><thead><tr><th>类型</th><th>车辆</th><th>对方</th><th>仿真帧</th><th>冲量</th></tr></thead><tbody>{events.map((e,i)=><tr key={i}><td>{e.type}</td><td>{e.actor_id??'—'}</td><td>{e.other_actor_id??'—'}</td><td>{e.carla_frame??'—'}</td><td>{e.impulse?.toFixed(1)??'—'}</td></tr>)}</tbody></table>{!events.length&&<p>暂无记录</p>}</section>:<div className="sim-grid">{[['raw','原始 RGB'],['annotation','算法标注'],['bev','米制俯视图'],['unity','Unity · 待接入']].map(([id,title])=><section className="sim-video" key={id}><h2>{title}<small>{id==='raw'&&tab==='online'?'采集预览 · 2 秒刷新':ready&&id!=='unity'?'离线输出':'等待输入'}</small></h2>{id==='raw'&&tab==='online'?<img src={`/sim-api/live.jpg?t=${tick}`} alt="CARLA RGB 采集预览；采集前暂无画面"/>:ready&&id!=='unity'?<video key={`${run}-${id}`} src={`/sim-media/${run}/${id}.mp4`} controls muted playsInline preload="metadata"/>:<div className="sim-placeholder">{id==='unity'?'预留 Unity 通道':'采集与算法处理完成后显示'}<small>不使用真值替代算法输出</small></div>}</section>)}</div>}
 <details className="sim-log"><summary>后台任务日志</summary><pre>{state?.log||'等待任务启动'}</pre></details></main>
 <aside><h2>CAROM</h2><p>仿真工作区</p>{[['online','Online / 采集'],['offline','Offline / 分析'],['events','事故故障记录'],['replay','回放']].map(([id,label])=><button className={tab===id?'active':''} key={id} onClick={()=>setTab(id)}>{label}</button>)}<a href="http://localhost:5173/">原工作台 ↗</a><footer>LOCAL ONLY<br/>控制服务 8766<br/>CARLA 2000 / TM 8050</footer></aside></div>;
}
