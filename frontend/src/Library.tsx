import { useEffect, useRef, useState } from 'react';
import { FolderOpen, Play, Pause, Upload, Download, Film, Layers3, Box, Camera } from 'lucide-react';
import './simulation.css';
import './library.css';

type Role='raw'|'annotation'|'bev'|'extra';
type Channel={file:string;label:string;sha256?:string};
type Group={id:string;schema:'carom-video-group-v1';name:string;kind:'real'|'carla';channels:Partial<Record<Role,Channel>>;note?:string};
const roles:Role[]=['raw','annotation','bev','extra'];
const titles:Record<Role,string>={raw:'原始视频',annotation:'算法标注',bev:'平面投影',extra:'SUMO / Unity / 其他'};
const icons={raw:Camera,annotation:Layers3,bev:Film,extra:Box};
function isGroup(x:unknown):x is Group {
 if(!x||typeof x!=='object')return false;
 const g=x as Record<string,unknown>;
 if(typeof g.id!=='string'||!/^[a-f0-9]{32}$/.test(g.id)||typeof g.name!=='string'||!['real','carla'].includes(String(g.kind))||!g.channels||typeof g.channels!=='object')return false;
 const c=g.channels as Record<string,unknown>;
 return ['raw','annotation','bev'].every(r=>r in c)&&Object.entries(c).every(([r,v])=>roles.includes(r as Role)&&!!v&&typeof v==='object'&&'file' in v&&v.file===`${r}.mp4`&&'label' in v&&typeof v.label==='string');
}
async function response(r:Response):Promise<unknown>{const d:unknown=await r.json();if(!r.ok)throw new Error(d&&typeof d==='object'&&'error'in d?String(d.error):`HTTP ${r.status}`);return d;}
function saved(id:string){try{const v:unknown=JSON.parse(localStorage.getItem(`carom.library.${id}`)||'{}');if(v&&typeof v==='object'){const x=v as Record<string,unknown>;return {time:typeof x.time==='number'&&Number.isFinite(x.time)?Math.max(0,x.time):0,rate:typeof x.rate==='number'&&[.5,1,2].includes(x.rate)?x.rate:1};}}catch{/* Ignore corrupt optional viewing preferences. */}return {time:0,rate:1};}

function Player({group}:{group:Group}){
 const [resume]=useState(()=>saved(group.id));
 const [time,setTime]=useState(resume.time),[rate,setRate]=useState(resume.rate),[playing,setPlaying]=useState(false);
 const [metadata,setMetadata]=useState<Partial<Record<Role,number>>>({}),[loaded,setLoaded]=useState<Partial<Record<Role,boolean>>>({}),[error,setError]=useState('');
 const videos=useRef<Partial<Record<Role,HTMLVideoElement>>>({}),generation=useRef(0),lastSaved=useRef(-1);
 const active=roles.filter(r=>!!group.channels[r]);
 const durations=active.map(r=>metadata[r]||0);
 const duration=durations.every(d=>d>0)?Math.min(...durations):0;
 const mismatch=duration>0&&Math.max(...durations)-duration>.25;
 function persist(t:number,r:number){try{localStorage.setItem(`carom.library.${group.id}`,JSON.stringify({time:t,rate:r}));}catch{/* Playback still works when storage is disabled. */}}
 function pause(){generation.current++;Object.values(videos.current).forEach(v=>v.pause());setPlaying(false);}
 function seek(t:number){pause();const next=Math.max(0,Math.min(t,duration));Object.values(videos.current).forEach(v=>{if(Number.isFinite(v.duration))v.currentTime=Math.min(next,Math.max(0,v.duration-.02));});setTime(next);persist(next,rate);}
 async function play(){if(playing){pause();return;}if(!duration||active.some(r=>!loaded[r])){setError('等待全部通道加载完成。推荐 H.264 编码的 MP4。');return;}
 const list=active.map(r=>videos.current[r]!);if(list.some(v=>v.error)){setError('存在解码失败的通道，请检查文件。');return;}
 if(time>=duration-.1)seek(0);setError('');const token=++generation.current;
 const results=await Promise.allSettled(list.map(v=>{v.playbackRate=rate;return v.play();}));
 if(token!==generation.current){list.forEach(v=>v.pause());return;}
 if(results.some(r=>r.status==='rejected')){pause();setError('某一路播放失败，已同步暂停。');}else setPlaying(true);}
 function clock(v:HTMLVideoElement){const t=v.currentTime;setTime(t);if(Math.floor(t)!==lastSaved.current){lastSaved.current=Math.floor(t);persist(t,rate);}if(!v.paused){if(t>=duration-.03){pause();return;}Object.values(videos.current).forEach(other=>{if(other!==v&&other.readyState>=2&&Math.abs(other.currentTime-t)>.15)other.currentTime=Math.min(t,other.duration-.02);});}}
 useEffect(()=>()=>{generation.current++;},[]);
 return <><section className="lib-group"><div><small>{group.kind==='real'?'REAL WORLD / 真实道路':'SIMULATION / CARLA'}</small><h1>{group.name}</h1></div><a className="lib-export" href={`/library/export/${group.id}`} download><Download size={16}/> 导出结果包</a></section>
 <div className="sim-grid">{roles.map(role=>{const channel=group.channels[role],Icon=icons[role];return <section className="sim-video" key={role}><h2><span><Icon size={15}/> {channel?.label||titles[role]}</span><small>{channel?`${time.toFixed(1)} s`:'未配置'}</small></h2>{channel?<video ref={v=>{if(v)videos.current[role]=v;else delete videos.current[role];}} src={`/library/media/${group.id}/${role}.mp4`} muted playsInline preload="auto" aria-label={channel.label}
 onLoadedMetadata={e=>{const v=e.currentTarget;setMetadata(m=>({...m,[role]:v.duration}));v.currentTime=Math.min(resume.time,Math.max(0,v.duration-.02));v.playbackRate=rate;}}
 onCanPlay={()=>setLoaded(s=>({...s,[role]:true}))} onWaiting={()=>{if(playing){pause();setError('视频正在缓冲，全部窗口已暂停；缓冲后点击同步播放。');}}}
 onTimeUpdate={e=>{if(role==='raw')clock(e.currentTarget);}} onEnded={pause} onError={()=>{pause();setLoaded(s=>({...s,[role]:false}));setError(`${channel.label} 加载或解码失败，请使用 H.264 MP4。`);}}/>:<div className="sim-placeholder"><Box size={32}/><b>三路结果组</b><small>第 4 路为可选，不补造 SUMO 或 Unity 结果</small><p>{group.note||'直接回放已处理视频，不重新运行检测。'}</p></div>}</section>;})}</div>
 <section className="sim-controls lib-timeline"><button className="sim-primary" onClick={()=>void play()} disabled={!duration||active.some(r=>!loaded[r])}>{playing?<Pause size={16}/>:<Play size={16}/>} {playing?'同步暂停':`同步播放 ${active.length} 路`}</button><button onClick={()=>seek(0)} disabled={!duration}>回到起点</button><span>{time.toFixed(1)} / {duration.toFixed(1)} s</span><input aria-label="结果组时间轴" type="range" min={0} max={duration||1} step={.1} value={Math.min(time,duration||1)} disabled={!duration} onChange={e=>seek(Number(e.target.value))}/><select aria-label="回放倍速" value={rate} onChange={e=>{const n=Number(e.target.value);setRate(n);Object.values(videos.current).forEach(v=>{v.playbackRate=n;});persist(time,n);}}><option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></section>
 {error&&<p role="alert" className="sim-error">{error}</p>}{mismatch&&<p role="status">各路时长不同，当前只播放共同区间 {duration.toFixed(1)} 秒。请确认素材来自同一起始帧。</p>}
 <p className="lib-caption">本地文件回放 · 位置自动保存 · 按共同时间轴同步（非逐帧精度保证） · 不加载 CARLA 真值或重新生成车辆</p></>;
}

export default function Library(){
 const [groups,setGroups]=useState<Group[]>([]),[selected,setSelected]=useState(()=>{try{return localStorage.getItem('carom.library.selected')||'';}catch{return '';}}),[filter,setFilter]=useState('all');
 const [error,setError]=useState(''),[notice,setNotice]=useState(''),[busy,setBusy]=useState(false),[form,setForm]=useState(false);
 const [name,setName]=useState(''),[kind,setKind]=useState<'real'|'carla'>('real'),[extra,setExtra]=useState('SUMO 回放'),[files,setFiles]=useState<Partial<Record<Role,File>>>({});
 const packageInput=useRef<HTMLInputElement>(null);
 async function refresh(){const d=await response(await fetch('/library/groups'));if(!d||typeof d!=='object'||!('groups'in d)||!Array.isArray(d.groups))throw new Error('素材库响应格式错误');setGroups(d.groups.filter(isGroup));}
 useEffect(()=>{let active=true;void fetch('/library/groups').then(response).then(d=>{if(active&&d&&typeof d==='object'&&'groups'in d&&Array.isArray(d.groups))setGroups(d.groups.filter(isGroup));}).catch(e=>{if(active)setError(String(e));});return()=>{active=false;};},[]);
 const visible=groups.filter(g=>filter==='all'||g.kind===filter);
 const current=visible.find(g=>g.id===selected)||visible[0];
 useEffect(()=>{if(current)try{localStorage.setItem('carom.library.selected',current.id);}catch{/* Optional preference only. */}},[current]);
 function select(id:string){setSelected(id);try{localStorage.setItem('carom.library.selected',id);}catch{/* Optional preference only. */}}
 async function post(path:string,body:BodyInit){return response(await fetch(path,{method:'POST',headers:{'X-CAROM-Client':'1'},body}));}
 async function importPackage(file:File){setBusy(true);setError('');setNotice('正在导入并校验结果包…');try{const r=await post('/library/import',file);await refresh();if(r&&typeof r==='object'&&'id'in r&&typeof r.id==='string'){setFilter('all');select(r.id);}setNotice('结果包已保存，之后打开直接回放。');}catch(e){setError(String(e));setNotice('');}finally{setBusy(false);}}
 async function importVideos(e:React.FormEvent){e.preventDefault();if(!files.raw||!files.annotation||!files.bev){setError('请为原始、标注、平面三路各选择一个 MP4 文件。');return;}setBusy(true);setError('');const id=crypto.randomUUID().replaceAll('-','');
 try{const channels:Partial<Record<Role,Channel>>={};for(const role of roles){const file=files[role];if(!file)continue;setNotice(`正在保存 ${titles[role]} · ${(file.size/1048576).toFixed(1)} MB…`);await post(`/library/upload/${id}/${role}`,file);channels[role]={file:`${role}.mp4`,label:role==='extra'?extra:titles[role]};}
 await post('/library/commit',JSON.stringify({id,manifest:{schema:'carom-video-group-v1',name:name.trim(),kind,channels,note:'用户导入的已处理视频。时间对应关系与算法准确性需由原处理结果验证。'}}));await refresh();setFilter('all');select(id);setForm(false);setFiles({});setName('');setNotice('整组视频已保存到客户端素材库。');}catch(e){setError(String(e));setNotice('');}finally{setBusy(false);}}
 return <div className="sim-shell lib-shell"><main><header><span>CAROM STUDIO / RESULT LIBRARY</span><b>本地持久保存 · 无需运行仿真</b></header><section className="lib-toolbar"><div><h2>真实与仿真 · 统一回放</h2><p>将已处理视频作为一组载入，保留原有检测结果。</p></div><button onClick={()=>{setForm(!form);setError('');}} disabled={busy}><FolderOpen size={16}/> 载入 3 / 4 路视频</button><button onClick={()=>packageInput.current?.click()} disabled={busy}><Upload size={16}/> 导入结果包</button><input ref={packageInput} aria-label="导入结果包文件" type="file" accept=".zip" hidden onChange={e=>{const file=e.target.files?.[0];if(file)void importPackage(file);e.target.value='';}}/></section>
 {error&&<div role="alert" className="sim-error">{error}</div>}{notice&&<p role="status">{notice}</p>}
 {form&&<form className="lib-import" onSubmit={e=>void importVideos(e)}><h3>创建视频组</h3><p>三路必选，第四路可选。请选择同一起始时刻的 H.264 MP4；文件会复制保存，不依赖原路径。</p><div className="lib-fields"><label>组名称<input value={name} maxLength={160} required onChange={e=>setName(e.target.value)} disabled={busy}/></label><label>素材类型<select value={kind} onChange={e=>setKind(e.target.value as 'real'|'carla')} disabled={busy}><option value="real">真实道路</option><option value="carla">CARLA 仿真</option></select></label></div>{roles.map(role=><label className="lib-file" key={role}>{titles[role]}{role==='extra'?'（可选）':'（必选）'}<input type="file" accept="video/mp4,.mp4" aria-label={`选择${titles[role]}`} required={role!=='extra'} disabled={busy} onChange={e=>setFiles(f=>({...f,[role]:e.target.files?.[0]}))}/></label>)}<label>第 4 路标题<input value={extra} maxLength={100} required disabled={busy} onChange={e=>setExtra(e.target.value)}/></label><button className="sim-primary" type="submit" disabled={busy||!name.trim()}>{busy?'保存中…':'保存并载入视频组'}</button><button type="button" disabled={busy} onClick={()=>setForm(false)}>取消</button></form>}
 <section className="sim-controls"><label>素材分组<select aria-label="素材分类" value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">全部</option><option value="real">真实道路</option><option value="carla">CARLA 仿真</option></select></label><label>已保存结果<select aria-label="已保存结果" value={current?.id||''} onChange={e=>select(e.target.value)}>{!visible.length&&<option value="">暂无结果</option>}{visible.map(g=><option key={g.id} value={g.id}>{g.name} · {Object.keys(g.channels).length} 路</option>)}</select></label><span>{visible.length} 组 · 视频保持原始比例</span></section>
 {current?<Player key={current.id} group={current}/>:<section className="sim-placeholder lib-empty"><FolderOpen size={40}/><h2>载入第一组处理结果</h2><p>支持真实道路和 CARLA 的三路／四路 MP4，或已导出的 .carom.zip 结果包。</p></section>}</main><aside><h2>CAROM</h2><p>统一结果工作台 · v1.0</p><button className="active">Offline / 素材库</button><a href="http://127.0.0.1:8766/?multi" target="_blank" rel="noreferrer">CARLA / 采集与处理 ↗</a><a href="http://127.0.0.1:5173/" target="_blank" rel="noreferrer">真实视频 / 原工作台 ↗</a><footer>结果回放独立于 WSL、CUDA 和 CARLA。<br/>处理入口需对应后台已启动。<br/>素材库保存在本机，不上传云端。</footer></aside></div>;
}

