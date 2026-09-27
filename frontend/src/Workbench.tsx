import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowDownToLine, ArrowLeft, ArrowRight, Box, Camera, Check, ChevronRight, CircleHelp, Clock3, FileVideo, FolderOpen, Grid2X2, History, Layers3, Link2, Maximize2, Pause, Play, Plus, Radio, RotateCcw, Settings2, ShieldAlert, Video, X } from 'lucide-react';
import './workbench.css';

type Page = 'online' | 'offline' | 'incidents' | 'replay';
type Mode = 'online' | 'offline';
type Channel = 'raw' | 'annotated' | 'bev' | 'unity';
type Sources = Record<Channel, string>;
type Incident = { id: string; title: string; kind: string; time: string; position: number; status: '待处理' | '已处理'; mode: Mode };
const channels = [
  { id: 'raw', title: '原始视频', english: 'ORIGINAL', icon: Camera, description: '未经标注的相机画面' },
  { id: 'annotated', title: '标注视频', english: 'ANNOTATION', icon: Layers3, description: '实例分割 · 全局 ID · 体积框' },
  { id: 'bev', title: '平面视频', english: 'BIRD’S-EYE VIEW', icon: Grid2X2, description: '地面投影 · 车辆位置 · 轨迹' },
  { id: 'unity', title: 'Unity 视频', english: 'DIGITAL RECONSTRUCTION', icon: Box, description: '三维场景与交通流重建' },
] satisfies { id: Channel; title: string; english: string; icon: typeof Camera; description: string }[];
const empty: Sources = { raw: '', annotated: '', bev: '', unity: '' };
const demo: Sources = { raw: '/demo/original.mp4', annotated: '/demo/annotated.mp4', bev: '/demo/bev.mp4', unity: '' };
const nav = [
  { id: 'online', title: 'Online', sub: '实时监测', icon: Radio },
  { id: 'offline', title: 'Offline', sub: '离线分析', icon: FileVideo },
  { id: 'incidents', title: '事故故障记录', sub: '事件与异常', icon: ShieldAlert },
  { id: 'replay', title: '回放', sub: '历史片段', icon: History },
] satisfies { id: Page; title: string; sub: string; icon: typeof Camera }[];
function route(): Page { const value = location.hash.slice(1); return nav.some(n => n.id === value) ? value as Page : 'offline'; }
function stamp(t: number) { const n = Number.isFinite(t) ? Math.max(0, t) : 0; return `${Math.floor(n / 60).toString().padStart(2, '0')}:${Math.floor(n % 60).toString().padStart(2, '0')}.${Math.floor(n % 1 * 10)}`; }
function storedSources(mode: Mode): Sources {
  try { const value: unknown = JSON.parse(localStorage.getItem(`carom.sources.${mode}`) || 'null');
    if (value && typeof value === 'object' && channels.every(c => c.id in value && typeof (value as Record<string, unknown>)[c.id] === 'string')) return value as Sources;
  } catch { /* An invalid saved configuration falls back to the documented defaults. */ }
  return mode === 'offline' ? demo : empty;
}
function storedIncidents(): Incident[] {
  try { const data: unknown = JSON.parse(localStorage.getItem('carom.incidents') || '[]');
    if (Array.isArray(data)) return data.filter((r): r is Incident => r && typeof r.id === 'string' && typeof r.title === 'string' && typeof r.kind === 'string' && typeof r.time === 'string' && typeof r.position === 'number' && ['待处理', '已处理'].includes(r.status) && ['online', 'offline'].includes(r.mode));
  } catch { /* Corrupt local data is not treated as a real incident. */ } return [];
}

export default function Workbench() {
  const [page, setPage] = useState<Page>(route);
  const mode: Mode = page === 'online' ? 'online' : 'offline';
  const [configs, setConfigs] = useState<Record<Mode, Sources>>(() => ({ online: storedSources('online'), offline: storedSources('offline') }));
  const sources = configs[mode];
  const videos = useRef<Partial<Record<Channel, HTMLVideoElement>>>({});
  const [states, setStates] = useState<Partial<Record<Channel, string>>>({});
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [notice, setNotice] = useState('');
  const [settings, setSettings] = useState(false);
  const [draft, setDraft] = useState<Sources>(empty);
  const [formError, setFormError] = useState('');
  const [focused, setFocused] = useState<Channel | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>(storedIncidents);
  const [eventForm, setEventForm] = useState(false);
  const [eventTitle, setEventTitle] = useState('');
  const [eventKind, setEventKind] = useState('跟踪异常');
  const [filter, setFilter] = useState('全部');
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const dialogRef = useRef<HTMLDialogElement>(null);
  const eventRef = useRef<HTMLDialogElement>(null);
  const first = channels.find(c => sources[c.id])?.id;
  useEffect(() => { const change = () => setPage(route()); window.addEventListener('hashchange', change); return () => window.removeEventListener('hashchange', change); }, []);
  useEffect(() => { if (settings) dialogRef.current?.showModal(); else dialogRef.current?.close(); }, [settings]);
  useEffect(() => { if (eventForm) eventRef.current?.showModal(); else eventRef.current?.close(); }, [eventForm]);
  function pause() { Object.values(videos.current).forEach(v => v?.pause()); setPlaying(false); }
  function navigate(next: Page) { pause(); setTime(0); setDuration(0); setStates({}); setFocused(null); location.hash = next; setPage(next); }
  function seek(value: number) { setTime(value); Object.values(videos.current).forEach(v => { if (v && Number.isFinite(v.duration)) v.currentTime = Math.min(value, v.duration); }); }
  async function play() {
    if (playing) { pause(); return; }
    const active = Object.values(videos.current).filter((v): v is HTMLVideoElement => !!v && v.readyState >= 2 && !v.error);
    if (!active.length) { setNotice('还没有可播放的视频，请先配置视频源或等待加载。'); return; }
    if (duration > 0 && time >= duration - .05) seek(0);
    const result = await Promise.allSettled(active.map(v => { v.playbackRate = rate; return v.play(); }));
    setPlaying(result.some(r => r.status === 'fulfilled'));
    if (result.some(r => r.status === 'rejected')) setNotice('部分窗口播放失败，请检查视频格式、地址或浏览器权限。');
  }
  function clock(id: Channel, v: HTMLVideoElement) {
    if (id !== first) return;
    setTime(v.currentTime);
    if (mode === 'offline' && !v.paused) Object.entries(videos.current).forEach(([key, other]) => { if (key !== id && other && other.readyState >= 2 && Number.isFinite(other.duration) && Math.abs(other.currentTime - v.currentTime) > .2) other.currentTime = Math.min(v.currentTime, other.duration); });
  }
  function openSettings() { setDraft({ ...sources }); setFormError(''); setSettings(true); }
  function saveSources(e: React.FormEvent) {
    e.preventDefault(); const next = Object.fromEntries(Object.entries(draft).map(([k, v]) => [k, v.trim()])) as Sources;
    for (const value of Object.values(next)) if (value) {
      try { const url = new URL(value, location.origin); if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error(); }
      catch { setFormError('请输入 HTTP(S) 视频地址或本站相对路径，不要在地址中填写账号密码。'); return; }
    }
    try { localStorage.setItem(`carom.sources.${mode}`, JSON.stringify(next)); } catch { setFormError('浏览器存储不可用，配置尚未保存。'); return; }
    pause(); setConfigs(old => ({ ...old, [mode]: next })); setStates({}); setTime(0); setDuration(0); setSettings(false); setNotice('视频源配置已保存到本机浏览器。');
  }
  function persistEvents(next: Incident[]) { try { localStorage.setItem('carom.incidents', JSON.stringify(next)); setIncidents(next); return true; } catch { setNotice('浏览器存储不可用，事件尚未保存。'); return false; } }
  function addEvent(e: React.FormEvent) { e.preventDefault(); if (!eventTitle.trim()) return;
    if (persistEvents([{ id: crypto.randomUUID(), title: eventTitle.trim(), kind: eventKind, time: new Date().toISOString(), position: time, status: '待处理', mode }, ...incidents])) { setEventForm(false); setEventTitle(''); setNotice('已添加手动记录，不代表算法自动判定事故。'); }
  }
  function exportEvents() { const url = URL.createObjectURL(new Blob([JSON.stringify(incidents, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = 'carom-events.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
  const visibleEvents = incidents.filter(r => (filter === '全部' || r.status === filter) && `${r.title} ${r.kind}`.includes(query));
  const currentNav = nav.find(n => n.id === page)!;
  return <div className="wb">
    <main className="wb-workspace">
      <header className="wb-top"><div><FolderOpen size={17} /><span>CAROM 工作区</span><ChevronRight size={14} /><b>{currentNav.title}</b></div><div><a href="http://127.0.0.1:8766/">仿真实验室</a><span className="wb-local"><i />本地工作区</span><a href="?evidence" title="查看旧版算法核验页"><CircleHelp size={18} /></a></div></header>
      <div className="wb-content">
        <section className="wb-heading"><div><div className="wb-eyebrow">TRAFFIC RECONSTRUCTION WORKSPACE</div><h1>{page === 'online' ? '实时交通观测' : page === 'incidents' ? '事故与故障记录' : page === 'replay' ? '回放工作台' : '离线场景分析'}<span>{page === 'online' ? 'ONLINE' : page === 'offline' ? 'OFFLINE' : page === 'replay' ? 'REPLAY' : 'EVENTS'}</span></h1><p>{page === 'online' ? '从相机画面到数字场景，四路视角集中观测。' : page === 'incidents' ? '集中记录异常、关联片段，保留每一次人工核查。' : page === 'replay' ? '选择片段，在同一时间轴下复查四路画面。' : '同一段交通流，四个观察维度。'}</p></div><button className="wb-button" onClick={page === 'incidents' ? () => setEventForm(true) : openSettings}>{page === 'incidents' ? <Plus size={16} /> : <Settings2 size={16} />}{page === 'incidents' ? '新增记录' : '配置视频源'}</button></section>
        {notice && <div className="wb-notice" role="status">{notice}<button aria-label="关闭提示" onClick={() => setNotice('')}><X size={15} /></button></div>}
        {page === 'incidents' ? <>
          <div className="wb-event-summary"><div><ShieldAlert size={20} /><strong>{incidents.length}</strong><span>本机手动记录</span></div><div><Clock3 size={20} /><strong>{incidents.filter(r => r.status === '待处理').length}</strong><span>待核查</span></div><p>事件服务尚未接入 · 不生成虚构事故数据</p></div>
          <section className="wb-events"><div className="wb-table-tools"><input aria-label="搜索事件" placeholder="搜索事件名称或类型…" value={query} onChange={e => setQuery(e.target.value)} /><select aria-label="筛选事件状态" value={filter} onChange={e => setFilter(e.target.value)}><option>全部</option><option>待处理</option><option>已处理</option></select><button className="wb-button" onClick={exportEvents}><ArrowDownToLine size={15} />导出 JSON</button></div>
          {visibleEvents.length ? <div className="wb-table-scroll"><table><thead><tr><th>事件</th><th>类型</th><th>记录时间</th><th>片段时间</th><th>状态</th><th>操作</th></tr></thead><tbody>{visibleEvents.map(r => <tr key={r.id}><td>{r.title}<small>人工记录 · {r.mode}</small></td><td>{r.kind}</td><td>{new Date(r.time).toLocaleString()}</td><td>{stamp(r.position)}</td><td><span className="wb-tag">{r.status}</span></td><td><button className="wb-text-button" onClick={() => persistEvents(incidents.map(i => i.id === r.id ? { ...i, status: i.status === '待处理' ? '已处理' : '待处理' } : i))}>{r.status === '待处理' ? '标记处理' : '重新打开'}</button><button className="wb-text-button" onClick={() => { if (deleteId === r.id) { persistEvents(incidents.filter(i => i.id !== r.id)); setDeleteId(null); } else setDeleteId(r.id); }}>{deleteId === r.id ? '确认删除' : '删除'}</button>{deleteId === r.id && <button onClick={() => setDeleteId(null)}>取消</button>}</td></tr>)}</tbody></table></div> : <div className="wb-empty-events"><ShieldAlert size={34} /><h2>{incidents.length ? '没有匹配的记录' : '还没有事件记录'}</h2><p>可手动记录跟踪异常、视频中断或待核查事故。</p><button className="wb-button" onClick={() => setEventForm(true)}><Plus size={16} />创建第一条记录</button></div>}</section>
        </> : <>
          {page === 'replay' && <div className="wb-replay-bar"><FileVideo size={22} /><div><b>当前离线视频源</b><small>与 Offline 共用配置 · 可通过“配置视频源”切换片段</small></div><button className="wb-button" onClick={() => { pause(); seek(0); }}><RotateCcw size={15} />回到起点</button></div>}
          <div className="wb-scene"><span><span className={`wb-dot ${mode === 'online' ? 'muted' : ''}`} />{mode === 'online' ? '在线输入' : '离线视频源'}</span><span>{Object.values(sources).filter(Boolean).length} / 4 已配置</span><span className="wb-scene-end">{mode === 'offline' ? '本地 V5 样例 / 自定义输入' : '等待外部流接入'}</span></div>
          <div className={`wb-stream-grid ${focused ? 'wb-focused' : ''}`}>{channels.map((c, index) => <section className={`wb-stream ${focused && focused !== c.id ? 'wb-hidden' : ''}`} key={`${page}-${c.id}`}>
            <div className="wb-stream-head"><span className="wb-channel-number">0{index + 1}</span><c.icon size={16} /><h2>{c.title}</h2><span className="wb-stream-state"><i className={states[c.id] === '就绪' ? 'ready' : ''} />{sources[c.id] ? states[c.id] || '加载中' : '待接入'}</span><button aria-label={`${focused === c.id ? '还原' : '放大'}${c.title}`} onClick={() => setFocused(focused === c.id ? null : c.id)}><Maximize2 size={15} /></button></div>
            <div className="wb-video-surface">{sources[c.id] ? <><video key={sources[c.id]} ref={el => { if (el) videos.current[c.id] = el; else delete videos.current[c.id]; }} src={sources[c.id]} muted playsInline preload="auto" onLoadedMetadata={e => { if (c.id === first) setDuration(Number.isFinite(e.currentTarget.duration) ? e.currentTarget.duration : 0); }} onCanPlay={() => setStates(s => ({ ...s, [c.id]: '就绪' }))} onWaiting={() => setStates(s => ({ ...s, [c.id]: '缓冲中' }))} onPlaying={() => setStates(s => ({ ...s, [c.id]: '就绪' }))} onError={() => setStates(s => ({ ...s, [c.id]: '加载失败' }))} onTimeUpdate={e => clock(c.id, e.currentTarget)} onEnded={() => { if (c.id === first) pause(); }} />{states[c.id] === '加载失败' && <div className="wb-video-error"><Video size={24} /><b>视频加载失败</b><span>检查地址、编码格式或跨域设置</span><button onClick={openSettings}>修改视频源</button></div>}</> : <div className="wb-placeholder"><div className="wb-placeholder-icon"><c.icon size={30} strokeWidth={1.1} /></div><b>{c.id === 'unity' ? '等待 Unity 画面' : '等待视频源'}</b><p>{c.id === 'unity' ? '为数字场景预留的独立视频通道' : '配置输入后，在此展示视频画面'}</p><button onClick={openSettings}><Link2 size={14} />连接视频源<ArrowRight size={14} /></button></div>}<span className="wb-watermark">{c.english}</span></div>
            <footer><span>{c.description}</span><span>{sources[c.id] ? 'VIDEO' : 'NOT CONNECTED'}</span></footer>
          </section>)}</div>
          <section className="wb-transport"><button className="wb-play" aria-label={playing ? '暂停全部视频' : '播放全部视频'} onClick={() => void play()}>{playing ? <Pause size={18} /> : <Play size={18} />}</button><button aria-label="回退一秒" disabled={mode === 'online' || !duration} onClick={() => seek(Math.max(0, time - 1))}><ArrowLeft size={16} /></button><div className="wb-timeline"><div><b>{stamp(time)}</b><span>{mode === 'offline' ? '联动回放 · 首个已配置通道为主时钟' : '在线播放 · 各通道独立时钟'}</span><span>{stamp(duration)}</span></div><input aria-label="回放进度" type="range" min="0" max={duration || 1} step="0.01" disabled={mode === 'online' || !duration} value={Math.min(time, duration || 1)} onChange={e => seek(Number(e.target.value))} /></div><select aria-label="播放速度" value={rate} disabled={mode === 'online'} onChange={e => { const n = Number(e.target.value); setRate(n); Object.values(videos.current).forEach(v => { if (v) v.playbackRate = n; }); }}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select><button className="wb-record" onClick={() => setEventForm(true)}><Plus size={16} />记录事件</button></section>
          <div className="wb-footnote"><ShieldAlert size={14} /><span>V5 样例为本地算法输出，标定精度未验证；Unity 尚未接入。联动播放用于观察，不替代逐帧数据核验。</span><a href="?evidence">算法核验页<ArrowRight size={13} /></a></div>
        </>}
      </div>
      <footer className="wb-statusbar"><span><Activity size={13} />CAROM Studio <b> / </b> 前端工作台</span><span>固定单目相机 · 四视图观测</span></footer>
    </main>
    <aside className="wb-sidebar"><div className="wb-brand"><span className="wb-brand-icon"><Layers3 size={21} /></span><div><b>CAROM</b><small>交通场景重建</small></div><span className="wb-beta">工作台</span></div><div className="wb-nav-label">工作空间</div><nav aria-label="主导航">{nav.map(n => <a key={n.id} href={`#${n.id}`} aria-current={page === n.id ? 'page' : undefined} className={page === n.id ? 'active' : ''} onClick={e => { e.preventDefault(); navigate(n.id); }}><n.icon size={18} /><span>{n.title}<small>{n.sub}</small></span>{page === n.id && <span className="wb-nav-indicator" />}</a>)}</nav><a href="http://127.0.0.1:8767/?library">统一结果库 / 客户端 ↗</a><div className="wb-sidebar-rule" /><div className="wb-nav-label">输入通道 <span>04</span></div><div className="wb-channel-list">{channels.map(c => <div key={c.id}><c.icon size={15} /><span>{c.title}</span><small>{sources[c.id] ? '已配置' : '未配置'}</small></div>)}</div><div className="wb-sidebar-note"><Link2 size={17} /><b>为真实数据预留接口</b><p>每个窗口独立配置。当前支持浏览器可播放的视频地址；实时协议按后续服务接入。</p><button onClick={openSettings}>管理输入<ArrowRight size={14} /></button></div><div className="wb-sidebar-bottom"><span className="wb-avatar">C</span><div><b>本地项目</b><small>配置仅存于当前浏览器</small></div><Settings2 size={16} /></div></aside>
    <dialog ref={dialogRef} className="wb-dialog" onCancel={() => setSettings(false)}><form onSubmit={saveSources}><div className="wb-dialog-head"><div><h2>配置视频源</h2><p>{mode === 'online' ? 'Online' : 'Offline / 回放'} · 四路输入独立配置</p></div><button type="button" aria-label="关闭配置" onClick={() => setSettings(false)}><X size={19} /></button></div><p className="wb-help">支持 MP4 / WebM 等浏览器可播放的 HTTP(S) 地址。RTSP、WebRTC、HLS 专用适配后续接入；不要填写密钥或带凭据的地址。</p>{channels.map(c => <label key={c.id}>{c.title}<input value={draft[c.id]} onChange={e => setDraft({ ...draft, [c.id]: e.target.value })} placeholder="https://… / 留空表示待接入" /></label>)}{formError && <p role="alert" className="wb-form-error">{formError}</p>}<div className="wb-dialog-actions"><button type="button" onClick={() => setDraft(mode === 'offline' ? { ...demo } : { ...empty })}>{mode === 'offline' ? '填入本地样例' : '清空输入'}</button><button className="wb-primary" type="submit"><Check size={15} />保存配置</button></div></form></dialog>
    <dialog ref={eventRef} className="wb-dialog" onCancel={() => setEventForm(false)}><form onSubmit={addEvent}><div className="wb-dialog-head"><div><h2>记录事件</h2><p>人工记录 · 当前片段 {stamp(time)}</p></div><button type="button" aria-label="关闭事件表单" onClick={() => setEventForm(false)}><X size={19} /></button></div><label>事件描述<input required maxLength={160} value={eventTitle} onChange={e => setEventTitle(e.target.value)} placeholder="例如：遮挡后车辆 ID 发生变化" /></label><label>事件类型<select value={eventKind} onChange={e => setEventKind(e.target.value)}><option>跟踪异常</option><option>视频故障</option><option>疑似事故</option><option>其他</option></select></label><p className="wb-help">保存在当前浏览器，不会提交到事故平台；人工记录不等同于已确认事故。</p><div className="wb-dialog-actions"><button type="button" onClick={() => setEventForm(false)}>取消</button><button className="wb-primary" type="submit">保存记录</button></div></form></dialog>
  </div>;
}

