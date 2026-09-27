import { useEffect, useRef, useState } from "react";

const API = `${import.meta.env.VITE_API_BASE ?? ""}/api/carom-method`;
type Run = { run_id: string; version: string; frames: number; fps: number; preview_frame: number };
type Frame = { frame_index: number; observed_count: number; localized_count: number; coasting_count: number; objects: unknown[] };
function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
function readRun(value: unknown): Run {
  if (!object(value) || typeof value.run_id !== "string" || typeof value.version !== "string"
    || typeof value.frames !== "number" || !Number.isInteger(value.frames) || value.frames < 1
    || typeof value.fps !== "number" || !Number.isFinite(value.fps) || value.fps <= 0
    || typeof value.preview_frame !== "number") throw new Error("版本元数据格式错误");
  return { run_id: value.run_id, version: value.version, frames: value.frames, fps: value.fps, preview_frame: value.preview_frame };
}
function readFrame(value: unknown): Frame {
  if (!object(value) || typeof value.frame_index !== "number" || typeof value.observed_count !== "number"
    || typeof value.localized_count !== "number" || typeof value.coasting_count !== "number" || !Array.isArray(value.objects)) {
    throw new Error("逐帧数据格式错误");
  }
  return { frame_index: value.frame_index, observed_count: value.observed_count,
    localized_count: value.localized_count, coasting_count: value.coasting_count, objects: value.objects };
}
async function load(path: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(`${API}/${path}`, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`${path}：HTTP ${response.status}，当前版本数据尚未就绪`);
  return response.json();
}

export default function Release() {
  const [run, setRun] = useState<Run | null>(null);
  const [reports, setReports] = useState<unknown>(null);
  const [record, setRecord] = useState<Frame | null>(null);
  const [rawRecord, setRawRecord] = useState<unknown>(null);
  const [error, setError] = useState("");
  const [frameError, setFrameError] = useState("");
  const [frame, setFrame] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const bev = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([load("run-info", controller.signal), load("verification", controller.signal),
      load("smoothing-report", controller.signal), load("calibration-readiness", controller.signal)])
      .then(([info, verification, smoothing, calibration]) => {
        setRun(readRun(info)); setReports({ verification, smoothing, calibration }); setError("");
      }).catch((reason: unknown) => { if (!controller.signal.aborted) setError(String(reason)); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!run) return;
    const controller = new AbortController();
    load(`frames/${frame}`, controller.signal).then(data => {
      const parsed = readFrame(data);
      if (parsed.frame_index !== frame) throw new Error("请求帧号与返回帧号不一致");
      setRecord(parsed); setRawRecord(data); setFrameError("");
    }).catch((reason: unknown) => { if (!controller.signal.aborted) setFrameError(String(reason)); });
    return () => controller.abort();
  }, [frame, run]);
  function seek(index: number) {
    if (!run) return;
    video.current?.pause(); bev.current?.pause();
    if (video.current) video.current.currentTime = index / run.fps;
    if (bev.current) bev.current.currentTime = index / run.fps;
    setFrame(index);
  }
  const displayedRecord = record?.frame_index === frame ? record : null;
  return <main className="app-shell">
    <header className="panel-head"><h1>CAROM · 离线核验版 {run?.version ?? "加载中"}</h1><a href="?legacy">历史演示（独立旧数据）</a></header>
    <p>固定单目 · 本地 CV · 原始观测可追溯 · 无最终三维场景</p>
    <p role="note">实验标定，车道、距离与体积准确性尚未验证。平滑使用后续帧，既非实时处理，也非完整论文复现。</p>
    {error && <p role="alert">{error}。不回退展示旧版本；生成完成后刷新页面。</p>}
    {run && <>
      <p data-testid="run-version">数据集：{run.run_id} · {run.frames} 帧 · API：/api/carom-method</p>
      <section className="panel-shell"><h2>原始画面 / 实例与编号 / 俯视位置</h2>
        <video ref={video} src={`${API}/video`} controls playsInline preload="metadata" width="100%" aria-label="当前版本处理视频"
          onTimeUpdate={event => {
            const time = event.currentTarget.currentTime;
            setFrame(Math.min(run.frames - 1, Math.floor(time * run.fps + 1e-4)));
            if (bev.current && Math.abs(bev.current.currentTime - time) > 0.15) bev.current.currentTime = time;
          }} onPlay={() => { bev.current?.play().catch((reason: unknown) => setFrameError(`俯视视频播放：${String(reason)}`)); }}
          onPause={() => bev.current?.pause()} />
        <video ref={bev} src={`${API}/bev-detail-video`} playsInline muted preload="metadata" width="100%" aria-label="同步俯视视频" />
        <p>上方播放器驱动视频和逐帧数据。S / smoothed 为离线估计，P / predicted 为预测；accepted、rejected、missing 分别表示当前量测被采用、被拒绝、缺失。</p>
        <p><a href={`${API}/video`} download>下载当前处理视频</a> · <a href={`${API}/bev-view`}>查看等比例俯视大图</a></p>
      </section>
      <section className="panel-shell"><h2>可追溯帧记录</h2>
        <label>帧 {frame} / {run.frames - 1} <input aria-label="选择帧" type="range" min={0} max={run.frames - 1} value={frame} onChange={event => seek(Number(event.target.value))} /></label>
        <label>跳至帧 <input aria-label="跳至帧" type="number" min={0} max={run.frames - 1} value={frame} onChange={event => {
          const n = Number(event.target.value); if (Number.isInteger(n) && n >= 0 && n < run.frames) seek(n);
        }} /></label>
        {frameError && <p role="alert">{frameError}</p>}
        {displayedRecord ? <>
          <p data-testid="frame-counts">帧 {displayedRecord.frame_index}：发布的检测对象 {displayedRecord.observed_count}；漏检保留 {displayedRecord.coasting_count}；平面定位 {displayedRecord.localized_count}；待定位 {displayedRecord.objects.length - displayedRecord.localized_count}。</p>
          <table><thead><tr><th>ID</th><th>位置状态</th><th>当前量测</th><th>检测丢失帧</th><th>位置</th></tr></thead>
            <tbody>{displayedRecord.objects.filter(object).map(row => <tr key={String(row.id)}>
              <td>{String(row.id)}</td><td>{String(row.geometry_status)}</td><td>{String(row.measurement_status ?? "—")}</td>
              <td>{String(row.lost_frames)}</td><td>{Array.isArray(row.ground_xy) ? row.ground_xy.map(v => typeof v === "number" ? v.toFixed(2) : "?").join(", ") : "待定位"}</td>
            </tr>)}</tbody></table>
          <details><summary>当前帧完整数据、ID、量测和修正量</summary><pre>{JSON.stringify(rawRecord, null, 2)}</pre></details>
        </> : <p role="status">正在读取帧 {frame}…</p>}
        <a href={`${API}/frames/${frame}`} target="_blank" rel="noreferrer">打开当前帧 JSON</a>
        <details><summary>一致性测试、平滑对比及标定缺项</summary><pre>{JSON.stringify(reports, null, 2)}</pre></details>
      </section>
      <section className="panel-shell"><h2>固定验证截图：第 {run.preview_frame} 帧（不随播放变化）</h2>
        <img src={`${API}/validation`} alt="当前版本三阶段验证截图" width="100%" />
      </section>
    </>}
  </main>;
}