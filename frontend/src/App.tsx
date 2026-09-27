import { useEffect, useMemo, useState, type MouseEvent, type ReactNode } from "react";
import {
  Activity,
  Box,
  Camera,
  Crosshair,
  Gauge,
  Layers3,
  Map,
  Navigation,
  Radar,
  Route,
} from "lucide-react";
import { fetchScene, projectPixel } from "./api";
import type { Detection, ProjectedPoint, Scene } from "./types";

const FRAME_WIDTH = 880;
const FRAME_HEIGHT = 390;
const MAP_WIDTH = 520;
const MAP_HEIGHT = 600;
const MAP_PADDING = 34;

function Panel({
  step,
  eyebrow,
  title,
  icon,
  chip,
  children,
  className = "",
}: {
  step: string;
  eyebrow: string;
  title: string;
  icon: ReactNode;
  chip: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel-shell stage-panel ${className}`}>
      <header className="panel-head">
        <span className="step-index">{step}</span>
        <div className="panel-title">
          <span className="eyebrow">{eyebrow}</span>
          <h2>{icon}{title}</h2>
        </div>
        <span className="status-chip">{chip}</span>
      </header>
      {children}
    </section>
  );
}

function RawPanel({
  scene,
  probe,
  onProject,
}: {
  scene: Scene;
  probe: ProjectedPoint | null;
  onProject: (u: number, v: number) => void;
}) {
  const handleClick = (event: MouseEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    onProject(
      ((event.clientX - bounds.left) / bounds.width) * scene.frame.width,
      ((event.clientY - bounds.top) / bounds.height) * scene.frame.height,
    );
  };

  return (
    <Panel step="01" eyebrow="SOURCE FRAME" title="原始固定相机画面" icon={<Camera size={17} />} chip="RAW">
      <div className="image-stage">
        <svg viewBox={`0 0 ${FRAME_WIDTH} ${FRAME_HEIGHT}`} onClick={handleClick} aria-label="原始固定相机帧">
          <image href={scene.frame.url} width={FRAME_WIDTH} height={FRAME_HEIGHT} preserveAspectRatio="none" />
          {probe && (
            <g className="probe" transform={`translate(${probe.u},${probe.v})`}>
              <circle r="11" /><line x1="-18" x2="18" /><line y1="-18" y2="18" />
            </g>
          )}
        </svg>
        <div className="frame-stamp">FRAME #{scene.frame_index} · {scene.timestamp.replace("T", " ")}</div>
        <div className="click-hint"><Crosshair size={13} /> 点选像素反投影</div>
      </div>
      <div className="stage-note"><span>INPUT</span> 单目 RGB · 880×390 · 固定机位</div>
    </Panel>
  );
}

const cuboidEdges: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

function InstancePanel({
  scene,
  selected,
  onSelect,
}: {
  scene: Scene;
  selected: number | null;
  onSelect: (id: number | null) => void;
}) {
  return (
    <Panel step="02" eyebrow="INSTANCE + VOLUME" title="车辆实例与体积框" icon={<Layers3 size={17} />} chip="MASK / ID">
      <div className="image-stage">
        <svg viewBox={`0 0 ${FRAME_WIDTH} ${FRAME_HEIGHT}`} aria-label="车辆实例掩膜与体积框">
          <image href={scene.frame.url} width={FRAME_WIDTH} height={FRAME_HEIGHT} preserveAspectRatio="none" className="processed-frame" />
          <polygon className="calibration-zone" points={scene.calibration.image_points.slice(0, 4).map((p) => p.join(",")).join(" ")} />
          {scene.calibration.image_points.map(([x, y], index) => (
            <g key={scene.calibration.control_labels[index]} className="control-point" transform={`translate(${x},${y})`}>
              <circle r="4" /><text x="7" y="-6">C{index + 1}</text>
            </g>
          ))}
          {scene.detections.map((item) => {
            const active = item.track_id === selected;
            const [labelX, labelY] = item.mask_polygon[0];
            return (
              <g
                key={item.track_id}
                className={`instance ${active ? "active" : ""}`}
                onPointerEnter={() => onSelect(item.track_id)}
                onPointerLeave={() => onSelect(null)}
                onClick={() => onSelect(item.track_id)}
              >
                <polygon points={item.mask_polygon.map((p) => p.join(",")).join(" ")} style={{ fill: item.color, stroke: item.color }} />
                {cuboidEdges.map(([from, to]) => (
                  <line
                    key={`${from}-${to}`}
                    x1={item.volume_box[from][0]}
                    y1={item.volume_box[from][1]}
                    x2={item.volume_box[to][0]}
                    y2={item.volume_box[to][1]}
                    style={{ stroke: item.color }}
                  />
                ))}
                <g transform={`translate(${labelX},${Math.max(20, labelY - 28)})`}>
                  <rect width="106" height="23" rx="2" style={{ fill: item.color }} />
                  <text x="7" y="16">ID {item.track_id} · {item.class_name.toUpperCase()}</text>
                </g>
              </g>
            );
          })}
        </svg>
        <div className="frame-stamp">MASK CONTOUR + TYPE PRIOR BOX</div>
        <div className="overlay-legend"><i className="mask-key" />实例轮廓<i className="box-key" />体积框</div>
      </div>
      <div className="stage-note"><span>OUTPUT</span> {scene.summary.vehicles} 个稳定 ID · 类别先验尺寸 · 朝向估计</div>
    </Panel>
  );
}

function toMap(scene: Scene, x: number, y: number) {
  const { x_min, x_max, y_min, y_max } = scene.ground_extent;
  return {
    x: MAP_PADDING + ((x - x_min) / (x_max - x_min)) * (MAP_WIDTH - MAP_PADDING * 2),
    y: MAP_HEIGHT - MAP_PADDING - ((y - y_min) / (y_max - y_min)) * (MAP_HEIGHT - MAP_PADDING * 2),
  };
}

function pointsToMap(scene: Scene, points: [number, number][]) {
  return points.map(([x, y]) => {
    const mapped = toMap(scene, x, y);
    return `${mapped.x},${mapped.y}`;
  }).join(" ");
}

function VehicleOnMap({
  scene,
  item,
  active,
  onSelect,
}: {
  scene: Scene;
  item: Detection;
  active: boolean;
  onSelect: (id: number | null) => void;
}) {
  const center = toMap(scene, item.ground.x, item.ground.y);
  const ppmX = (MAP_WIDTH - MAP_PADDING * 2) / (scene.ground_extent.x_max - scene.ground_extent.x_min);
  const ppmY = (MAP_HEIGHT - MAP_PADDING * 2) / (scene.ground_extent.y_max - scene.ground_extent.y_min);
  return (
    <g
      className={`map-vehicle ${active ? "active" : ""}`}
      onPointerEnter={() => onSelect(item.track_id)}
      onPointerLeave={() => onSelect(null)}
      onClick={() => onSelect(item.track_id)}
    >
      <polyline points={pointsToMap(scene, item.trajectory_m)} style={{ stroke: item.color }} className="trajectory" />
      <ellipse
        cx={center.x}
        cy={center.y}
        rx={item.uncertainty_m.lateral * ppmX}
        ry={item.uncertainty_m.longitudinal * ppmY}
        transform={`rotate(${item.heading_deg} ${center.x} ${center.y})`}
        style={{ stroke: item.color }}
        className="uncertainty"
      />
      <polygon points={pointsToMap(scene, item.footprint_m)} style={{ fill: item.color, stroke: item.color }} className="metric-footprint" />
      <g transform={`translate(${center.x},${center.y}) rotate(${item.heading_deg})`} className="heading-arrow">
        <line y1="0" y2="-24" /><path d="M -4 -18 L 0 -25 L 4 -18" />
      </g>
      <g transform={`translate(${center.x + 9},${center.y - 8})`} className="map-label">
        <rect width="70" height="28" rx="2" style={{ stroke: item.color }} />
        <text x="5" y="11">ID {item.track_id} · {item.lane_id}</text>
        <text x="5" y="23">{item.speed_kmh.toFixed(1)} km/h</text>
      </g>
    </g>
  );
}

function MapPanel({
  scene,
  selected,
  onSelect,
  probe,
}: {
  scene: Scene;
  selected: number | null;
  onSelect: (id: number | null) => void;
  probe: ProjectedPoint | null;
}) {
  const probePoint = probe ? toMap(scene, probe.x, probe.y) : null;
  return (
    <Panel step="03" eyebrow="METRIC ROAD PLANE" title="道路与车辆平面重建" icon={<Map size={17} />} chip="WORLD / m" className="map-panel">
      <div className="map-stage">
        <svg viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`} aria-label="米制道路平面重建">
          <defs>
            <pattern id="map-grid" width="25" height="25" patternUnits="userSpaceOnUse">
              <path d="M 25 0 L 0 0 0 25" fill="none" stroke="#51606a" strokeWidth=".6" opacity=".25" />
            </pattern>
            <pattern id="hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
              <line x1="0" y1="0" x2="0" y2="9" stroke="#9aa6aa" strokeWidth="2" opacity=".35" />
            </pattern>
          </defs>
          <rect width={MAP_WIDTH} height={MAP_HEIGHT} className="map-bg" />
          <rect width={MAP_WIDTH} height={MAP_HEIGHT} fill="url(#map-grid)" />
          <polygon points={pointsToMap(scene, scene.road_model.surface_polygon_m)} className="road-surface" />
          {scene.road_model.zones.map((zone) => <polygon key={zone.id} points={pointsToMap(scene, zone.polygon_m)} fill="url(#hatch)" className="road-zone" />)}
          {scene.road_model.lane_boundaries.map((boundary) => (
            <polyline
              key={boundary.id}
              points={pointsToMap(scene, boundary.points_m)}
              className={`lane-boundary ${boundary.kind}`}
            />
          ))}
          {scene.road_model.stop_lines.map((line) => (
            <polyline key={line.id} points={pointsToMap(scene, line.points_m)} className="stop-line" />
          ))}
          {scene.road_model.lanes.map((lane) => {
            const marker = toMap(scene, lane.centerline_m[0][0], 4);
            return <text key={lane.id} x={marker.x} y={marker.y} className="lane-id" textAnchor="middle">{lane.id}</text>;
          })}
          {[0, 12, 24, 36, 48, 60, 72].map((meter) => {
            const y = toMap(scene, 0, meter).y;
            return <g key={meter} className="meter-rule"><line x1="8" x2={MAP_WIDTH - 8} y1={y} y2={y} /><text x="6" y={y - 4}>{meter}m</text></g>;
          })}
          {scene.detections.map((item) => (
            <VehicleOnMap key={item.track_id} scene={scene} item={item} active={selected === item.track_id} onSelect={onSelect} />
          ))}
          {probePoint && (
            <g className="map-probe" transform={`translate(${probePoint.x},${probePoint.y})`}>
              <circle r="8" /><line x1="-14" x2="14" /><line y1="-14" y2="14" />
              <text x="-11" y="-12" textAnchor="end">({probe!.x.toFixed(1)}, {probe!.y.toFixed(1)})</text>
            </g>
          )}
        </svg>
        <div className="north-mark">+Y<span /></div>
        <div className="map-legend"><i className="uncertainty-key" />定位不确定度<i className="path-key" />历史轨迹</div>
      </div>
      <div className="stage-note"><span>OUTPUT</span> 米制车道图 · 真实车身轮廓 · 轨迹/转向可读</div>
    </Panel>
  );
}

export default function App() {
  const [scene, setScene] = useState<Scene | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(29);
  const [probe, setProbe] = useState<ProjectedPoint | null>(null);

  useEffect(() => {
    fetchScene().then(setScene).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : "加载场景失败");
    });
  }, []);

  const selectedVehicle = useMemo(
    () => scene?.detections.find((item) => item.track_id === selected) ?? null,
    [scene, selected],
  );

  const handleProject = async (u: number, v: number) => {
    try {
      setProbe(await projectPixel(u, v));
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "投影失败");
    }
  };

  if (error) return <main className="state-screen"><Radar size={42} /><h1>数据链路异常</h1><p>{error}</p></main>;
  if (!scene) return <main className="state-screen"><Radar className="spin" size={42} /><h1>载入离线场景</h1></main>;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Radar size={25} /></div>
          <div><span>CAROM // REBUILD</span><h1>固定相机交通流空间重建</h1></div>
        </div>
        <div className="pipeline-badge"><span>01 RAW</span><b>→</b><span>02 INSTANCE</span><b>→</b><span>03 WORLD</span></div>
        <div className="run-meta"><span className="live-dot" /> OFFLINE <strong>V0.2</strong></div>
      </header>

      <section className="metrics-rail">
        <div><Activity size={17} /><span>稳定目标</span><strong>{scene.summary.vehicles}</strong></div>
        <div><Route size={17} /><span>语义车道</span><strong>{scene.summary.road_lanes}</strong></div>
        <div><Crosshair size={17} /><span>标定控制点</span><strong>{scene.calibration.control_point_count}</strong></div>
        <div><Gauge size={17} /><span>标定 RMSE</span><strong>{scene.calibration.rmse_m}<small> m</small></strong></div>
      </section>

      <div className="stage-grid">
        <RawPanel scene={scene} probe={probe} onProject={handleProject} />
        <InstancePanel scene={scene} selected={selected} onSelect={setSelected} />
        <MapPanel scene={scene} selected={selected} onSelect={setSelected} probe={probe} />
      </div>

      <section className="inspection-row">
        <div className="registry panel-shell">
          <header className="registry-head"><span>TRACK REGISTRY</span><strong>生命周期与米制状态</strong></header>
          <div className="track-table">
            <div className="track-row table-label"><span>ID / 类型</span><span>车道</span><span>X / Y (m)</span><span>尺寸 L×W×H</span><span>速度</span><span>运动状态</span></div>
            {scene.detections.map((item) => (
              <button
                key={item.track_id}
                className={`track-row ${selected === item.track_id ? "selected" : ""}`}
                onClick={() => setSelected(item.track_id)}
              >
                <span className="track-id"><i style={{ background: item.color }} />#{item.track_id} {item.class_name}</span>
                <span>{item.lane_id}</span>
                <span>{item.ground.x.toFixed(2)} / {item.ground.y.toFixed(2)}</span>
                <span>{item.dimensions_m.length.toFixed(2)}×{item.dimensions_m.width.toFixed(2)}×{item.dimensions_m.height.toFixed(2)}</span>
                <span>{item.speed_kmh.toFixed(1)} km/h</span>
                <span className={item.motion.state === "turning" ? "turning" : ""}>{item.motion.state} / {item.motion.turn_intent}</span>
              </button>
            ))}
          </div>
        </div>
        <aside className="target-card panel-shell">
          <span className="eyebrow">ACTIVE OBJECT</span>
          {selectedVehicle ? (
            <>
              <div className="target-title"><Box size={22} /><strong>TRACK #{selectedVehicle.track_id}</strong></div>
              <p>{selectedVehicle.class_name.toUpperCase()} · {Math.round(selectedVehicle.confidence * 100)}% 检测置信度</p>
              <dl>
                <div><dt>朝向</dt><dd>{selectedVehicle.heading_deg}°</dd></div>
                <div><dt>角速度</dt><dd>{selectedVehicle.motion.yaw_rate_deg_s.toFixed(1)}°/s</dd></div>
                <div><dt>轨迹寿命</dt><dd>{selectedVehicle.age_frames} 帧</dd></div>
                <div><dt>朝向来源</dt><dd>{selectedVehicle.heading_source}</dd></div>
              </dl>
              <div className="intent"><Navigation size={14} /> {selectedVehicle.motion.state === "turning" ? "转向行为已识别" : "车道方向稳定"}</div>
            </>
          ) : <p>在实例图或平面图中选择车辆</p>}
        </aside>
      </section>

      <footer className="method-footer">
        <span>CALIBRATION · {scene.calibration.method.toUpperCase()}</span>
        <span>ROAD MODEL · {scene.road_model.version.toUpperCase()}</span>
        <span>COORDINATE · +Y FORWARD / +X CAMERA-RIGHT</span>
        <strong>3D SCENE RENDER · DEFERRED</strong>
      </footer>
    </main>
  );
}
