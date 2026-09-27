export interface GroundPoint {
  x: number;
  y: number;
}

export interface Detection {
  track_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  mask_polygon: [number, number][];
  volume_box: [number, number][];
  anchor: [number, number];
  dimensions_m: { length: number; width: number; height: number };
  speed_kmh: number;
  heading_deg: number;
  heading_source: string;
  heading_confidence: number;
  lane_id: string;
  motion: { state: string; turn_intent: string; yaw_rate_deg_s: number };
  trajectory_m: [number, number][];
  uncertainty_m: { longitudinal: number; lateral: number };
  footprint_m: [number, number][];
  velocity_mps: GroundPoint;
  age_frames: number;
  missed_frames: number;
  color: string;
  ground: GroundPoint;
}

export interface RoadModel {
  version: string;
  source: string;
  surface_polygon_m: [number, number][];
  curbs: { id: string; points_m: [number, number][] }[];
  lane_boundaries: { id: string; kind: "solid" | "dashed"; points_m: [number, number][] }[];
  lanes: { id: string; centerline_m: [number, number][]; width_m: number; direction: string }[];
  stop_lines: { id: string; points_m: [number, number][]; width_m: number }[];
  zones: { id: string; kind: string; polygon_m: [number, number][] }[];
}

export interface Scene {
  scene_id: string;
  source: string;
  timestamp: string;
  frame_index: number;
  frame: { width: number; height: number; url: string };
  ground_extent: { x_min: number; x_max: number; y_min: number; y_max: number };
  calibration: {
    method: string;
    coordinate_system: string;
    image_points: [number, number][];
    ground_points: [number, number][];
    control_labels: string[];
    control_point_count: number;
    rmse_m: number;
    max_error_m: number;
  };
  road_model: RoadModel;
  detections: Detection[];
  summary: {
    vehicles: number;
    mean_speed_kmh: number;
    processing_ms: number;
    projection: string;
    road_lanes: number;
    turning_vehicles: number;
  };
}

export interface ProjectedPoint extends GroundPoint {
  u: number;
  v: number;
}
