import type { ProjectedPoint, Scene } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchScene(): Promise<Scene> {
  const response = await fetch(`${API_BASE}/api/scene`);
  if (!response.ok) throw new Error(`scene request failed: ${response.status}`);
  return (await response.json()) as Scene;
}

export async function projectPixel(u: number, v: number): Promise<ProjectedPoint> {
  const response = await fetch(`${API_BASE}/api/project`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points: [{ u, v }] }),
  });
  if (!response.ok) throw new Error(`projection request failed: ${response.status}`);
  const payload = (await response.json()) as { points: ProjectedPoint[] };
  return payload.points[0];
}
