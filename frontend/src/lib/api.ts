import type {
  AnalysisResult,
  AnalysisSummary,
  AnalyzeStartResponse,
  CompareResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export function mediaUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    /* ignore */
  }
  return `Request failed (${res.status})`;
}

export async function fetchAnalyses(limit = 50): Promise<AnalysisSummary[]> {
  const res = await fetch(`${API_BASE}/analyses?limit=${limit}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await parseError(res));
  const data = await res.json();
  return (data.analyses || []) as AnalysisSummary[];
}

export async function fetchAnalysis(id: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/analyses/${encodeURIComponent(id)}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as AnalysisResult;
}

export async function fetchCompare(
  left: string,
  right: string
): Promise<CompareResponse> {
  const qs = new URLSearchParams({ left, right });
  const res = await fetch(`${API_BASE}/compare?${qs}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as CompareResponse;
}

export async function startAnalyze(
  file: File,
  opts?: { meshOverlay?: boolean; strokeType?: string }
): Promise<AnalyzeStartResponse> {
  const form = new FormData();
  form.append("video", file);
  const mesh = opts?.meshOverlay ? "true" : "false";
  const stroke = opts?.strokeType || "FOREHAND_SMASH";
  const qs = new URLSearchParams({
    mesh_overlay: mesh,
    stroke_type: stroke,
  });
  const res = await fetch(`${API_BASE}/analyze?${qs}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as AnalyzeStartResponse;
}
