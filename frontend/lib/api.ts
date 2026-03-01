/**
 * Backend API client.
 * All functions return null on network error — inference still works without backend.
 */

import { BACKEND_URL, PREPROCESS_VERSION } from "./constants";

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/v1/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export interface ExplainResponse {
  heatmap_png_base64: string;
  method: string;
  target: number;
  model_version: string;
  preprocess_version: string;
  git_commit: string;
}

export async function getExplanation(
  pixels: number[][],
  target: number,
  method: "saliency" | "gradcam" = "saliency"
): Promise<ExplainResponse | null> {
  try {
    const res = await fetch(`${BACKEND_URL}/v1/explain?method=${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pixels,
        preprocess_version: PREPROCESS_VERSION,
        target,
      }),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) return null;
    return res.json() as Promise<ExplainResponse>;
  } catch {
    return null;
  }
}

export interface VersionResponse {
  model_version: string;
  preprocess_version: string;
  git_commit: string;
}

export async function getVersion(): Promise<VersionResponse | null> {
  try {
    const res = await fetch(`${BACKEND_URL}/v1/version`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    return res.json() as Promise<VersionResponse>;
  } catch {
    return null;
  }
}
