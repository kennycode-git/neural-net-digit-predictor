"use client";

import { useState, useCallback } from "react";
import { getExplanation } from "@/lib/api";
import type { InferenceResult } from "@/lib/infer_onnx";

interface ExplainCompareProps {
  backendOnline: boolean;
  inferenceResult: InferenceResult | null;
  /** Raw 28×28 pixel values in [0,1] (inverted, pre-norm) for backend */
  previewPixels: number[][] | null;
}

type Method = "saliency" | "gradcam";

interface HeatmapPair {
  predicted: string | null;   // base64 PNG
  runnerUp: string | null;
}

export default function ExplainCompare({
  backendOnline,
  inferenceResult,
  previewPixels,
}: ExplainCompareProps) {
  const [loading, setLoading] = useState(false);
  const [heatmaps, setHeatmaps] = useState<HeatmapPair | null>(null);
  const [method, setMethod] = useState<Method>("saliency");
  const [error, setError] = useState<string | null>(null);

  const explain = useCallback(async () => {
    if (!inferenceResult || !previewPixels) return;
    setLoading(true);
    setError(null);

    const { predicted, topK } = inferenceResult;
    const runnerUpDigit = topK[1]?.digit ?? (predicted === 0 ? 1 : 0);

    const [predResult, runnerResult] = await Promise.all([
      getExplanation(previewPixels, predicted, method),
      getExplanation(previewPixels, runnerUpDigit, method),
    ]);

    if (!predResult || !runnerResult) {
      setError("Backend returned an error. Please try again.");
      setLoading(false);
      return;
    }

    setHeatmaps({
      predicted: predResult.heatmap_png_base64,
      runnerUp: runnerResult.heatmap_png_base64,
    });
    setLoading(false);
  }, [inferenceResult, previewPixels, method]);

  const canExplain = backendOnline && inferenceResult !== null && previewPixels !== null;

  return (
    <div className="border border-gray-200 dark:border-slate-700 rounded-xl p-4 space-y-4 bg-white dark:bg-slate-800">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="font-semibold text-gray-800 dark:text-slate-100">Explanation Heatmaps</h3>
        <div className="flex items-center gap-2">
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as Method)}
            disabled={!canExplain || loading}
            className="text-xs border border-gray-200 dark:border-slate-600 rounded px-2 py-1 bg-white dark:bg-slate-700 dark:text-slate-200 disabled:opacity-50"
          >
            <option value="saliency">Saliency</option>
            <option value="gradcam">GradCAM</option>
          </select>
          <button
            onClick={explain}
            disabled={!canExplain || loading}
            className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Explaining…" : "Explain"}
          </button>
        </div>
      </div>

      {!backendOnline && (
        <p className="text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30 rounded-md p-2">
          Start the backend server to enable heatmap explanations. Inference still works offline.
        </p>
      )}

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 rounded-md p-2">{error}</p>
      )}

      {heatmaps && inferenceResult && (
        <div className="grid grid-cols-2 gap-4">
          <HeatmapPanel
            title={`Predicted: ${inferenceResult.predicted}`}
            base64={heatmaps.predicted}
            highlight
          />
          <HeatmapPanel
            title={`Runner-up: ${inferenceResult.topK[1]?.digit ?? "?"}`}
            base64={heatmaps.runnerUp}
          />
        </div>
      )}

      {!heatmaps && canExplain && (
        <p className="text-sm text-gray-400 dark:text-slate-500 text-center py-4">
          Click &ldquo;Explain&rdquo; to see which pixels influenced each class score.
        </p>
      )}
    </div>
  );
}

function HeatmapPanel({
  title,
  base64,
  highlight = false,
}: {
  title: string;
  base64: string | null;
  highlight?: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <p
        className={`text-sm font-medium ${
          highlight ? "text-indigo-600 dark:text-indigo-400" : "text-gray-600 dark:text-slate-300"
        }`}
      >
        {title}
      </p>
      {base64 ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`data:image/png;base64,${base64}`}
          alt={title}
          className="w-28 h-28 rounded border border-gray-200 dark:border-slate-600"
          style={{ imageRendering: "pixelated" }}
        />
      ) : (
        <div className="w-28 h-28 rounded bg-gray-100 dark:bg-slate-700 flex items-center justify-center text-xs text-gray-400 dark:text-slate-500">
          No data
        </div>
      )}
    </div>
  );
}
