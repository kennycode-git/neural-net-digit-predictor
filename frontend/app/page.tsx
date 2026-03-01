"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import DigitCanvas from "@/components/DigitCanvas";
import ProcessedPreview from "@/components/ProcessedPreview";
import ProbChart from "@/components/ProbChart";
import ExplainCompare from "@/components/ExplainCompare";
import EdgeCasePicker from "@/components/EdgeCasePicker";
import BackendStatus from "@/components/BackendStatus";
import ThemeToggle from "@/components/ThemeToggle";
import { preprocessCanvas } from "@/lib/preprocess";
import { loadModel, runInference } from "@/lib/infer_onnx";
import type { InferenceResult } from "@/lib/infer_onnx";
import { loadActivationsModel, runActivations } from "@/lib/infer_activations";
import type { ActivationResult } from "@/lib/infer_activations";
import NetworkViz from "@/components/NetworkViz";
import type { ExplorePayload } from "@/lib/explore-types";

export default function HomePage() {
  const [preview, setPreview] = useState<ImageData | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [activations, setActivations] = useState<ActivationResult | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  // 28×28 float pixels (inverted, pre-norm) for explain backend
  const previewPixelsRef = useRef<number[][] | null>(null);
  const router = useRouter();

  // Pre-load ONNX models on mount
  useEffect(() => {
    loadModel()
      .then(() => setModelLoading(false))
      .catch((e) => {
        setModelError(String(e));
        setModelLoading(false);
      });
    // Load activations model lazily — doesn't block main inference
    loadActivationsModel().catch(() => {});
  }, []);

  const handleStrokeEnd = useCallback(
    async (canvas: HTMLCanvasElement) => {
      if (modelLoading) return;
      try {
        const { tensor, preview: prev } = await preprocessCanvas(canvas);
        setPreview(prev);

        // Store raw [0,1] inverted pixels for backend explain call
        // (from preview RGBA — R channel = inverted grayscale value)
        const px: number[][] = Array.from({ length: 28 }, (_, r) =>
          Array.from({ length: 28 }, (_, c) => {
            const idx = (r * 28 + c) * 4;
            return prev.data[idx] / 255;
          })
        );
        previewPixelsRef.current = px;

        const [inferResult, actResult] = await Promise.all([
          runInference(tensor),
          runActivations(tensor),
        ]);
        setResult(inferResult);
        setActivations(actResult);
      } catch (e) {
        console.error("Inference failed:", e);
      }
    },
    [modelLoading]
  );

  const handleExplore = useCallback(() => {
    if (!result || !activations || !preview) return;
    const offscreen = document.createElement("canvas");
    offscreen.width = 28;
    offscreen.height = 28;
    offscreen.getContext("2d")!.putImageData(preview, 0, 0);
    const payload: ExplorePayload = {
      predicted: result.predicted,
      probs: result.probs,
      topK: result.topK,
      conv1Pool: Array.from(activations.conv1Pool),
      conv2Pool: Array.from(activations.conv2Pool),
      fc1: Array.from(activations.fc1),
      logits: Array.from(activations.logits),
      previewDataUrl: offscreen.toDataURL("image/png"),
    };
    sessionStorage.setItem("explore_payload", JSON.stringify(payload));
    router.push("/explore");
  }, [result, activations, preview, router]);

  const handleClear = useCallback(() => {
    setPreview(null);
    setResult(null);
    setActivations(null);
    previewPixelsRef.current = null;
  }, []);

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-slate-900 transition-colors">
      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">
              Digit-Net Visual Lab
            </h1>
            <p className="text-gray-500 dark:text-slate-400 mt-1 text-sm">
              Browser-based MNIST digit inference with explainability
            </p>
          </div>
          <nav className="flex items-center gap-3 text-sm">
            <Link href="/model" className="text-indigo-600 dark:text-indigo-400 hover:underline">
              Model Card →
            </Link>
            <ThemeToggle />
          </nav>
        </div>

        {/* Model loading banner */}
        {modelLoading && (
          <div className="mb-6 p-3 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-lg text-sm text-blue-700 dark:text-blue-300">
            Loading ONNX model…
          </div>
        )}
        {modelError && (
          <div className="mb-6 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-lg text-sm text-red-700 dark:text-red-300">
            Failed to load model: {modelError}. Make sure{" "}
            <code className="bg-red-100 dark:bg-red-900/50 px-1 rounded">
              /artifacts/model.onnx
            </code>{" "}
            exists (run <code className="bg-red-100 dark:bg-red-900/50 px-1 rounded">python ml/export_onnx.py</code>).
          </div>
        )}

        {/* Main grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left: draw + preview */}
          <div className="space-y-6">
            <DigitCanvas onStrokeEnd={handleStrokeEnd} onClear={handleClear} />
            <ProcessedPreview preview={preview} />
          </div>

          {/* Right: predictions */}
          <div className="space-y-6">
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-5">
              <h2 className="text-sm font-semibold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-4">
                Prediction
              </h2>
              <ProbChart result={result} />
              {result !== null && (
                <button
                  onClick={handleExplore}
                  className="w-full mt-4 px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  Explore Decision Path →
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Explain comparison */}
        <div className="mt-8">
          <ExplainCompare
            backendOnline={backendOnline}
            inferenceResult={result}
            previewPixels={previewPixelsRef.current}
          />
        </div>

        {/* Network activation flow */}
        <div className="mt-8">
          <NetworkViz
            preview={preview}
            activations={activations}
            predicted={result?.predicted ?? null}
          />
        </div>

        {/* Edge cases */}
        <div className="mt-8 bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-5">
          <h2 className="text-sm font-semibold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-4">
            Edge Case Gallery
          </h2>
          <EdgeCasePicker
            onSelect={async (ec) => {
              // Decode base64 PNG → run inference
              const img = new Image();
              img.src = `data:image/png;base64,${ec.thumbnail_png_b64}`;
              img.onload = async () => {
                const tmpCanvas = document.createElement("canvas");
                tmpCanvas.width = 56;
                tmpCanvas.height = 56;
                const ctx = tmpCanvas.getContext("2d")!;
                ctx.fillStyle = "white";
                ctx.fillRect(0, 0, 56, 56);
                ctx.drawImage(img, 0, 0);
                await handleStrokeEnd(tmpCanvas);
              };
            }}
          />
        </div>

        {/* Footer */}
        <div className="mt-6 flex items-center justify-between text-xs text-gray-400 dark:text-slate-500">
          <BackendStatus onStatusChange={setBackendOnline} />
          <span>Inference runs fully in your browser · No data sent to server</span>
        </div>
      </div>
    </main>
  );
}
