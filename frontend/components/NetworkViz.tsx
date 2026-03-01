"use client";

import { useEffect, useRef, useState } from "react";
import type { ActivationResult } from "@/lib/infer_activations";
import { hotColor, divergingColor, drawFeatureMap } from "@/lib/canvas-utils";

interface NetworkVizProps {
  preview: ImageData | null;        // 28×28 input (what the model sees)
  activations: ActivationResult | null;
  predicted: number | null;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StageLabel({ label, sub }: { label: string; sub: string }) {
  return (
    <div className="text-center mb-2">
      <p className="text-xs font-semibold text-gray-700 dark:text-slate-200">{label}</p>
      <p className="text-xs text-gray-400 dark:text-slate-500">{sub}</p>
    </div>
  );
}

function Arrow({ strength = 1 }: { strength?: number }) {
  const opacity = 0.2 + strength * 0.8;
  return (
    <div className="flex items-center self-start mt-12 px-1 shrink-0">
      <svg width="32" height="16" viewBox="0 0 32 16" style={{ opacity }}>
        <line x1="0" y1="8" x2="24" y2="8" stroke="#6366f1" strokeWidth="2" />
        <polygon points="24,4 32,8 24,12" fill="#6366f1" />
      </svg>
    </div>
  );
}

// Input stage — renders the 28×28 preprocessed preview
function InputStage({ preview }: { preview: ImageData | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    if (preview) {
      ctx.putImageData(preview, 0, 0);
    } else {
      ctx.fillStyle = "#f3f4f6";
      ctx.fillRect(0, 0, 28, 28);
    }
  }, [preview]);

  return (
    <div className="flex flex-col items-center shrink-0">
      <StageLabel label="Input" sub="28 × 28" />
      <canvas
        ref={canvasRef}
        width={28}
        height={28}
        className="border border-gray-200 dark:border-slate-600 rounded"
        style={{ width: 84, height: 84, imageRendering: "pixelated" }}
      />
    </div>
  );
}

// Single feature map canvas
function FeatureMapCanvas({
  data,
  w,
  h,
  displaySize,
  colorFn,
  title,
}: {
  data: Float32Array;
  w: number;
  h: number;
  displaySize: number;
  colorFn: (v: number) => [number, number, number];
  title?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) drawFeatureMap(canvasRef.current, data, w, h, colorFn);
  }, [data, w, h, colorFn]);

  return (
    <canvas
      ref={canvasRef}
      width={w}
      height={h}
      title={title}
      className="border border-gray-100 dark:border-slate-700 rounded"
      style={{ width: displaySize, height: displaySize, imageRendering: "pixelated" }}
    />
  );
}

// Conv feature maps stage (conv1_pool or conv2_pool)
function ConvStage({
  label,
  sub,
  maps,
  mapW,
  mapH,
  count,
}: {
  label: string;
  sub: string;
  maps: Float32Array[];
  mapW: number;
  mapH: number;
  count: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const DISPLAY = 8;
  const displaySize = 42;

  // Sort by max activation descending for the top-8 view
  const sorted = maps
    .map((m, i) => ({ m, i, max: Math.max(...m) }))
    .sort((a, b) => b.max - a.max);

  const toShow = expanded ? sorted : sorted.slice(0, DISPLAY);

  return (
    <div className="flex flex-col items-center shrink-0">
      <StageLabel label={label} sub={sub} />
      <div
        className="flex flex-wrap gap-0.5"
        style={{ width: expanded ? displaySize * 8 + 28 : displaySize * 4 + 12 }}
      >
        {toShow.map(({ m, i }) => (
          <FeatureMapCanvas
            key={i}
            data={m}
            w={mapW}
            h={mapH}
            displaySize={displaySize}
            colorFn={hotColor}
            title={`Filter ${i} · max ${Math.max(...m).toFixed(2)}`}
          />
        ))}
      </div>
      <button
        onClick={() => setExpanded((e) => !e)}
        className="mt-1.5 text-xs text-indigo-500 dark:text-indigo-400 hover:underline"
      >
        {expanded ? `Show top ${DISPLAY}` : `Show all ${count}`}
      </button>
    </div>
  );
}

// FC1 stage — 128 neurons as a 16×8 coloured grid
function FC1Stage({ data }: { data: Float32Array }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const COLS = 16, ROWS = 8;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Map to [0,1] using diverging scale (0.5 = zero activation)
    let min = Infinity, max = -Infinity;
    for (const v of data) { if (v < min) min = v; if (v > max) max = v; }
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;

    const ctx = canvas.getContext("2d")!;
    const imgData = ctx.createImageData(COLS, ROWS);
    for (let i = 0; i < 128; i++) {
      const v = (data[i] / absMax) * 0.5 + 0.5; // map [-absMax,+absMax]→[0,1]
      const [r, g, b] = divergingColor(v);
      imgData.data[i * 4 + 0] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }, [data]);

  const activeCount = data.filter((v) => v > 0).length;

  return (
    <div className="flex flex-col items-center shrink-0">
      <StageLabel label="FC1" sub="128 neurons" />
      <canvas
        ref={canvasRef}
        width={COLS}
        height={ROWS}
        className="border border-gray-200 dark:border-slate-600 rounded"
        style={{ width: COLS * 6, height: ROWS * 6, imageRendering: "pixelated" }}
        title="Blue = negative, White = ~0, Red = active"
      />
      <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">{activeCount}/128 active</p>
    </div>
  );
}

// Output stage — 10-class softmax bars
function OutputStage({
  logits,
  predicted,
}: {
  logits: Float32Array;
  predicted: number | null;
}) {
  const arr = Array.from(logits);
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);
  const predIdx = probs.indexOf(Math.max(...probs));

  return (
    <div className="flex flex-col items-center shrink-0">
      <StageLabel label="Output" sub="10 classes" />
      <div className="flex gap-1 items-end h-24">
        {probs.map((p, i) => (
          <div key={i} className="flex flex-col items-center gap-0.5">
            <div
              className={`w-6 rounded-t transition-all ${
                i === predIdx ? "bg-indigo-500" : "bg-gray-200 dark:bg-slate-600"
              }`}
              style={{ height: `${Math.round(p * 88)}px` }}
            />
            <span
              className={`text-xs font-mono ${
                i === predIdx ? "text-indigo-600 dark:text-indigo-400 font-bold" : "text-gray-400 dark:text-slate-500"
              }`}
            >
              {i}
            </span>
          </div>
        ))}
      </div>
      {predicted !== null && (
        <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
          Predicted: <span className="text-indigo-600 dark:text-indigo-400 font-semibold">{predicted}</span>
          {" · "}{(probs[predicted] * 100).toFixed(1)}%
        </p>
      )}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function NetworkViz({ preview, activations, predicted }: NetworkVizProps) {
  if (!preview && !activations) {
    return (
      <div className="border border-gray-200 dark:border-slate-700 rounded-xl p-4 bg-white dark:bg-slate-800">
        <h3 className="font-semibold text-gray-800 dark:text-slate-100 mb-2">Network Activations</h3>
        <p className="text-sm text-gray-400 dark:text-slate-500">Draw a digit to see what happens inside the network.</p>
      </div>
    );
  }

  // Slice flat arrays into per-filter Float32Arrays
  const conv1Maps: Float32Array[] = activations
    ? Array.from({ length: 32 }, (_, i) =>
        activations.conv1Pool.slice(i * 14 * 14, (i + 1) * 14 * 14)
      )
    : [];

  const conv2Maps: Float32Array[] = activations
    ? Array.from({ length: 64 }, (_, i) =>
        activations.conv2Pool.slice(i * 7 * 7, (i + 1) * 7 * 7)
      )
    : [];

  // Arrow strength = mean absolute activation of the source layer, normalised
  const conv1Strength = activations
    ? Math.min(1, activations.conv1Pool.reduce((s, v) => s + Math.abs(v), 0) / (32 * 14 * 14 * 2))
    : 0;
  const conv2Strength = activations
    ? Math.min(1, activations.conv2Pool.reduce((s, v) => s + Math.abs(v), 0) / (64 * 7 * 7 * 2))
    : 0;
  const fc1Strength = activations
    ? Math.min(1, activations.fc1.reduce((s, v) => s + Math.abs(v), 0) / (128 * 3))
    : 0;

  return (
    <div className="border border-gray-200 dark:border-slate-700 rounded-xl p-4 space-y-3 bg-white dark:bg-slate-800">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-800 dark:text-slate-100">Network Activations</h3>
        <span className="text-xs text-gray-400 dark:text-slate-500">
          Arrow brightness = signal strength · Red = active · Blue = suppressed
        </span>
      </div>

      {/* Scrollable horizontal flow */}
      <div className="overflow-x-auto pb-2">
        <div className="flex items-start gap-0 min-w-max">
          <InputStage preview={preview} />

          <Arrow strength={preview ? 0.6 : 0} />

          {activations ? (
            <ConvStage
              label="Conv1 Pool"
              sub="32 maps · 14×14"
              maps={conv1Maps}
              mapW={14}
              mapH={14}
              count={32}
            />
          ) : (
            <PlaceholderStage label="Conv1 Pool" sub="32 maps · 14×14" />
          )}

          <Arrow strength={conv1Strength} />

          {activations ? (
            <ConvStage
              label="Conv2 Pool"
              sub="64 maps · 7×7"
              maps={conv2Maps}
              mapW={7}
              mapH={7}
              count={64}
            />
          ) : (
            <PlaceholderStage label="Conv2 Pool" sub="64 maps · 7×7" />
          )}

          <Arrow strength={conv2Strength} />

          {activations ? (
            <FC1Stage data={activations.fc1} />
          ) : (
            <PlaceholderStage label="FC1" sub="128 neurons" />
          )}

          <Arrow strength={fc1Strength} />

          {activations ? (
            <OutputStage logits={activations.logits} predicted={predicted} />
          ) : (
            <PlaceholderStage label="Output" sub="10 classes" />
          )}
        </div>
      </div>
    </div>
  );
}

function PlaceholderStage({ label, sub }: { label: string; sub: string }) {
  return (
    <div className="flex flex-col items-center shrink-0">
      <StageLabel label={label} sub={sub} />
      <div className="w-24 h-24 rounded bg-gray-50 dark:bg-slate-700 border border-gray-100 dark:border-slate-600 flex items-center justify-center">
        <span className="text-xs text-gray-300 dark:text-slate-500">–</span>
      </div>
    </div>
  );
}
