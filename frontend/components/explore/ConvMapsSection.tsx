"use client";

import { useEffect, useRef } from "react";
import type { ExplorePayload } from "@/lib/explore-types";
import { drawFeatureMap, hotColor } from "@/lib/canvas-utils";

interface Props {
  payload: ExplorePayload;
}

interface GridProps {
  label: string;
  flat: number[];
  channelCount: number;
  mapW: number;
  mapH: number;
  displaySize: number;
}

function FeatureMapCanvas({
  data,
  w,
  h,
  displaySize,
  title,
}: {
  data: Float32Array;
  w: number;
  h: number;
  displaySize: number;
  title: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) {
      drawFeatureMap(canvasRef.current, data, w, h, hotColor);
    }
  }, [data, w, h]);

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

function FeatureMapGrid({ label, flat, channelCount, mapW, mapH, displaySize }: GridProps) {
  const typed = new Float32Array(flat);
  const pixelsPerMap = mapW * mapH;

  // Slice into per-channel arrays
  const maps = Array.from({ length: channelCount }, (_, i) =>
    typed.slice(i * pixelsPerMap, (i + 1) * pixelsPerMap)
  );

  // Sort by max activation descending
  const sorted = maps
    .map((m, i) => ({ m, i, max: Math.max(...m) }))
    .sort((a, b) => b.max - a.max);

  return (
    <div>
      <p className="text-xs font-semibold text-gray-600 dark:text-slate-300 mb-2">{label}</p>
      <div className="flex flex-wrap gap-1">
        {sorted.map(({ m, i, max }) => (
          <FeatureMapCanvas
            key={i}
            data={m}
            w={mapW}
            h={mapH}
            displaySize={displaySize}
            title={`Filter ${i} · max ${max.toFixed(2)}`}
          />
        ))}
      </div>
    </div>
  );
}

export default function ConvMapsSection({ payload }: Props) {
  return (
    <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">
          Convolutional Layer Feature Maps
        </h2>
        <p className="text-sm text-gray-500 dark:text-slate-400 mt-1">
          Each tile shows one filter&apos;s response to the input. Brighter = stronger activation. Sorted by peak activation.
        </p>
      </div>

      <FeatureMapGrid
        label="Conv1 Pool — 32 filters × 14×14 (detecting edges and simple patterns)"
        flat={payload.conv1Pool}
        channelCount={32}
        mapW={14}
        mapH={14}
        displaySize={42}
      />

      <FeatureMapGrid
        label="Conv2 Pool — 64 filters × 7×7 (detecting complex shapes)"
        flat={payload.conv2Pool}
        channelCount={64}
        mapW={7}
        mapH={7}
        displaySize={36}
      />
    </section>
  );
}
