"use client";

import { useEffect, useRef } from "react";

interface ProcessedPreviewProps {
  preview: ImageData | null;
}

const DISPLAY_SIZE = 112; // 28×28 scaled 4×

export default function ProcessedPreview({ preview }: ProcessedPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    if (!preview) {
      ctx.fillStyle = "#f9fafb";
      ctx.fillRect(0, 0, 28, 28);
      return;
    }

    ctx.putImageData(preview, 0, 0);
  }, [preview]);

  return (
    <div className="flex flex-col items-center gap-2">
      <p className="text-xs text-gray-500 dark:text-slate-400 font-medium uppercase tracking-wide">
        What the model sees
      </p>
      <canvas
        ref={canvasRef}
        width={28}
        height={28}
        className="border border-gray-200 dark:border-slate-600 rounded"
        style={{
          width: DISPLAY_SIZE,
          height: DISPLAY_SIZE,
          imageRendering: "pixelated",
          background: "#f9fafb",
        }}
      />
      <p className="text-xs text-gray-400 dark:text-slate-500">28 × 28 px</p>
    </div>
  );
}
