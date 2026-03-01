"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";
import type { InferenceResult } from "@/lib/infer_onnx";

interface ProbChartProps {
  result: InferenceResult | null;
}

export default function ProbChart({ result }: ProbChartProps) {
  if (!result) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-400 dark:text-slate-500 text-sm">
        Draw a digit to see predictions
      </div>
    );
  }

  const { probs, predicted, topK, latencyMs } = result;
  const chartData = probs.map((p, i) => ({
    digit: String(i),
    prob: parseFloat((p * 100).toFixed(1)),
  }));

  return (
    <div className="flex flex-col gap-4">
      {/* Prediction headline */}
      <div className="flex items-center justify-between">
        <div>
          <span className="text-5xl font-bold text-indigo-600 dark:text-indigo-400">{predicted}</span>
          <span className="ml-2 text-gray-500 dark:text-slate-400 text-sm">
            {(probs[predicted] * 100).toFixed(1)}% confidence
          </span>
        </div>
        <span className="text-xs text-gray-400 dark:text-slate-500">{latencyMs.toFixed(1)} ms</span>
      </div>

      {/* Bar chart */}
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <XAxis dataKey="digit" tick={{ fontSize: 12 }} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
          <Tooltip formatter={(v: number | undefined) => [v != null ? `${v}%` : "", "Probability"]} />
          <Bar dataKey="prob" radius={[3, 3, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={index}
                fill={index === predicted ? "#6366f1" : "#d1d5db"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Top-3 list */}
      <div className="space-y-1">
        <p className="text-xs text-gray-500 dark:text-slate-400 font-medium uppercase tracking-wide">
          Top 3
        </p>
        {topK.map(({ digit, prob }, rank) => (
          <div key={digit} className="flex items-center gap-2">
            <span className="w-4 text-xs text-gray-400 dark:text-slate-500">{rank + 1}.</span>
            <span
              className={`font-semibold ${
                digit === predicted ? "text-indigo-600 dark:text-indigo-400" : "text-gray-700 dark:text-slate-200"
              }`}
            >
              {digit}
            </span>
            <div className="flex-1 h-2 bg-gray-100 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  digit === predicted ? "bg-indigo-500" : "bg-gray-300 dark:bg-slate-500"
                }`}
                style={{ width: `${(prob * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="text-xs text-gray-500 dark:text-slate-400 w-12 text-right">
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
