"use client";

import type { ExplorePayload, Fc2WeightsJson } from "@/lib/explore-types";
import { computeContributions } from "@/lib/explore-types";

interface Props {
  payload: ExplorePayload;
  fc2Weights: Fc2WeightsJson;
}

export default function DecisionSummary({ payload, fc2Weights }: Props) {
  const topContributions = computeContributions(payload.fc1, fc2Weights.fc2_weight, payload.predicted, 3);
  const confidence = payload.probs[payload.predicted] * 100;

  return (
    <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6">
      <h2 className="text-sm font-semibold text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-4">
        Decision Summary
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">

        {/* Col 1: preview image */}
        <div className="flex flex-col items-center gap-2">
          <p className="text-xs text-gray-400 dark:text-slate-500 uppercase tracking-wide">Input (28×28)</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={payload.previewDataUrl}
            alt="Preprocessed digit"
            width={96}
            height={96}
            className="rounded border border-gray-200 dark:border-slate-600"
            style={{ imageRendering: "pixelated", width: 96, height: 96 }}
          />
        </div>

        {/* Col 2: prediction + top-3 probs */}
        <div className="flex flex-col gap-3">
          <div className="flex items-baseline gap-2">
            <span className="text-5xl font-bold text-indigo-600 dark:text-indigo-400">
              {payload.predicted}
            </span>
            <span className="text-gray-500 dark:text-slate-400 text-sm">
              {confidence.toFixed(1)}% confidence
            </span>
          </div>
          <div className="space-y-1.5">
            {payload.topK.map(({ digit, prob }, rank) => (
              <div key={digit} className="flex items-center gap-2 text-sm">
                <span className="w-4 text-gray-400 dark:text-slate-500 text-xs">{rank + 1}.</span>
                <span className={`w-4 font-mono ${digit === payload.predicted ? "text-indigo-600 dark:text-indigo-400 font-bold" : "text-gray-700 dark:text-slate-200"}`}>
                  {digit}
                </span>
                <div className="flex-1 h-2 bg-gray-100 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${digit === payload.predicted ? "bg-indigo-500" : "bg-gray-300 dark:bg-slate-500"}`}
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 dark:text-slate-400 w-12 text-right tabular-nums">
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Col 3: top FC1 neuron contributions */}
        <div className="flex flex-col gap-2">
          <p className="text-xs text-gray-400 dark:text-slate-500 uppercase tracking-wide">
            Top FC1 Contributors → class {payload.predicted}
          </p>
          {topContributions.map((n, rank) => (
            <div key={n.neuronIndex} className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 text-xs">
                <span className="text-gray-400 dark:text-slate-500 w-4">{rank + 1}.</span>
                <span className="font-mono text-gray-700 dark:text-slate-300 w-12">
                  N{n.neuronIndex}
                </span>
                <span className={`font-semibold tabular-nums ${n.contribution >= 0 ? "text-indigo-600 dark:text-indigo-400" : "text-amber-600 dark:text-amber-400"}`}>
                  {n.contribution >= 0 ? "+" : ""}{n.contribution.toFixed(3)}
                </span>
              </div>
              <p className="text-[10px] text-gray-400 dark:text-slate-500 pl-16">
                {n.activation.toFixed(3)} × {n.weight.toFixed(3)}
              </p>
            </div>
          ))}
        </div>

      </div>
    </section>
  );
}
