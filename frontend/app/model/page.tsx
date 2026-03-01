"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface MetricsData {
  epochs: {
    epoch: number;
    train_loss: number;
    val_loss: number;
    train_acc: number;
    val_acc: number;
  }[];
  best_val_acc: number;
}

interface ReliabilityBin {
  bin_lower: number;
  bin_upper: number;
  avg_confidence: number;
  accuracy: number | null;
  count: number;
}

interface ReliabilityData {
  ece: number;
  n_bins: number;
  n_samples: number;
  bins: ReliabilityBin[];
}

interface MisclassifiedEntry {
  true_label: number;
  predicted: number;
  confidence: number;
  runner_up: number;
  thumbnail_png_b64: string;
}

interface EdgeCase {
  id: string;
  transform: string;
  note: string;
  true_label: number;
  predicted: number;
  confidence: number;
  thumbnail_png_b64: string;
}

function useJSON<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    fetch(path)
      .then((r) => r.json())
      .then(setData)
      .catch(() => {});
  }, [path]);
  return data;
}

export default function ModelCardPage() {
  const metrics = useJSON<MetricsData>("/artifacts/metrics.json");
  const reliability = useJSON<ReliabilityData>("/artifacts/reliability.json");
  const misclassified = useJSON<MisclassifiedEntry[]>("/artifacts/misclassified.json");
  const edgeCases = useJSON<EdgeCase[]>("/artifacts/edge_cases.json");

  const reliabilityChartData =
    reliability?.bins
      .filter((b) => b.accuracy !== null && b.count > 0)
      .map((b) => ({
        confidence: b.avg_confidence,
        accuracy: b.accuracy as number,
        count: b.count,
      })) ?? [];

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-slate-900 transition-colors">
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-10">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <Link href="/" className="text-indigo-600 dark:text-indigo-400 text-sm hover:underline">
              ← Back to Lab
            </Link>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50 mt-2">Model Card</h1>
            <p className="text-gray-500 dark:text-slate-400 text-sm mt-1">
              DigitCNN · Trained on MNIST · Exported to ONNX
            </p>
          </div>
          <ThemeToggle />
        </div>

        {/* Summary */}
        <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-3">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">Summary</h2>
          <p className="text-gray-600 dark:text-slate-300 text-sm leading-relaxed">
            A compact convolutional neural network trained on the MNIST handwritten digit dataset.
            The model uses two conv blocks followed by fully-connected layers, totalling ~420K parameters.
            Inference runs entirely in the browser via onnxruntime-web — no server round-trip required.
          </p>
          {metrics && (
            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
              <Stat
                label="Best Val Acc"
                value={`${(metrics.best_val_acc * 100).toFixed(2)}%`}
              />
              <Stat label="Architecture" value="2-Conv CNN" />
              <Stat label="Parameters" value="~420K" />
              <Stat label="Dataset" value="MNIST" />
            </dl>
          )}
        </section>

        {/* Training curves */}
        {metrics && (
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-4">
            <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">Training Curves</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Loss */}
              <div>
                <p className="text-xs text-gray-500 dark:text-slate-400 font-medium uppercase tracking-wide mb-2">
                  Loss
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={metrics.epochs}>
                    <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="train_loss"
                      stroke="#6366f1"
                      dot={false}
                      name="Train"
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      stroke="#f59e0b"
                      dot={false}
                      name="Val"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {/* Accuracy */}
              <div>
                <p className="text-xs text-gray-500 dark:text-slate-400 font-medium uppercase tracking-wide mb-2">
                  Accuracy
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={metrics.epochs}>
                    <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                    <YAxis
                      domain={[0.9, 1.0]}
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip formatter={(v: number | undefined) => v != null ? `${(v * 100).toFixed(2)}%` : ""} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="train_acc"
                      stroke="#6366f1"
                      dot={false}
                      name="Train"
                    />
                    <Line
                      type="monotone"
                      dataKey="val_acc"
                      stroke="#f59e0b"
                      dot={false}
                      name="Val"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>
        )}

        {/* Confusion matrix */}
        <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">Confusion Matrix</h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/artifacts/confusion_matrix.png"
            alt="Confusion matrix"
            className="w-full max-w-md rounded border border-gray-100 dark:border-slate-700"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <p className="text-xs text-gray-400 dark:text-slate-500">
            Run{" "}
            <code className="bg-gray-100 dark:bg-slate-700 dark:text-slate-300 px-1 rounded">python ml/evaluate.py</code> to generate.
          </p>
        </section>

        {/* Reliability diagram */}
        {reliability && (
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-4">
            <div className="flex items-start justify-between">
              <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">
                Calibration — Reliability Diagram
              </h2>
              <span className="bg-indigo-50 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 text-xs font-semibold px-2 py-1 rounded-full">
                ECE = {(reliability.ece * 100).toFixed(2)}%
              </span>
            </div>
            <p className="text-sm text-gray-500 dark:text-slate-400">
              Each point shows the empirical accuracy of predictions in a confidence bin.
              A well-calibrated model lies close to the diagonal.
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="confidence"
                  type="number"
                  domain={[0, 1]}
                  name="Confidence"
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: "Confidence", position: "insideBottom", offset: -10, fontSize: 12 }}
                />
                <YAxis
                  dataKey="accuracy"
                  type="number"
                  domain={[0, 1]}
                  name="Accuracy"
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: "Accuracy", angle: -90, position: "insideLeft", fontSize: 12 }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(v: number | undefined, name: string | undefined) => [
                    v != null ? `${(v * 100).toFixed(1)}%` : "",
                    name ?? "",
                  ]}
                />
                {/* Perfect calibration diagonal */}
                <ReferenceLine
                  segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                  stroke="#d1d5db"
                  strokeDasharray="5 5"
                  label={{ value: "Perfect", position: "insideTopRight", fontSize: 11 }}
                />
                <Scatter data={reliabilityChartData} fill="#6366f1" />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="bg-amber-50 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-700 rounded-lg p-3 text-sm text-amber-800 dark:text-amber-300">
              <strong>Calibration note:</strong> MNIST models are typically slightly
              overconfident — high-confidence predictions can still be wrong on ambiguous or
              transformed inputs. ECE &lt; 2% is generally considered well-calibrated for this
              task.
            </div>
          </section>
        )}

        {/* Misclassified gallery */}
        {misclassified && misclassified.length > 0 && (
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-4">
            <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">
              Failure Gallery — High-Confidence Mistakes
            </h2>
            <p className="text-sm text-gray-500 dark:text-slate-400">
              Test-set examples where the model was confident but wrong.
            </p>
            <div className="flex flex-wrap gap-3">
              {misclassified.slice(0, 15).map((m, i) => (
                <div
                  key={i}
                  className="flex flex-col items-center p-2 border border-gray-200 dark:border-slate-700 rounded-lg"
                  title={`True: ${m.true_label} | Predicted: ${m.predicted} | Conf: ${(m.confidence * 100).toFixed(1)}%`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${m.thumbnail_png_b64}`}
                    alt={`True ${m.true_label}`}
                    className="w-10 h-10 rounded"
                    style={{ imageRendering: "pixelated" }}
                  />
                  <span className="text-xs text-gray-400 dark:text-slate-500 mt-1">
                    {m.true_label}→{m.predicted}
                  </span>
                  <span className="text-xs text-red-500 dark:text-red-400">
                    {(m.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Edge case gallery */}
        {edgeCases && edgeCases.length > 0 && (
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-4">
            <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">
              Edge Case Gallery
            </h2>
            <p className="text-sm text-gray-500 dark:text-slate-400">
              Curated transformed examples: rotations, noise, erosion, and ambiguous pairs.
            </p>
            <div className="flex flex-wrap gap-3">
              {edgeCases.map((ec) => (
                <div
                  key={ec.id}
                  className="flex flex-col items-center p-2 border border-gray-200 dark:border-slate-700 rounded-lg max-w-[80px]"
                  title={ec.note}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${ec.thumbnail_png_b64}`}
                    alt={ec.note}
                    className="w-10 h-10 rounded"
                    style={{ imageRendering: "pixelated" }}
                  />
                  <span className="text-xs text-gray-600 dark:text-slate-300 font-medium mt-1">{ec.true_label}</span>
                  <span
                    className={`text-xs ${
                      ec.predicted === ec.true_label ? "text-green-600 dark:text-green-400" : "text-red-500 dark:text-red-400"
                    }`}
                  >
                    →{ec.predicted}
                  </span>
                  <span className="text-xs text-gray-400 dark:text-slate-500 text-center leading-tight mt-0.5 truncate w-full">
                    {ec.transform.replace(/_/g, " ")}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Limitations */}
        <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-3">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">Limitations &amp; Future Work</h2>
          <ul className="text-sm text-gray-600 dark:text-slate-300 space-y-1.5 list-disc list-inside">
            <li>
              Performance degrades for digits drawn in non-MNIST style (e.g., unusual stroke order or size).
            </li>
            <li>
              The model is not robust to large rotations (&gt;30°) or heavy noise.
            </li>
            <li>
              Calibration is reasonable but model can be overconfident on ambiguous inputs.
            </li>
            <li>
              Future: temperature scaling for better calibration; data augmentation for robustness.
            </li>
            <li>
              Future: quantized ONNX model for faster browser inference.
            </li>
          </ul>
        </section>
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-3 text-center">
      <p className="text-xs text-gray-400 dark:text-slate-400 uppercase tracking-wide">{label}</p>
      <p className="text-lg font-semibold text-gray-900 dark:text-slate-50 mt-0.5">{value}</p>
    </div>
  );
}
