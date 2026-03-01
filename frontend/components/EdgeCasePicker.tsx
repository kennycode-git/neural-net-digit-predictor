"use client";

import { useEffect, useState } from "react";

interface EdgeCase {
  id: string;
  transform: string;
  true_label: number;
  predicted: number;
  confidence: number;
  runner_up: number;
  note: string;
  thumbnail_png_b64: string;
  probs: number[];
}

interface EdgeCasePickerProps {
  onSelect: (edgeCase: EdgeCase) => void;
}

export default function EdgeCasePicker({ onSelect }: EdgeCasePickerProps) {
  const [cases, setCases] = useState<EdgeCase[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/artifacts/edge_cases.json")
      .then((r) => r.json())
      .then((data: EdgeCase[]) => {
        setCases(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="text-sm text-gray-400 dark:text-slate-500">Loading edge cases…</div>
    );
  }

  if (cases.length === 0) {
    return (
      <div className="text-sm text-gray-400 dark:text-slate-500">
        No edge cases found. Run{" "}
        <code className="bg-gray-100 dark:bg-slate-700 dark:text-slate-300 px-1 rounded">python ml/edge_cases.py</code>{" "}
        to generate them.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500 dark:text-slate-400 font-medium uppercase tracking-wide">
        Edge Cases — click to test
      </p>
      <div className="flex flex-wrap gap-2">
        {cases.map((ec) => (
          <button
            key={ec.id}
            onClick={() => {
              setSelected(ec.id);
              onSelect(ec);
            }}
            title={ec.note}
            className={`flex flex-col items-center p-1.5 rounded-lg border transition-all ${
              selected === ec.id
                ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950/50"
                : "border-gray-200 dark:border-slate-700 hover:border-gray-400 dark:hover:border-slate-500"
            }`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${ec.thumbnail_png_b64}`}
              alt={ec.note}
              className="w-10 h-10 rounded"
              style={{ imageRendering: "pixelated" }}
            />
            <span className="text-xs text-gray-500 dark:text-slate-400 mt-0.5">{ec.true_label}</span>
          </button>
        ))}
      </div>
      {selected && (
        <p className="text-xs text-gray-500 dark:text-slate-400">
          {cases.find((c) => c.id === selected)?.note}
        </p>
      )}
    </div>
  );
}
