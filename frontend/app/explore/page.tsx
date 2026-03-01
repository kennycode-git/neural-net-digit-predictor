"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";
import type { ExplorePayload, Fc2WeightsJson } from "@/lib/explore-types";
import { FC2_WEIGHTS_PATH } from "@/lib/constants";
import DecisionSummary from "@/components/explore/DecisionSummary";
import ConvMapsSection from "@/components/explore/ConvMapsSection";
import DenseGraphSection from "@/components/explore/DenseGraphSection";

export default function ExplorePage() {
  const [payload, setPayload] = useState<ExplorePayload | null>(null);
  const [fc2Weights, setFc2Weights] = useState<Fc2WeightsJson | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem("explore_payload");
    if (!raw) {
      setError("No inference data found. Draw a digit on the main page first, then click \"Explore Decision Path\".");
      setLoading(false);
      return;
    }

    const parsed: ExplorePayload = JSON.parse(raw);

    fetch(FC2_WEIGHTS_PATH)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((w: Fc2WeightsJson) => {
        setPayload(parsed);
        setFc2Weights(w);
        setLoading(false);
      })
      .catch(() => {
        setError(
          "Failed to load FC2 weights. Run: cd ml && python export_onnx.py"
        );
        setLoading(false);
      });
  }, []);

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-slate-900 transition-colors">
      <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">

        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <Link
              href="/"
              className="text-indigo-600 dark:text-indigo-400 text-sm hover:underline"
            >
              ← Back to Lab
            </Link>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50 mt-2">
              Decision Explorer
            </h1>
            <p className="text-gray-500 dark:text-slate-400 text-sm mt-1">
              Visualising how the network reached its prediction
            </p>
          </div>
          <ThemeToggle />
        </div>

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-24">
            <div className="w-8 h-8 border-4 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
            <span className="ml-3 text-gray-500 dark:text-slate-400">Loading activations…</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-lg text-sm text-red-700 dark:text-red-300">
            {error}
          </div>
        )}

        {/* Content */}
        {payload && fc2Weights && (
          <>
            <DecisionSummary payload={payload} fc2Weights={fc2Weights} />
            <ConvMapsSection payload={payload} />
            <DenseGraphSection payload={payload} fc2Weights={fc2Weights} />
          </>
        )}

      </div>
    </main>
  );
}
