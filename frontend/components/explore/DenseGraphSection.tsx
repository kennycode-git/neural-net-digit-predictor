"use client";

import { useEffect, useRef, useMemo, useState } from "react";
import type { ExplorePayload, Fc2WeightsJson } from "@/lib/explore-types";
import { computeContributions } from "@/lib/explore-types";

interface Props {
  payload: ExplorePayload;
  fc2Weights: Fc2WeightsJson;
}

const SVG_W = 580;
const SVG_H = 420;
const FC1_X = 140;
const OUT_X = 440;
const TOP_K = 10;

// Minimum probability (as fraction) for an output class to receive edges
const PROB_THRESHOLD = 0.0005; // 0.05%

function neuronFill(activation: number, maxAct: number): string {
  const t = Math.min(1, Math.abs(activation) / (maxAct || 1));
  if (t < 0.05) return "#9ca3af";
  const r = Math.round(209 - t * (209 - 5));
  const g = Math.round(250 - t * (250 - 150));
  const b = Math.round(229 - t * (229 - 105));
  return `rgb(${r},${g},${b})`;
}

export default function DenseGraphSection({ payload, fc2Weights }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  // false = show all qualifying classes (default); true = predicted class only
  const [showPredictedOnly, setShowPredictedOnly] = useState(false);

  const contributions = useMemo(
    () => computeContributions(payload.fc1, fc2Weights.fc2_weight, payload.predicted, TOP_K),
    [payload.fc1, fc2Weights.fc2_weight, payload.predicted]
  );

  const maxAct = useMemo(
    () => Math.max(...payload.fc1.map(Math.abs)),
    [payload.fc1]
  );

  const softmax = useMemo(() => {
    const arr = payload.logits;
    const max = Math.max(...arr);
    const exps = arr.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }, [payload.logits]);

  // Output classes that qualify for edge rendering in multi-class mode
  const qualifyingIndices = useMemo(
    () =>
      softmax
        .map((p, i) => ({ p, i }))
        .filter(({ p }) => p > PROB_THRESHOLD)
        .map(({ i }) => i),
    [softmax]
  );

  useEffect(() => {
    const svgEl = svgRef.current;
    const tooltipEl = tooltipRef.current;
    if (!svgEl || !tooltipEl) return;

    import("d3").then((d3) => {
      const svg = d3.select(svgEl);
      svg.selectAll("*").remove();

      const fc1Count = contributions.length;

      const fc1Ys = Array.from({ length: fc1Count }, (_, i) =>
        60 + i * ((SVG_H - 100) / (fc1Count - 1))
      );
      const outYs = Array.from({ length: 10 }, (_, i) =>
        60 + i * ((SVG_H - 100) / 9)
      );

      const targetIndices = showPredictedOnly
        ? [payload.predicted]
        : qualifyingIndices;

      // Normalise edge thickness across ALL edges that will be drawn
      let maxAbsEdge = 0;
      contributions.forEach((contrib) => {
        targetIndices.forEach((oi) => {
          const absC = Math.abs(
            contrib.activation * fc2Weights.fc2_weight[oi][contrib.neuronIndex]
          );
          if (absC > maxAbsEdge) maxAbsEdge = absC;
        });
      });
      if (maxAbsEdge === 0) maxAbsEdge = 1;

      const predProb = softmax[payload.predicted] || 1;

      // ─── SVG defs: glow filter ────────────────────────────────────────────
      const defs = svg.append("defs");
      const glowFilter = defs
        .append("filter")
        .attr("id", "glow-predicted")
        .attr("x", "-60%")
        .attr("y", "-60%")
        .attr("width", "220%")
        .attr("height", "220%");
      glowFilter
        .append("feGaussianBlur")
        .attr("stdDeviation", "5")
        .attr("result", "coloredBlur");
      const feMerge = glowFilter.append("feMerge");
      feMerge.append("feMergeNode").attr("in", "coloredBlur");
      feMerge.append("feMergeNode").attr("in", "SourceGraphic");

      // ─── Layer header labels ──────────────────────────────────────────────
      svg.append("text")
        .attr("x", FC1_X).attr("y", 22)
        .attr("text-anchor", "middle")
        .attr("font-size", 11).attr("font-weight", "600")
        .attr("fill", "#6b7280")
        .text(`FC1 (top ${TOP_K} / 128)`);

      svg.append("text")
        .attr("x", OUT_X).attr("y", 22)
        .attr("text-anchor", "middle")
        .attr("font-size", 11).attr("font-weight", "600")
        .attr("fill", "#6b7280")
        .text("Output (10 classes)");

      // ─── "Activation" column label ────────────────────────────────────────
      svg.append("text")
        .attr("x", FC1_X - 16).attr("y", 38)
        .attr("text-anchor", "end")
        .attr("font-size", 9).attr("font-weight", "600")
        .attr("fill", "#6b7280")
        .text("Activation");

      // ─── Edges ────────────────────────────────────────────────────────────
      const edgeGroup = svg.append("g").attr("class", "edges");

      contributions.forEach((contrib, fi) => {
        targetIndices.forEach((oi) => {
          const edgeContrib =
            contrib.activation * fc2Weights.fc2_weight[oi][contrib.neuronIndex];
          const absC = Math.abs(edgeContrib);
          const normC = absC / maxAbsEdge;
          const isPositive = edgeContrib >= 0;

          // Scale visual weight down for low-probability classes.
          // probScale = 1.0 for predicted class, lower for minor classes.
          // Floor at 0.25 so 0.1%-prob edges remain faintly visible.
          const probScale = showPredictedOnly
            ? 1
            : Math.max(0.25, softmax[oi] / predProb);

          const baseOpacity = (0.12 + normC * 0.68) * probScale;
          const strokeW = (0.5 + normC * 3.5) * probScale;

          edgeGroup
            .append("line")
            .attr("class", "edge")
            .attr("data-fi", fi)
            .attr("data-base-opacity", baseOpacity)
            .attr("x1", FC1_X).attr("y1", fc1Ys[fi])
            .attr("x2", OUT_X).attr("y2", outYs[oi])
            .attr("stroke", isPositive ? "#6366f1" : "#f59e0b")
            .attr("stroke-width", strokeW)
            .attr("opacity", baseOpacity);
        });
      });

      // ─── FC1 nodes ────────────────────────────────────────────────────────
      contributions.forEach((contrib, i) => {
        const cy = fc1Ys[i];
        const fill = neuronFill(contrib.activation, maxAct);

        const nodeGroup = svg
          .append("g")
          .attr("class", "fc1-node")
          .attr("data-fi", i)
          .style("cursor", "pointer");

        // Enlarged invisible hit area
        nodeGroup.append("circle")
          .attr("cx", FC1_X).attr("cy", cy).attr("r", 16)
          .attr("fill", "transparent").attr("stroke", "none");

        // Visible circle
        nodeGroup.append("circle")
          .attr("cx", FC1_X).attr("cy", cy).attr("r", 10)
          .attr("fill", fill)
          .attr("stroke", "#d1d5db").attr("stroke-width", 1);

        // N-index label
        nodeGroup.append("text")
          .attr("x", FC1_X).attr("y", cy + 4)
          .attr("text-anchor", "middle")
          .attr("font-size", 7).attr("fill", "#374151")
          .style("pointer-events", "none")
          .text(`N${contrib.neuronIndex}`);

        // Activation value
        svg.append("text")
          .attr("x", FC1_X - 16).attr("y", cy + 4)
          .attr("text-anchor", "end")
          .attr("font-size", 9).attr("fill", "#9ca3af")
          .style("pointer-events", "none")
          .text(contrib.activation.toFixed(2));

        // ─── Hover ────────────────────────────────────────────────────────
        nodeGroup
          .on("mouseover", function (event) {
            edgeGroup.selectAll<SVGLineElement, unknown>(".edge")
              .attr("opacity", 0.04);
            edgeGroup
              .selectAll<SVGLineElement, unknown>(`.edge[data-fi="${i}"]`)
              .attr("opacity", function () {
                return Math.min(
                  1,
                  parseFloat(d3.select(this).attr("data-base-opacity")) * 1.6
                );
              });

            // Build tooltip — include sign explanation for predicted class
            const predContrib =
              contrib.activation * fc2Weights.fc2_weight[payload.predicted][contrib.neuronIndex];
            const isPositive = predContrib >= 0;
            const signColor = isPositive ? "#818cf8" : "#f59e0b";
            const signLabel = isPositive
              ? `supports class ${payload.predicted} ✓`
              : `suppresses class ${payload.predicted}`;
            const signNote = isPositive
              ? ""
              : `<div style="color:#9ca3af;font-size:10px;margin-top:4px">Negative weights are normal — this<br/>neuron fires for features that argue<br/>against class ${payload.predicted}.</div>`;

            const [mx, my] = d3.pointer(event, svgEl.parentElement);
            tooltipEl.style.display = "block";
            tooltipEl.style.left = `${mx + 14}px`;
            tooltipEl.style.top = `${my - 52}px`;
            tooltipEl.innerHTML =
              `<span class="font-mono font-semibold">N${contrib.neuronIndex}</span>` +
              `<br/><span style="color:#9ca3af">act:&nbsp;</span>${contrib.activation.toFixed(4)}` +
              `<br/><span style="color:#9ca3af">→ class ${payload.predicted}:&nbsp;</span>` +
              `<span style="color:${signColor}">${isPositive ? "+" : ""}${predContrib.toFixed(3)}</span>` +
              `<br/><span style="color:${signColor};font-size:10px">${signLabel}</span>` +
              signNote;
          })
          .on("mousemove", function (event) {
            const [mx, my] = d3.pointer(event, svgEl.parentElement);
            tooltipEl.style.left = `${mx + 14}px`;
            tooltipEl.style.top = `${my - 52}px`;
          })
          .on("mouseout", function () {
            edgeGroup.selectAll<SVGLineElement, unknown>(".edge").attr(
              "opacity",
              function () {
                return parseFloat(d3.select(this).attr("data-base-opacity"));
              }
            );
            tooltipEl.style.display = "none";
          });
      });

      // ─── Output nodes ─────────────────────────────────────────────────────
      outYs.forEach((cy, i) => {
        const isPredicted = i === payload.predicted;
        const prob = softmax[i];
        const r = isPredicted ? 18 : 10;

        // Glow halo for predicted node
        if (isPredicted) {
          svg.append("circle")
            .attr("cx", OUT_X).attr("cy", cy).attr("r", r)
            .attr("fill", "#818cf8")
            .attr("filter", "url(#glow-predicted)")
            .attr("opacity", 0.7);
        }

        svg.append("circle")
          .attr("cx", OUT_X).attr("cy", cy).attr("r", r)
          .attr("fill", isPredicted ? "#6366f1" : "#94a3b8")
          .attr("stroke", isPredicted ? "#4338ca" : "#cbd5e1")
          .attr("stroke-width", isPredicted ? 2 : 1);

        svg.append("text")
          .attr("x", OUT_X).attr("y", cy + 4)
          .attr("text-anchor", "middle")
          .attr("font-size", isPredicted ? 11 : 8)
          .attr("font-weight", isPredicted ? "700" : "400")
          .attr("fill", "#ffffff")
          .style("pointer-events", "none")
          .text(String(i));

        svg.append("text")
          .attr("x", OUT_X + (isPredicted ? 24 : 16)).attr("y", cy + 4)
          .attr("text-anchor", "start")
          .attr("font-size", 9)
          .attr("font-weight", isPredicted ? "600" : "400")
          .attr("fill", isPredicted ? "#6366f1" : "#9ca3af")
          .text(`${(prob * 100).toFixed(1)}%`);
      });
    });
  }, [
    contributions, maxAct, softmax, payload.predicted, payload.fc1,
    fc2Weights, showPredictedOnly, qualifyingIndices,
  ]);

  const qualifyingCount = qualifyingIndices.length;

  return (
    <section className="bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 p-6 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-100">
            Dense Layer Decision Graph
          </h2>
          <p className="text-sm text-gray-500 dark:text-slate-400 mt-1">
            Top {TOP_K} FC1 neurons by |contribution| to class{" "}
            <span className="font-semibold text-indigo-600 dark:text-indigo-400">
              {payload.predicted}
            </span>
            .{" "}
            {!showPredictedOnly && qualifyingCount > 1 && (
              <>Edges shown to all {qualifyingCount} classes with &gt;0.05% probability — thinner = lower probability class. </>
            )}
            <span className="text-indigo-500">Indigo = supports</span>,{" "}
            <span className="text-amber-500">amber = suppresses</span> — both are normal.
            Hover a node for details.
          </p>
        </div>
        <button
          onClick={() => setShowPredictedOnly((v) => !v)}
          className="shrink-0 px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 dark:border-slate-600 text-gray-600 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors whitespace-nowrap"
        >
          {showPredictedOnly ? `All qualifying classes (${qualifyingCount})` : "Predicted class only"}
        </button>
      </div>

      <div className="relative overflow-x-auto">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          className="w-full max-w-2xl mx-auto block"
          aria-label="Dense layer decision graph"
        />
        <div
          ref={tooltipRef}
          className="pointer-events-none absolute z-10 rounded-lg border border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-900 px-3 py-2 text-xs text-gray-700 dark:text-slate-200 shadow-lg leading-relaxed"
          style={{ display: "none" }}
        />
      </div>
    </section>
  );
}
