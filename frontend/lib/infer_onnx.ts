/**
 * Client-side ONNX inference via onnxruntime-web.
 *
 * The InferenceSession is loaded once and reused across predictions.
 * Softmax is computed in TypeScript (model outputs raw logits).
 */

import { MODEL_PATH } from "./constants";

export interface InferenceResult {
  probs: number[];                            // length 10 — softmax probabilities
  predicted: number;                          // argmax class
  topK: { digit: number; prob: number }[];    // top 3 by prob descending
  latencyMs: number;
}

let session: import("onnxruntime-web").InferenceSession | null = null;
let loadPromise: Promise<void> | null = null;

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export async function loadModel(): Promise<void> {
  if (session !== null) return;
  if (loadPromise !== null) return loadPromise;

  loadPromise = (async () => {
    const ort = await import("onnxruntime-web");
    ort.env.logLevel = "error"; // suppress non-critical warnings (e.g. unknown CPU vendor)
    // Use WASM backend (works fully offline after first load)
    ort.env.wasm.numThreads = 1; // safe default; avoids SharedArrayBuffer requirement
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });
  })();

  return loadPromise;
}

export async function runInference(pixels: Float32Array): Promise<InferenceResult> {
  if (session === null) {
    await loadModel();
  }

  const ort = await import("onnxruntime-web");
  const inputName = session!.inputNames[0];
  const tensor = new ort.Tensor("float32", pixels, [1, 1, 28, 28]);

  const t0 = performance.now();
  const results = await session!.run({ [inputName]: tensor });
  const latencyMs = performance.now() - t0;

  const outputName = session!.outputNames[0];
  const logits = Array.from(results[outputName].data as Float32Array);
  const probs = softmax(logits);
  const predicted = probs.indexOf(Math.max(...probs));

  const indexed = probs.map((p, i) => ({ digit: i, prob: p }));
  indexed.sort((a, b) => b.prob - a.prob);
  const topK = indexed.slice(0, 3);

  return { probs, predicted, topK, latencyMs };
}
