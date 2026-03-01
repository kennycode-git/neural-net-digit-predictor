/**
 * Runs the 4-output activations ONNX model (model_activations.onnx) in-browser.
 * Loaded lazily — does not block main inference.
 *
 * Output tensor shapes:
 *   conv1Pool  Float32Array [32 * 14 * 14]  — pool after conv1
 *   conv2Pool  Float32Array [64 *  7 *  7]  — pool after conv2
 *   fc1        Float32Array [128]            — hidden layer activations
 *   logits     Float32Array [10]             — raw class scores
 */

import { ACTIVATIONS_MODEL_PATH } from "./constants";

export interface ActivationResult {
  conv1Pool: Float32Array; // [32 × 14 × 14]
  conv2Pool: Float32Array; // [64 × 7  × 7 ]
  fc1: Float32Array;       // [128]
  logits: Float32Array;    // [10]
}

let session: import("onnxruntime-web").InferenceSession | null = null;
let loadPromise: Promise<void> | null = null;

export async function loadActivationsModel(): Promise<void> {
  if (session !== null) return;
  if (loadPromise !== null) return loadPromise;

  loadPromise = (async () => {
    const ort = await import("onnxruntime-web");
    session = await ort.InferenceSession.create(ACTIVATIONS_MODEL_PATH, {
      executionProviders: ["wasm"],
    });
  })();

  return loadPromise;
}

export async function runActivations(
  pixels: Float32Array
): Promise<ActivationResult | null> {
  try {
    if (session === null) await loadActivationsModel();
    const ort = await import("onnxruntime-web");
    const inputName = session!.inputNames[0];
    const tensor = new ort.Tensor("float32", pixels, [1, 1, 28, 28]);
    const results = await session!.run({ [inputName]: tensor });

    return {
      conv1Pool: new Float32Array(results["conv1_pool"].data as Float32Array),
      conv2Pool: new Float32Array(results["conv2_pool"].data as Float32Array),
      fc1: new Float32Array(results["fc1"].data as Float32Array),
      logits: new Float32Array(results["logits"].data as Float32Array),
    };
  } catch (e) {
    console.error("Activations inference failed:", e);
    return null;
  }
}
