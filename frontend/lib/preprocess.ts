/**
 * Preprocessing contract v1.
 * Must match ml/preprocess.py exactly.
 *
 * Pipeline:
 *  1. Grayscale (luminance: 0.299R + 0.587G + 0.114B)
 *  2. Crop bounding box of non-white pixels + 2px padding
 *  3. Fit into 20×20 preserving aspect ratio (centered)
 *  4. Pad to 28×28 centered
 *  5. Normalize [0, 1]
 *  6. Invert (white bg → 0, black stroke → 1)
 *  7. (x - MNIST_MEAN) / MNIST_STD
 *
 * Returns:
 *  - tensor: Float32Array [1, 1, 28, 28]  (model input)
 *  - preview: Uint8ClampedArray [28 × 28 × 4]  (RGBA, before step 7, for display)
 */

import { MNIST_MEAN, MNIST_STD } from "./constants";

export interface PreprocessResult {
  tensor: Float32Array;    // shape [1,1,28,28] — normalized model input
  preview: ImageData;      // 28×28 RGBA — "what the model sees" before mean/std norm
}

/** Grayscale luminance of an RGBA pixel. */
function toLuminance(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Fit an ImageData into a 20×20 canvas preserving aspect ratio, centered.
 * A tall narrow '1' stays narrow instead of being squished into a fat block.
 */
async function fitInto20x20(src: ImageData): Promise<ImageData> {
  const scale = 20 / Math.max(src.width, src.height);
  const newW = Math.max(1, Math.round(src.width * scale));
  const newH = Math.max(1, Math.round(src.height * scale));

  // Draw src at scaled size
  const srcCanvas = new OffscreenCanvas(src.width, src.height);
  srcCanvas.getContext("2d")!.putImageData(src, 0, 0);
  const scaledCanvas = new OffscreenCanvas(newW, newH);
  scaledCanvas.getContext("2d")!.drawImage(srcCanvas, 0, 0, newW, newH);

  // Center into a white 20×20 canvas
  const out = new OffscreenCanvas(20, 20);
  const outCtx = out.getContext("2d")!;
  outCtx.fillStyle = "white";
  outCtx.fillRect(0, 0, 20, 20);
  const offC = Math.floor((20 - newW) / 2);
  const offR = Math.floor((20 - newH) / 2);
  outCtx.drawImage(scaledCanvas, offC, offR);
  return outCtx.getImageData(0, 0, 20, 20);
}

export async function preprocessCanvas(
  canvas: HTMLCanvasElement
): Promise<PreprocessResult> {
  const ctx = canvas.getContext("2d")!;
  const raw = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { width: W, height: H } = raw;
  const data = raw.data; // RGBA flat array

  // --- Step 1: Grayscale ---
  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    gray[i] = toLuminance(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
  }

  // --- Step 2: Bounding box of non-white pixels (gray < 250) ---
  let rMin = H, rMax = -1, cMin = W, cMax = -1;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      if (gray[r * W + c] < 250) {
        if (r < rMin) rMin = r;
        if (r > rMax) rMax = r;
        if (c < cMin) cMin = c;
        if (c > cMax) cMax = c;
      }
    }
  }

  let croppedData: ImageData;
  if (rMax === -1) {
    // Blank canvas — use full image
    croppedData = raw;
  } else {
    const pad = 2;
    const r0 = Math.max(0, rMin - pad);
    const r1 = Math.min(H, rMax + pad + 1);
    const c0 = Math.max(0, cMin - pad);
    const c1 = Math.min(W, cMax + pad + 1);
    const cropW = c1 - c0;
    const cropH = r1 - r0;

    const cropCanvas = new OffscreenCanvas(cropW, cropH);
    const cropCtx = cropCanvas.getContext("2d")!;
    cropCtx.drawImage(
      canvas,
      c0, r0, cropW, cropH,
      0, 0, cropW, cropH
    );
    croppedData = cropCtx.getImageData(0, 0, cropW, cropH);
  }

  // --- Step 3: Fit into 20×20 preserving aspect ratio ---
  const resized20 = await fitInto20x20(croppedData);

  // --- Step 4: Pad to 28×28 (centered) ---
  const canvas28 = new OffscreenCanvas(28, 28);
  const ctx28 = canvas28.getContext("2d")!;
  // fill white
  ctx28.fillStyle = "white";
  ctx28.fillRect(0, 0, 28, 28);
  const offset = Math.floor((28 - 20) / 2); // = 4
  const tmp = new OffscreenCanvas(20, 20);
  tmp.getContext("2d")!.putImageData(resized20, 0, 0);
  ctx28.drawImage(tmp, offset, offset);
  const padded = ctx28.getImageData(0, 0, 28, 28);

  // --- Step 5+6: Grayscale → [0,1] → invert ---
  const previewRaw = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const d = padded.data;
    const lum = toLuminance(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]);
    previewRaw[i] = 1.0 - lum / 255.0; // invert
  }

  // Preview ImageData (before mean/std norm) — grayscale RGBA
  const previewRGBA = new Uint8ClampedArray(28 * 28 * 4);
  for (let i = 0; i < 28 * 28; i++) {
    const v = Math.round(previewRaw[i] * 255);
    previewRGBA[i * 4 + 0] = v;
    previewRGBA[i * 4 + 1] = v;
    previewRGBA[i * 4 + 2] = v;
    previewRGBA[i * 4 + 3] = 255;
  }
  const preview = new ImageData(previewRGBA, 28, 28);

  // --- Step 7: Normalize (x - mean) / std ---
  const tensor = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    tensor[i] = (previewRaw[i] - MNIST_MEAN) / MNIST_STD;
  }

  return { tensor, preview };
}
