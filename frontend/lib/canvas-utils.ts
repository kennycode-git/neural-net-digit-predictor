/** Hot colourmap: 0→black, 0.33→red, 0.66→yellow, 1→white */
export function hotColor(v: number): [number, number, number] {
  const r = Math.min(1, v * 3);
  const g = Math.min(1, Math.max(0, v * 3 - 1));
  const b = Math.min(1, Math.max(0, v * 3 - 2));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/** Diverging colourmap: negative→blue, 0→white, positive→red */
export function divergingColor(v: number): [number, number, number] {
  if (v >= 0.5) {
    const t = (v - 0.5) * 2;
    return [255, Math.round((1 - t) * 255), Math.round((1 - t) * 255)];
  } else {
    const t = (0.5 - v) * 2;
    return [Math.round((1 - t) * 255), Math.round((1 - t) * 255), 255];
  }
}

export function drawFeatureMap(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  w: number,
  h: number,
  colorFn: (v: number) => [number, number, number]
): void {
  const ctx = canvas.getContext("2d")!;
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const range = max - min || 1;
  const imgData = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = (data[i] - min) / range;
    const [r, g, b] = colorFn(v);
    imgData.data[i * 4 + 0] = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}
