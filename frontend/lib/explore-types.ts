export interface ExplorePayload {
  predicted: number;
  probs: number[];
  topK: { digit: number; prob: number }[];
  conv1Pool: number[];    // Array.from(Float32Array), length 32*14*14 = 6272
  conv2Pool: number[];    // Array.from(Float32Array), length 64*7*7 = 3136
  fc1: number[];          // length 128
  logits: number[];       // length 10
  previewDataUrl: string; // 28×28 canvas.toDataURL("image/png")
}

export interface Fc2WeightsJson {
  fc2_weight: number[][];  // [10][128]
  fc2_bias: number[];      // [10]
  shape: [number, number]; // [10, 128]
}

export interface NeuronContribution {
  neuronIndex: number;
  activation: number;
  weight: number;
  contribution: number;     // activation × weight (signed)
  absContribution: number;
}

export function computeContributions(
  fc1: number[],
  fc2Weight: number[][],
  predicted: number,
  topK: number = 10
): NeuronContribution[] {
  return fc1
    .map((activation, i) => ({
      neuronIndex: i,
      activation,
      weight: fc2Weight[predicted][i],
      contribution: activation * fc2Weight[predicted][i],
      absContribution: Math.abs(activation * fc2Weight[predicted][i]),
    }))
    .sort((a, b) => b.absContribution - a.absContribution)
    .slice(0, topK);
}
