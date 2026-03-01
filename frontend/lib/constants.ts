// Preprocessing contract — must match ml/preprocess.py exactly
export const PREPROCESS_VERSION = "1" as const;
export const MNIST_MEAN = 0.1307;
export const MNIST_STD = 0.3081;

// Model artifact paths (served from /public/artifacts/)
export const MODEL_PATH = "/artifacts/model.onnx";
export const ACTIVATIONS_MODEL_PATH = "/artifacts/model_activations.onnx";
export const FC2_WEIGHTS_PATH = "/artifacts/fc2_weights.json";

// Backend URL — override with NEXT_PUBLIC_BACKEND_URL env var
export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
