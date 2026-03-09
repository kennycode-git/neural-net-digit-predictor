"""
Generate a curated set of edge-case examples for the model card failure gallery.

Transformations applied to MNIST test samples:
  - Rotation ±15°, ±30°, ±45°
  - Gaussian noise σ=0.1, 0.3
  - Erosion (thinning strokes)
  - Ambiguous pairs (e.g., 4↔9, 3↔8, 1↔7)

Usage:
    python ml/edge_cases.py [--model-path ml/output/model.pt]

Outputs:
    ml/output/edge_cases.json
    frontend/public/artifacts/edge_cases.json
"""
import argparse
import base64
import io
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from torchvision import datasets, transforms

from models import DigitCNN
from preprocess import MNIST_MEAN, MNIST_STD

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FRONTEND_ARTIFACTS = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "public", "artifacts"
)

# Pairs of digits that are visually ambiguous
AMBIGUOUS_PAIRS = [(4, 9), (3, 8), (1, 7), (5, 6), (0, 8)]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Denormalize a (1, 28, 28) float tensor → PIL L image."""
    arr = t.squeeze().numpy()
    arr = arr * MNIST_STD + MNIST_MEAN  # de-normalize
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def pil_to_base64(pil_img: Image.Image, size: int = 56) -> str:
    pil_img = pil_img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL L image → normalized (1,1,28,28) tensor for inference."""
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict(model, tensor: torch.Tensor):
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze()
    conf, pred = probs.max(0)
    runner_up = int(probs.argsort()[-2])
    return int(pred), float(conf), runner_up, probs.numpy().tolist()


def apply_rotation(pil_img: Image.Image, angle: float) -> Image.Image:
    return pil_img.rotate(angle, fillcolor=0, resample=Image.BILINEAR)


def apply_noise(pil_img: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(pil_img, dtype=np.float32)
    noise = np.random.normal(0, sigma * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def apply_erosion(pil_img: Image.Image) -> Image.Image:
    arr = np.array(pil_img)
    binary = arr > 50
    eroded = binary_erosion(binary, structure=np.ones((2, 2)))
    arr_out = (eroded * 255).astype(np.uint8)
    return Image.fromarray(arr_out, mode="L")


def find_ambiguous_samples(dataset, target_pairs, n_per_pair=2):
    """Find MNIST samples where a pair of digit classes look similar."""
    indices_by_label = {d: [] for pair in target_pairs for d in pair}
    for idx, (_, label) in enumerate(dataset):
        if label in indices_by_label and len(indices_by_label[label]) < 20:
            indices_by_label[label].append(idx)

    selected = []
    for a, b in target_pairs:
        for idx in indices_by_label[a][:n_per_pair]:
            selected.append((idx, a, b, f"visually ambiguous with {b}"))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "model.pt"))
    parser.add_argument(
        "--data-dir", default=os.path.join(os.path.dirname(__file__), "data")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRONTEND_ARTIFACTS, exist_ok=True)

    device = torch.device("cpu")  # edge cases done on CPU
    model = DigitCNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    raw_transform = transforms.ToTensor()  # for PIL access
    val_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    val_ds_raw = datasets.MNIST(args.data_dir, train=False, transform=raw_transform)

    examples = []

    # -- Rotation examples (pick digit 7, 1, 4 for visual clarity) --
    rotation_targets = {7: None, 1: None, 4: None}
    for idx, (_, label) in enumerate(val_ds_raw):
        if label in rotation_targets and rotation_targets[label] is None:
            rotation_targets[label] = idx
        if all(v is not None for v in rotation_targets.values()):
            break

    for angle in [15, 30, -30, 45, -45]:
        for digit, idx in rotation_targets.items():
            img_t, _ = val_ds[idx]
            pil = tensor_to_pil(img_t)
            rotated = apply_rotation(pil, angle)
            inp = pil_to_tensor(rotated)
            pred, conf, runner_up, probs = predict(model, inp)
            examples.append(
                {
                    "id": f"rot_{angle}_{digit}",
                    "transform": f"rotation_{angle}deg",
                    "true_label": digit,
                    "predicted": pred,
                    "confidence": round(conf, 4),
                    "runner_up": runner_up,
                    "probs": [round(p, 4) for p in probs],
                    "note": f"Digit {digit} rotated {angle}°",
                    "thumbnail_png_b64": pil_to_base64(rotated),
                }
            )
            if len(examples) >= 8:
                break
        if len(examples) >= 8:
            break

    # -- Noise examples --
    noise_targets = {8: None, 3: None}
    for idx, (_, label) in enumerate(val_ds_raw):
        if label in noise_targets and noise_targets[label] is None:
            noise_targets[label] = idx
        if all(v is not None for v in noise_targets.values()):
            break

    for sigma in [0.15, 0.30]:
        for digit, idx in noise_targets.items():
            img_t, _ = val_ds[idx]
            pil = tensor_to_pil(img_t)
            noisy = apply_noise(pil, sigma)
            inp = pil_to_tensor(noisy)
            pred, conf, runner_up, probs = predict(model, inp)
            examples.append(
                {
                    "id": f"noise_{sigma}_{digit}",
                    "transform": f"gaussian_noise_sigma_{sigma}",
                    "true_label": digit,
                    "predicted": pred,
                    "confidence": round(conf, 4),
                    "runner_up": runner_up,
                    "probs": [round(p, 4) for p in probs],
                    "note": f"Digit {digit} with Gaussian noise σ={sigma}",
                    "thumbnail_png_b64": pil_to_base64(noisy),
                }
            )

    # -- Erosion examples --
    erosion_targets = {5: None, 2: None}
    for idx, (_, label) in enumerate(val_ds_raw):
        if label in erosion_targets and erosion_targets[label] is None:
            erosion_targets[label] = idx
        if all(v is not None for v in erosion_targets.values()):
            break

    for digit, idx in erosion_targets.items():
        img_t, _ = val_ds[idx]
        pil = tensor_to_pil(img_t)
        eroded = apply_erosion(pil)
        inp = pil_to_tensor(eroded)
        pred, conf, runner_up, probs = predict(model, inp)
        examples.append(
            {
                "id": f"erosion_{digit}",
                "transform": "stroke_erosion",
                "true_label": digit,
                "predicted": pred,
                "confidence": round(conf, 4),
                "runner_up": runner_up,
                "probs": [round(p, 4) for p in probs],
                "note": f"Digit {digit} with thinned strokes (erosion)",
                "thumbnail_png_b64": pil_to_base64(eroded),
            }
        )

    # -- Ambiguous pairs --
    for a, b in AMBIGUOUS_PAIRS[:3]:
        for idx, (_, label) in enumerate(val_ds_raw):
            if label == a:
                img_t, _ = val_ds[idx]
                pil = tensor_to_pil(img_t)
                inp = pil_to_tensor(pil)
                pred, conf, runner_up, probs = predict(model, inp)
                if runner_up == b and conf < 0.80:
                    examples.append(
                        {
                            "id": f"ambiguous_{a}_{b}",
                            "transform": "none",
                            "true_label": a,
                            "predicted": pred,
                            "confidence": round(conf, 4),
                            "runner_up": runner_up,
                            "probs": [round(p, 4) for p in probs],
                            "note": f"Digit {a} visually similar to {b} (runner-up confidence {probs[b]:.2f})",
                            "thumbnail_png_b64": pil_to_base64(pil),
                        }
                    )
                    break

    out_path = os.path.join(OUTPUT_DIR, "edge_cases.json")
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Generated {len(examples)} edge cases -> {out_path}")
    shutil.copy2(out_path, os.path.join(FRONTEND_ARTIFACTS, "edge_cases.json"))


if __name__ == "__main__":
    main()
