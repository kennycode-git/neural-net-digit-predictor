"""
Compute calibration metrics: reliability diagram data + ECE.

Usage:
    python ml/calibration.py [--model-path ml/output/model.pt]

Outputs (ml/output/ and frontend/public/artifacts/):
    reliability.json  — bins, ece, and per-bin data
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import DigitCNN
from preprocess import MNIST_MEAN, MNIST_STD

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FRONTEND_ARTIFACTS = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "public", "artifacts"
)


@torch.no_grad()
def get_confidences_and_correctness(model, loader, device):
    model.eval()
    all_conf, all_correct = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probs = torch.softmax(model(images), dim=1)
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()
        all_conf.append(conf.cpu())
        all_correct.append(correct.cpu())
    return torch.cat(all_conf).numpy(), torch.cat(all_correct).numpy()


def compute_calibration(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_data.append(
                {
                    "bin_lower": round(lo, 3),
                    "bin_upper": round(hi, 3),
                    "avg_confidence": round((lo + hi) / 2, 3),
                    "accuracy": None,
                    "count": 0,
                }
            )
            continue
        avg_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        ece += (count / n) * abs(acc - avg_conf)
        bin_data.append(
            {
                "bin_lower": round(lo, 3),
                "bin_upper": round(hi, 3),
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(acc, 4),
                "count": count,
            }
        )

    return bin_data, float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "model.pt"))
    parser.add_argument(
        "--data-dir", default=os.path.join(os.path.dirname(__file__), "data")
    )
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRONTEND_ARTIFACTS, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    val_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)

    confidences, correct = get_confidences_and_correctness(model, val_loader, device)
    bin_data, ece = compute_calibration(confidences, correct, args.n_bins)

    result = {
        "ece": round(ece, 6),
        "n_bins": args.n_bins,
        "n_samples": len(confidences),
        "bins": bin_data,
    }

    out_path = os.path.join(OUTPUT_DIR, "reliability.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"ECE = {ece:.4f}")
    print(f"Saved: {out_path}")

    shutil.copy2(out_path, os.path.join(FRONTEND_ARTIFACTS, "reliability.json"))


if __name__ == "__main__":
    main()
