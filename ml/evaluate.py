"""
Evaluate DigitCNN: generate confusion matrix + misclassified.json.

Usage:
    python ml/evaluate.py [--model-path ml/output/model.pt]

Outputs (ml/output/ and frontend/public/artifacts/):
    confusion_matrix.png
    misclassified.json
"""
import argparse
import base64
import io
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import DigitCNN
from preprocess import MNIST_MEAN, MNIST_STD

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FRONTEND_ARTIFACTS = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "public", "artifacts"
)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs, all_images = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = logits.argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(probs)
        all_images.append(images.cpu())
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
        torch.cat(all_images).numpy(),
    )


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix — MNIST Test Set", fontsize=14)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def image_to_base64(img_arr_chw: np.ndarray, mean=MNIST_MEAN, std=MNIST_STD) -> str:
    """Convert normalized CHW tensor to base64 PNG."""
    img = img_arr_chw[0]  # HW
    img = img * std + mean  # de-normalize
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L").resize((56, 56), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_misclassified(preds, labels, probs, images, output_path, top_n=20):
    wrong_mask = preds != labels
    wrong_indices = np.where(wrong_mask)[0]
    wrong_conf = probs[wrong_indices, preds[wrong_indices]]
    sorted_idx = wrong_indices[np.argsort(-wrong_conf)][:top_n]

    records = []
    for idx in sorted_idx:
        records.append(
            {
                "true_label": int(labels[idx]),
                "predicted": int(preds[idx]),
                "confidence": float(probs[idx, preds[idx]]),
                "runner_up": int(np.argsort(probs[idx])[-2]),
                "thumbnail_png_b64": image_to_base64(images[idx]),
            }
        )

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {output_path} ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "model.pt"))
    parser.add_argument(
        "--data-dir", default=os.path.join(os.path.dirname(__file__), "data")
    )
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

    preds, labels, probs, images = collect_predictions(model, val_loader, device)
    acc = (preds == labels).mean()
    print(f"Test accuracy: {acc:.4f}")

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    save_confusion_matrix(labels, preds, cm_path)
    shutil.copy2(cm_path, os.path.join(FRONTEND_ARTIFACTS, "confusion_matrix.png"))

    mc_path = os.path.join(OUTPUT_DIR, "misclassified.json")
    save_misclassified(preds, labels, probs, images, mc_path)
    shutil.copy2(mc_path, os.path.join(FRONTEND_ARTIFACTS, "misclassified.json"))


if __name__ == "__main__":
    main()
