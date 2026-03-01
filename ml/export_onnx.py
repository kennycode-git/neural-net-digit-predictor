"""
Export trained DigitCNN to ONNX format and copy artifacts to frontend.

Usage:
    python ml/export_onnx.py [--model-path ml/output/model.pt]

Outputs:
    ml/output/model.onnx                        (inference model)
    ml/output/model_activations.onnx            (4-output model for NetworkViz)
    frontend/public/artifacts/model.onnx
    frontend/public/artifacts/model_activations.onnx
"""
import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DigitCNN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FRONTEND_ARTIFACTS = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "public", "artifacts"
)


class DigitCNNActivations(nn.Module):
    """Thin wrapper that returns intermediate layer outputs for visualisation."""

    def __init__(self, base: DigitCNN) -> None:
        super().__init__()
        self.conv1 = base.conv1
        self.conv2 = base.conv2
        self.pool = base.pool
        self.fc1 = base.fc1
        self.fc2 = base.fc2

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        conv1_pool = self.pool(x)                          # (B, 32, 14, 14)
        x = F.relu(self.conv2(conv1_pool))
        conv2_pool = self.pool(x)                          # (B, 64,  7,  7)
        fc1_out = F.relu(self.fc1(conv2_pool.flatten(1)))  # (B, 128)
        logits = self.fc2(fc1_out)                         # (B, 10)
        return conv1_pool, conv2_pool, fc1_out, logits


def _load_base(model_path: str) -> DigitCNN:
    model = DigitCNN()
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model


def export(model_path: str) -> str:
    model = _load_base(model_path)
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"Exported: {onnx_path}")
    return onnx_path


def export_activations(model_path: str) -> str:
    base = _load_base(model_path)
    act_model = DigitCNNActivations(base)
    act_model.eval()

    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    onnx_path = os.path.join(OUTPUT_DIR, "model_activations.onnx")

    torch.onnx.export(
        act_model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["conv1_pool", "conv2_pool", "fc1", "logits"],
        dynamic_axes={
            "input":      {0: "batch_size"},
            "conv1_pool": {0: "batch_size"},
            "conv2_pool": {0: "batch_size"},
            "fc1":        {0: "batch_size"},
            "logits":     {0: "batch_size"},
        },
    )
    print(f"Exported activations model: {onnx_path}")
    return onnx_path


def export_fc2_weights(model_path: str) -> str:
    import json
    model = _load_base(model_path)
    weights = model.fc2.weight.detach().cpu().tolist()  # [10][128]
    bias = model.fc2.bias.detach().cpu().tolist()        # [10]
    out = {"fc2_weight": weights, "fc2_bias": bias, "shape": [10, 128]}
    json_path = os.path.join(OUTPUT_DIR, "fc2_weights.json")
    with open(json_path, "w") as f:
        json.dump(out, f)
    print(f"Exported FC2 weights: {json_path}")
    copy_to_frontend(json_path)
    return json_path


def copy_to_frontend(onnx_path: str, filename: str | None = None) -> None:
    os.makedirs(FRONTEND_ARTIFACTS, exist_ok=True)
    dest_name = filename or os.path.basename(onnx_path)
    dest = os.path.join(FRONTEND_ARTIFACTS, dest_name)
    shutil.copy2(onnx_path, dest)
    print(f"Copied to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join(OUTPUT_DIR, "model.pt"),
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    onnx_path = export(args.model_path)
    copy_to_frontend(onnx_path)

    act_path = export_activations(args.model_path)
    copy_to_frontend(act_path)

    export_fc2_weights(args.model_path)

    print("ONNX export complete.")


if __name__ == "__main__":
    main()
