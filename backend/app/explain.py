"""
Explainability engine using Captum.
Lazy-loads model on first call.
"""
import base64
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_model = None
_device = torch.device("cpu")

# Resolve model path relative to this file
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ml", "output", "model.pt"
)


def _get_model():
    global _model
    if _model is not None:
        return _model

    import sys
    ml_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    from models import DigitCNN

    m = DigitCNN()
    state = torch.load(_MODEL_PATH, map_location=_device)
    m.load_state_dict(state)
    m.eval()
    _model = m
    return _model


def _pixels_to_tensor(pixels: list[list[float]]) -> torch.Tensor:
    """Convert 28×28 list[list[float]] to (1,1,28,28) tensor with grad."""
    arr = np.array(pixels, dtype=np.float32)  # (28,28)
    # Apply MNIST normalization (backend receives inverted [0,1] values)
    from preprocess import MNIST_MEAN, MNIST_STD  # type: ignore[import]
    arr = (arr - MNIST_MEAN) / MNIST_STD
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return t.requires_grad_(True)


def _heatmap_to_png_b64(heatmap: np.ndarray) -> str:
    """Render a (28,28) heatmap as a base64 PNG using a diverging colormap."""
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=100)
    vmax = np.abs(heatmap).max() or 1.0
    ax.imshow(
        heatmap,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def saliency_heatmap(pixels: list[list[float]], target: int) -> str:
    """Vanilla saliency: gradient of target score w.r.t. input."""
    from captum.attr import Saliency  # type: ignore[import]

    model = _get_model()
    inp = _pixels_to_tensor(pixels)

    saliency = Saliency(model)
    attr = saliency.attribute(inp, target=target)
    heatmap = attr.squeeze().detach().numpy()
    return _heatmap_to_png_b64(heatmap)


def gradcam_heatmap(pixels: list[list[float]], target: int) -> str:
    """GradCAM on the second conv layer, upsampled to 28×28."""
    from captum.attr import LayerGradCam  # type: ignore[import]

    model = _get_model()
    inp = _pixels_to_tensor(pixels)

    layer_gc = LayerGradCam(model, model.conv2)
    attr = layer_gc.attribute(inp, target=target)
    # attr is (1, 64, 7, 7) — upsample to 28×28
    import torch.nn.functional as F
    attr_up = F.interpolate(attr, size=(28, 28), mode="bilinear", align_corners=False)
    heatmap = attr_up.squeeze().detach().numpy()
    return _heatmap_to_png_b64(heatmap)
