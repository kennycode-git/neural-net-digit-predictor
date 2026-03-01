"""
Preprocessing contract v1.
Must match frontend/lib/preprocess.ts exactly.
"""
import numpy as np
from PIL import Image

PREPROCESS_VERSION = "1"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _fit_into_20x20(crop_img: Image.Image) -> np.ndarray:
    """
    Resize a cropped digit image to fit within 20×20 while preserving aspect
    ratio, then center it in a 20×20 white canvas.

    This matches MNIST convention: a narrow '1' stays narrow rather than being
    squished into a fat block that resembles an '8'.
    """
    w, h = crop_img.size
    scale = 20 / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = crop_img.resize((new_w, new_h), Image.BILINEAR)

    canvas = np.ones((20, 20), dtype=np.float32) * 255.0
    off_r = (20 - new_h) // 2
    off_c = (20 - new_w) // 2
    canvas[off_r : off_r + new_h, off_c : off_c + new_w] = np.array(
        resized, dtype=np.float32
    )
    return canvas


def preprocess(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image (any mode/size) to a model-ready float32 tensor.

    Steps:
    1. Grayscale
    2. Crop bounding box of drawn pixels + 2px padding
    3. Fit into 20×20 preserving aspect ratio (centered)
    4. Pad to 28×28 (centered)
    5. Normalize to [0, 1]
    6. Invert (white bg → 0, black strokes → 1; MNIST convention)
    7. Normalize with MNIST mean/std

    Returns:
        np.ndarray of shape (1, 1, 28, 28), dtype float32
    """
    img = pil_image.convert("L")
    arr = np.array(img, dtype=np.float32)  # H×W, [0, 255]

    # --- crop to bounding box of non-white pixels ---
    mask = arr < 250
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        pad = 2
        r0 = max(0, rows[0] - pad)
        r1 = min(arr.shape[0], rows[-1] + pad + 1)
        c0 = max(0, cols[0] - pad)
        c1 = min(arr.shape[1], cols[-1] + pad + 1)
        arr = arr[r0:r1, c0:c1]

    # --- fit into 20×20 preserving aspect ratio ---
    crop_img = Image.fromarray(arr.astype(np.uint8))
    arr20 = _fit_into_20x20(crop_img)

    # --- pad to 28×28, centered ---
    canvas = np.ones((28, 28), dtype=np.float32) * 255.0
    off = (28 - 20) // 2  # = 4
    canvas[off : off + 20, off : off + 20] = arr20

    # --- normalize [0, 1] ---
    x = canvas / 255.0

    # --- invert: white bg → 0, black stroke → 1 ---
    x = 1.0 - x

    # --- MNIST mean/std normalization ---
    x = (x - MNIST_MEAN) / MNIST_STD

    return x.astype(np.float32).reshape(1, 1, 28, 28)


def preprocess_for_preview(pil_image: Image.Image) -> np.ndarray:
    """
    Same pipeline but stops before mean/std norm.
    Returns (28, 28) float32 array in [0, 1] (inverted).
    Useful for generating the "what the model sees" preview.
    """
    img = pil_image.convert("L")
    arr = np.array(img, dtype=np.float32)

    mask = arr < 250
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        pad = 2
        r0 = max(0, rows[0] - pad)
        r1 = min(arr.shape[0], rows[-1] + pad + 1)
        c0 = max(0, cols[0] - pad)
        c1 = min(arr.shape[1], cols[-1] + pad + 1)
        arr = arr[r0:r1, c0:c1]

    crop_img = Image.fromarray(arr.astype(np.uint8))
    arr20 = _fit_into_20x20(crop_img)

    canvas = np.ones((28, 28), dtype=np.float32) * 255.0
    off = (28 - 20) // 2
    canvas[off : off + 20, off : off + 20] = arr20

    x = canvas / 255.0
    x = 1.0 - x  # invert
    return x.astype(np.float32)
