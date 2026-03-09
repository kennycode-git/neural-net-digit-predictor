"""
Tests for ml/preprocess.py — verifies shape, dtype, and value range.
"""
import numpy as np
from PIL import Image, ImageDraw

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocess import preprocess, preprocess_for_preview, PREPROCESS_VERSION


def make_digit_image(size: int = 280) -> Image.Image:
    """Create a synthetic digit-like drawing: white bg, black strokes."""
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 60, 200, 220], outline=0, width=8)
    return img


def make_blank_image(size: int = 280) -> Image.Image:
    return Image.new("L", (size, size), color=255)


class TestPreprocessVersion:
    def test_version_constant(self):
        assert PREPROCESS_VERSION == "1"


class TestPreprocessShape:
    def test_output_shape(self):
        img = make_digit_image()
        result = preprocess(img)
        assert result.shape == (1, 1, 28, 28), f"Expected (1,1,28,28), got {result.shape}"

    def test_output_dtype(self):
        img = make_digit_image()
        result = preprocess(img)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_blank_image_shape(self):
        img = make_blank_image()
        result = preprocess(img)
        assert result.shape == (1, 1, 28, 28)

    def test_rgb_input_converted(self):
        img = Image.new("RGB", (280, 280), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([60, 60, 220, 220], fill=(0, 0, 0))
        result = preprocess(img)
        assert result.shape == (1, 1, 28, 28)


class TestPreprocessRange:
    def test_values_are_finite(self):
        img = make_digit_image()
        result = preprocess(img)
        assert np.all(np.isfinite(result)), "Output contains NaN or Inf"

    def test_values_not_all_zero(self):
        img = make_digit_image()
        result = preprocess(img)
        assert not np.allclose(result, 0.0), "Output is all zeros (preprocessing may have failed)"

    def test_blank_image_values_finite(self):
        img = make_blank_image()
        result = preprocess(img)
        assert np.all(np.isfinite(result))


class TestPreprocessForPreview:
    def test_preview_shape(self):
        img = make_digit_image()
        result = preprocess_for_preview(img)
        assert result.shape == (28, 28)

    def test_preview_dtype(self):
        img = make_digit_image()
        result = preprocess_for_preview(img)
        assert result.dtype == np.float32

    def test_preview_range(self):
        img = make_digit_image()
        result = preprocess_for_preview(img)
        assert result.min() >= 0.0 - 1e-6, f"Min value {result.min()} < 0"
        assert result.max() <= 1.0 + 1e-6, f"Max value {result.max()} > 1"

    def test_strokes_have_high_values(self):
        """After inversion, strokes (originally dark) should be > 0."""
        img = make_digit_image()
        result = preprocess_for_preview(img)
        assert result.max() > 0.5, "Strokes should have high values after inversion"
