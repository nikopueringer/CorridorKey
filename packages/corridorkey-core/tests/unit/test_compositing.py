"""Tests for corridorkey_core.compositing.

All tests use synthetic numpy arrays and run without GPU or model files.
"""

import numpy as np
import pytest
import torch
from corridorkey_core.compositing import (
    clean_matte,
    composite_premul,
    composite_straight,
    create_checkerboard,
    despill,
    linear_to_srgb,
    premultiply,
    srgb_to_linear,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(h: int, w: int, r: float, g: float, b: float) -> np.ndarray:
    """Create a solid-color float32 [H, W, 3] array."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


# ---------------------------------------------------------------------------
# sRGB <-> linear roundtrip
# ---------------------------------------------------------------------------


class TestColorSpaceConversion:
    def test_roundtrip_numpy(self):
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-5)

    def test_roundtrip_tensor(self):
        x = torch.linspace(0.0, 1.0, 256)
        result = srgb_to_linear(linear_to_srgb(x))
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, x, atol=1e-5)

    def test_linear_to_srgb_black_and_white(self):
        x = np.array([0.0, 1.0], dtype=np.float32)
        result = linear_to_srgb(x)
        assert np.isclose(result[0], 0.0, atol=1e-6)
        assert np.isclose(result[1], 1.0, atol=1e-4)

    def test_srgb_to_linear_clamps_negative(self):
        x = np.array([-0.5], dtype=np.float32)
        result = srgb_to_linear(x)
        assert result[0] >= 0.0

    def test_linear_to_srgb_clamps_negative(self):
        x = np.array([-0.5], dtype=np.float32)
        result = linear_to_srgb(x)
        assert result[0] >= 0.0

    def test_linear_to_srgb_is_brighter(self):
        # sRGB-encoded values are always >= linear for values in (0, 1)
        x = np.linspace(0.01, 0.99, 100, dtype=np.float32)
        assert np.all(linear_to_srgb(x) >= x)


# ---------------------------------------------------------------------------
# premultiply
# ---------------------------------------------------------------------------


class TestPremultiply:
    def test_full_alpha_unchanged(self):
        fg = _solid(4, 4, 0.8, 0.2, 0.5)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        result = premultiply(fg, alpha)
        assert np.allclose(result, fg)

    def test_zero_alpha_gives_black(self):
        fg = _solid(4, 4, 1.0, 1.0, 1.0)
        alpha = np.zeros((4, 4, 1), dtype=np.float32)
        result = premultiply(fg, alpha)
        assert np.allclose(result, 0.0)

    def test_half_alpha_halves_values(self):
        fg = _solid(4, 4, 1.0, 1.0, 1.0)
        alpha = np.full((4, 4, 1), 0.5, dtype=np.float32)
        result = premultiply(fg, alpha)
        assert np.allclose(result, 0.5)


# ---------------------------------------------------------------------------
# composite_straight / composite_premul
# ---------------------------------------------------------------------------


class TestCompositing:
    def test_straight_full_alpha_shows_fg(self):
        fg = _solid(4, 4, 1.0, 0.0, 0.0)
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        assert np.allclose(result, fg)

    def test_straight_zero_alpha_shows_bg(self):
        fg = _solid(4, 4, 1.0, 0.0, 0.0)
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.zeros((4, 4, 1), dtype=np.float32)
        result = composite_straight(fg, bg, alpha)
        assert np.allclose(result, bg)

    def test_premul_full_alpha_shows_fg(self):
        fg = _solid(4, 4, 1.0, 0.0, 0.0)
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.ones((4, 4, 1), dtype=np.float32)
        result = composite_premul(fg, bg, alpha)
        assert np.allclose(result, fg)

    def test_premul_zero_alpha_shows_bg(self):
        fg = _solid(4, 4, 0.0, 0.0, 0.0)  # premul: fg already multiplied by alpha=0
        bg = _solid(4, 4, 0.0, 0.0, 1.0)
        alpha = np.zeros((4, 4, 1), dtype=np.float32)
        result = composite_premul(fg, bg, alpha)
        assert np.allclose(result, bg)


# ---------------------------------------------------------------------------
# despill
# ---------------------------------------------------------------------------


class TestDespill:
    def test_no_spill_unchanged(self):
        # Pure red — no green to remove
        img = _solid(4, 4, 1.0, 0.0, 0.0)
        result = despill(img, strength=1.0)
        assert np.allclose(result, img, atol=1e-6)

    def test_green_channel_reduced(self):
        # Green-heavy image should have G reduced
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        result = despill(img, strength=1.0)
        assert np.all(result[..., 1] <= img[..., 1] + 1e-6)

    def test_zero_strength_unchanged(self):
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        result = despill(img, strength=0.0)
        assert np.allclose(result, img)

    def test_output_shape_preserved(self):
        img = _solid(8, 8, 0.3, 0.8, 0.3)
        result = despill(img)
        assert result.shape == img.shape

    def test_max_mode(self):
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        result = despill(img, green_limit_mode="max", strength=1.0)
        assert result.shape == img.shape

    def test_invalid_mode_raises(self):
        img = _solid(4, 4, 0.2, 0.9, 0.2)
        with pytest.raises(ValueError, match="green_limit_mode"):
            despill(img, green_limit_mode="median")

    def test_tensor_input(self):
        img = torch.tensor(_solid(4, 4, 0.2, 0.9, 0.2))
        result = despill(img, strength=1.0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# clean_matte
# ---------------------------------------------------------------------------


class TestCleanMatte:
    def _make_alpha_with_island(self) -> np.ndarray:
        """100x100 alpha with a large foreground blob and a small island."""
        alpha = np.zeros((100, 100), dtype=np.float32)
        alpha[10:80, 10:80] = 1.0  # large region (4900 px)
        alpha[90:93, 90:93] = 1.0  # small island (9 px)
        return alpha

    def test_removes_small_island(self):
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100, dilation=0, blur_size=0)
        # Small island region should be zeroed out
        assert np.allclose(result[90:93, 90:93], 0.0)

    def test_preserves_large_region(self):
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100, dilation=0, blur_size=0)
        # Center of large region should remain non-zero
        assert result[40, 40] > 0.0

    def test_3d_input_returns_3d(self):
        alpha = self._make_alpha_with_island()[:, :, np.newaxis]
        result = clean_matte(alpha, area_threshold=100)
        assert result.ndim == 3
        assert result.shape[2] == 1

    def test_2d_input_returns_2d(self):
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100)
        assert result.ndim == 2

    def test_output_range(self):
        alpha = self._make_alpha_with_island()
        result = clean_matte(alpha, area_threshold=100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# create_checkerboard
# ---------------------------------------------------------------------------


class TestCreateCheckerboard:
    def test_output_shape(self):
        result = create_checkerboard(320, 240)
        assert result.shape == (240, 320, 3)

    def test_output_dtype(self):
        result = create_checkerboard(64, 64)
        assert result.dtype == np.float32

    def test_only_two_colors(self):
        result = create_checkerboard(64, 64, checker_size=32, color1=0.2, color2=0.8)
        unique = np.unique(result[..., 0].round(4))
        assert len(unique) == 2

    def test_color_values(self):
        result = create_checkerboard(64, 64, checker_size=32, color1=0.1, color2=0.9)
        flat = result[..., 0].flatten()
        assert np.any(np.isclose(flat, 0.1, atol=1e-4))
        assert np.any(np.isclose(flat, 0.9, atol=1e-4))

    def test_rgb_channels_equal(self):
        # Checkerboard is grayscale — all three channels must be identical
        result = create_checkerboard(64, 64)
        assert np.allclose(result[..., 0], result[..., 1])
        assert np.allclose(result[..., 1], result[..., 2])
