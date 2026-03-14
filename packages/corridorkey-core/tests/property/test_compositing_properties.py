"""Property-based tests for corridorkey_core.compositing.

Uses Hypothesis to verify mathematical invariants across many random inputs.
These tests run in the fast suite - no GPU or checkpoint required.
"""

import numpy as np
from corridorkey_core.compositing import (
    composite_premul,
    composite_straight,
    create_checkerboard,
    despill,
    linear_to_srgb,
    premultiply,
    srgb_to_linear,
)
from hypothesis import given, settings
from hypothesis import strategies as st

# Hypothesis profile: keep examples low enough to stay fast in CI
settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")

# Valid normalized float values
_unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Small spatial dimensions to keep array allocation fast
_small_dim = st.integers(min_value=2, max_value=32)


def _solid_array(h: int, w: int, r: float, g: float, b: float) -> np.ndarray:
    img = np.full((h, w, 3), 0.0, dtype=np.float32)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


@given(_unit_float)
def test_srgb_linear_roundtrip(value: float):
    """sRGB -> linear -> sRGB must recover the original value."""
    x = np.array([value], dtype=np.float32)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-5)


@given(_unit_float)
def test_linear_srgb_roundtrip(value: float):
    """linear -> sRGB -> linear must recover the original value."""
    x = np.array([value], dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


@given(_unit_float)
def test_linear_to_srgb_output_in_range(value: float):
    """linear_to_srgb output must stay in [0, 1] for any input in [0, 1]."""
    x = np.array([value], dtype=np.float32)
    result = linear_to_srgb(x)
    assert 0.0 <= float(result[0]) <= 1.0 + 1e-6


@given(_unit_float)
def test_srgb_to_linear_output_in_range(value: float):
    """srgb_to_linear output must stay in [0, 1] for any input in [0, 1]."""
    x = np.array([value], dtype=np.float32)
    result = srgb_to_linear(x)
    assert 0.0 <= float(result[0]) <= 1.0 + 1e-6


@given(st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
def test_linear_to_srgb_is_monotone(value: float):
    """linear_to_srgb must be monotonically non-decreasing."""
    x = np.array([value, value + 0.005], dtype=np.float32)
    result = linear_to_srgb(x)
    assert result[1] >= result[0] - 1e-6


@given(_unit_float, _unit_float, _small_dim, _small_dim)
def test_composite_straight_full_alpha_returns_fg(fg_val: float, bg_val: float, h: int, w: int):
    """composite_straight with alpha=1 must return the foreground exactly."""
    fg = _solid_array(h, w, fg_val, fg_val, fg_val)
    bg = _solid_array(h, w, bg_val, bg_val, bg_val)
    alpha = np.ones((h, w, 1), dtype=np.float32)
    result = composite_straight(fg, bg, alpha)
    assert np.allclose(result, fg, atol=1e-6)


@given(_unit_float, _unit_float, _small_dim, _small_dim)
def test_composite_straight_zero_alpha_returns_bg(fg_val: float, bg_val: float, h: int, w: int):
    """composite_straight with alpha=0 must return the background exactly."""
    fg = _solid_array(h, w, fg_val, fg_val, fg_val)
    bg = _solid_array(h, w, bg_val, bg_val, bg_val)
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    result = composite_straight(fg, bg, alpha)
    assert np.allclose(result, bg, atol=1e-6)


@given(_unit_float, _unit_float, _small_dim, _small_dim)
def test_composite_premul_zero_alpha_returns_bg(fg_val: float, bg_val: float, h: int, w: int):
    """composite_premul with alpha=0 and premul fg=0 must return the background."""
    fg = np.zeros((h, w, 3), dtype=np.float32)  # premultiplied by alpha=0
    bg = _solid_array(h, w, bg_val, bg_val, bg_val)
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    result = composite_premul(fg, bg, alpha)
    assert np.allclose(result, bg, atol=1e-6)


@given(_unit_float, _unit_float, _small_dim, _small_dim)
def test_composite_output_in_range(fg_val: float, bg_val: float, h: int, w: int):
    """composite_straight output must stay in [0, 1] when inputs are in [0, 1]."""
    fg = _solid_array(h, w, fg_val, fg_val, fg_val)
    bg = _solid_array(h, w, bg_val, bg_val, bg_val)
    alpha = np.full((h, w, 1), 0.5, dtype=np.float32)
    result = composite_straight(fg, bg, alpha)
    assert result.min() >= -1e-6
    assert result.max() <= 1.0 + 1e-6


@given(_unit_float, _small_dim, _small_dim)
def test_premultiply_full_alpha_identity(val: float, h: int, w: int):
    """premultiply with alpha=1 must return the foreground unchanged."""
    fg = _solid_array(h, w, val, val, val)
    alpha = np.ones((h, w, 1), dtype=np.float32)
    assert np.allclose(premultiply(fg, alpha), fg, atol=1e-6)


@given(_unit_float, _small_dim, _small_dim)
def test_premultiply_zero_alpha_gives_black(val: float, h: int, w: int):
    """premultiply with alpha=0 must return zeros regardless of fg."""
    fg = _solid_array(h, w, val, val, val)
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    assert np.allclose(premultiply(fg, alpha), 0.0, atol=1e-6)


@given(_unit_float, _unit_float, _small_dim, _small_dim)
def test_despill_never_increases_green(r: float, b: float, h: int, w: int):
    """despill must never increase the green channel."""
    img = _solid_array(h, w, r, min(r, b) + 0.3, b)  # green is always higher
    img = np.clip(img, 0.0, 1.0)
    result = despill(img, strength=1.0)
    assert np.all(result[..., 1] <= img[..., 1] + 1e-6)


@given(_unit_float, _unit_float, _unit_float, _small_dim, _small_dim)
def test_despill_zero_strength_identity(r: float, g: float, b: float, h: int, w: int):
    """despill with strength=0 must return the input unchanged."""
    img = _solid_array(h, w, r, g, b)
    result = despill(img, strength=0.0)
    assert np.allclose(result, img, atol=1e-6)


@given(_unit_float, _unit_float, _unit_float, _small_dim, _small_dim)
def test_despill_output_in_range(r: float, g: float, b: float, h: int, w: int):
    """despill output must stay in [0, 1] for any input in [0, 1]."""
    img = _solid_array(h, w, r, g, b)
    result = despill(img, strength=1.0)
    assert np.all(result >= -1e-6)
    assert np.all(result <= 1.0 + 1e-6)


@given(_small_dim, _small_dim)
def test_checkerboard_shape(w: int, h: int):
    """create_checkerboard must return [H, W, 3] for any positive dimensions."""
    result = create_checkerboard(w, h)
    assert result.shape == (h, w, 3)


@given(_unit_float, _unit_float)
def test_checkerboard_only_two_values(c1: float, c2: float):
    """create_checkerboard must contain exactly two distinct color values."""
    result = create_checkerboard(64, 64, checker_size=16, color1=c1, color2=c2)
    unique = np.unique(result[..., 0].round(5))
    # If c1 == c2 there is only one value, otherwise exactly two
    expected = 1 if np.isclose(c1, c2, atol=1e-5) else 2
    assert len(unique) == expected
