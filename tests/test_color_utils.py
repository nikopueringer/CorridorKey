"""Unit tests for CorridorKeyModule.core.color_utils.

These tests verify the color math that underpins CorridorKey's compositing
pipeline.  Every function is tested with both numpy arrays and PyTorch tensors
because color_utils supports both backends and bugs can hide in one path.

No GPU or model weights required — pure math.
"""

import numpy as np
import pytest
import torch

from CorridorKeyModule.core import color_utils as cu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_np(x):
    """Ensure value is a numpy float32 array."""
    return np.asarray(x, dtype=np.float32)


def _to_torch(x):
    """Ensure value is a float32 torch tensor."""
    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------------------
# linear_to_srgb  /  srgb_to_linear
# ---------------------------------------------------------------------------


class TestSrgbLinearConversion:
    """sRGB ↔ linear transfer function tests.

    The piecewise sRGB spec uses exponent 2.4 (not "gamma 2.2").
    Breakpoints: 0.0031308 (linear side), 0.04045 (sRGB side).
    """

    # Known identity values: 0 → 0, 1 → 1
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_identity_values_numpy(self, value):
        x = _to_np(value)
        assert cu.linear_to_srgb(x) == pytest.approx(value, abs=1e-7)
        assert cu.srgb_to_linear(x) == pytest.approx(value, abs=1e-7)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_identity_values_torch(self, value):
        x = _to_torch(value)
        assert cu.linear_to_srgb(x).item() == pytest.approx(value, abs=1e-7)
        assert cu.srgb_to_linear(x).item() == pytest.approx(value, abs=1e-7)

    # Mid-gray: sRGB 0.5 ≈ linear 0.214
    def test_mid_gray_numpy(self):
        srgb_half = _to_np(0.5)
        linear_val = cu.srgb_to_linear(srgb_half)
        assert linear_val == pytest.approx(0.214, abs=0.001)

    def test_mid_gray_torch(self):
        srgb_half = _to_torch(0.5)
        linear_val = cu.srgb_to_linear(srgb_half)
        assert linear_val.item() == pytest.approx(0.214, abs=0.001)

    # Roundtrip: linear → sRGB → linear ≈ original
    @pytest.mark.parametrize("value", [0.0, 0.001, 0.0031308, 0.05, 0.214, 0.5, 0.8, 1.0])
    def test_roundtrip_numpy(self, value):
        x = _to_np(value)
        roundtripped = cu.srgb_to_linear(cu.linear_to_srgb(x))
        assert roundtripped == pytest.approx(value, abs=1e-5)

    @pytest.mark.parametrize("value", [0.0, 0.001, 0.0031308, 0.05, 0.214, 0.5, 0.8, 1.0])
    def test_roundtrip_torch(self, value):
        x = _to_torch(value)
        roundtripped = cu.srgb_to_linear(cu.linear_to_srgb(x))
        assert roundtripped.item() == pytest.approx(value, abs=1e-5)

    # Piecewise continuity: both branches must agree at the breakpoint
    def test_breakpoint_continuity_linear_to_srgb(self):
        # At linear = 0.0031308, the two branches should produce the same sRGB value
        bp = 0.0031308
        below = cu.linear_to_srgb(_to_np(bp - 1e-7))
        above = cu.linear_to_srgb(_to_np(bp + 1e-7))
        at = cu.linear_to_srgb(_to_np(bp))
        # All three should be very close (no discontinuity)
        assert below == pytest.approx(float(at), abs=1e-4)
        assert above == pytest.approx(float(at), abs=1e-4)

    def test_breakpoint_continuity_srgb_to_linear(self):
        bp = 0.04045
        below = cu.srgb_to_linear(_to_np(bp - 1e-7))
        above = cu.srgb_to_linear(_to_np(bp + 1e-7))
        at = cu.srgb_to_linear(_to_np(bp))
        assert below == pytest.approx(float(at), abs=1e-4)
        assert above == pytest.approx(float(at), abs=1e-4)

    # Negative inputs should be clamped to 0
    def test_negative_clamped_linear_to_srgb_numpy(self):
        result = cu.linear_to_srgb(_to_np(-0.5))
        assert float(result) == pytest.approx(0.0, abs=1e-7)

    def test_negative_clamped_linear_to_srgb_torch(self):
        result = cu.linear_to_srgb(_to_torch(-0.5))
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_negative_clamped_srgb_to_linear_numpy(self):
        result = cu.srgb_to_linear(_to_np(-0.5))
        assert float(result) == pytest.approx(0.0, abs=1e-7)

    def test_negative_clamped_srgb_to_linear_torch(self):
        result = cu.srgb_to_linear(_to_torch(-0.5))
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    # Vectorized: works on arrays, not just scalars
    def test_vectorized_numpy(self):
        x = _to_np([0.0, 0.1, 0.5, 1.0])
        result = cu.linear_to_srgb(x)
        assert result.shape == (4,)
        roundtripped = cu.srgb_to_linear(result)
        np.testing.assert_allclose(roundtripped, x, atol=1e-5)

    def test_vectorized_torch(self):
        x = _to_torch([0.0, 0.1, 0.5, 1.0])
        result = cu.linear_to_srgb(x)
        assert result.shape == (4,)
        roundtripped = cu.srgb_to_linear(result)
        torch.testing.assert_close(roundtripped, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# premultiply  /  unpremultiply
# ---------------------------------------------------------------------------


class TestPremultiply:
    """Premultiply / unpremultiply tests.

    The core compositing contract: premultiplied RGB = straight RGB * alpha.
    """

    def test_roundtrip_numpy(self):
        fg = _to_np([[0.8, 0.5, 0.2]])
        alpha = _to_np([[0.6]])
        premul = cu.premultiply(fg, alpha)
        recovered = cu.unpremultiply(premul, alpha)
        np.testing.assert_allclose(recovered, fg, atol=1e-5)

    def test_roundtrip_torch(self):
        fg = _to_torch([[0.8, 0.5, 0.2]])
        alpha = _to_torch([[0.6]])
        premul = cu.premultiply(fg, alpha)
        recovered = cu.unpremultiply(premul, alpha)
        torch.testing.assert_close(recovered, fg, atol=1e-5, rtol=1e-5)

    def test_output_bounded_by_fg_numpy(self):
        """Premultiplied RGB must be <= straight RGB when 0 <= alpha <= 1."""
        fg = _to_np([[1.0, 0.5, 0.3]])
        alpha = _to_np([[0.7]])
        premul = cu.premultiply(fg, alpha)
        assert np.all(premul <= fg + 1e-7)

    def test_output_bounded_by_fg_torch(self):
        fg = _to_torch([[1.0, 0.5, 0.3]])
        alpha = _to_torch([[0.7]])
        premul = cu.premultiply(fg, alpha)
        assert torch.all(premul <= fg + 1e-7)

    def test_zero_alpha_numpy(self):
        """Premultiply by zero alpha → zero RGB."""
        fg = _to_np([[0.8, 0.5, 0.2]])
        alpha = _to_np([[0.0]])
        premul = cu.premultiply(fg, alpha)
        np.testing.assert_allclose(premul, 0.0, atol=1e-7)

    def test_one_alpha_numpy(self):
        """Premultiply by alpha=1 → unchanged."""
        fg = _to_np([[0.8, 0.5, 0.2]])
        alpha = _to_np([[1.0]])
        premul = cu.premultiply(fg, alpha)
        np.testing.assert_allclose(premul, fg, atol=1e-7)


# ---------------------------------------------------------------------------
# composite_straight  /  composite_premul
# ---------------------------------------------------------------------------


class TestCompositing:
    """The Porter-Duff 'over' operator: A over B.

    composite_straight and composite_premul must produce the same result
    given equivalent inputs.
    """

    def test_straight_vs_premul_equivalence_numpy(self):
        fg = _to_np([0.9, 0.3, 0.1])
        bg = _to_np([0.1, 0.2, 0.8])
        alpha = _to_np(0.6)

        result_straight = cu.composite_straight(fg, bg, alpha)
        fg_premul = cu.premultiply(fg, alpha)
        result_premul = cu.composite_premul(fg_premul, bg, alpha)

        np.testing.assert_allclose(result_straight, result_premul, atol=1e-6)

    def test_straight_vs_premul_equivalence_torch(self):
        fg = _to_torch([0.9, 0.3, 0.1])
        bg = _to_torch([0.1, 0.2, 0.8])
        alpha = _to_torch(0.6)

        result_straight = cu.composite_straight(fg, bg, alpha)
        fg_premul = cu.premultiply(fg, alpha)
        result_premul = cu.composite_premul(fg_premul, bg, alpha)

        torch.testing.assert_close(result_straight, result_premul, atol=1e-6, rtol=1e-6)

    def test_alpha_zero_shows_background(self):
        fg = _to_np([1.0, 0.0, 0.0])
        bg = _to_np([0.0, 0.0, 1.0])
        alpha = _to_np(0.0)
        result = cu.composite_straight(fg, bg, alpha)
        np.testing.assert_allclose(result, bg, atol=1e-7)

    def test_alpha_one_shows_foreground(self):
        fg = _to_np([1.0, 0.0, 0.0])
        bg = _to_np([0.0, 0.0, 1.0])
        alpha = _to_np(1.0)
        result = cu.composite_straight(fg, bg, alpha)
        np.testing.assert_allclose(result, fg, atol=1e-7)


# ---------------------------------------------------------------------------
# despill
# ---------------------------------------------------------------------------


def _make_pixel(screen_channel: int, screen_value: float, off_a: float, off_b: float):
    """Build a single-pixel RGB array with arbitrary screen channel."""
    px = [0.0, 0.0, 0.0]
    others = [i for i in (0, 1, 2) if i != screen_channel]
    px[screen_channel] = screen_value
    px[others[0]] = off_a
    px[others[1]] = off_b
    return _to_np([px])


class TestDespill:
    """Screen-color spill removal (channel-agnostic).

    The despill function clamps excess in the screen channel based on the two
    other channels, then redistributes the removed energy to preserve luminance.
    Default screen_channel=1 (green) preserves historical behavior; pass 2 for
    blue-screen plates.
    """

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("screen_channel", [1, 2])
    def test_pure_screen_reduced_average_mode(self, backend, screen_channel):
        """Pure screen-color pixel should have its screen channel clamped to 0."""
        img = _make_pixel(screen_channel, 1.0, 0.0, 0.0)
        if backend == "openCV":
            result = cu.despill_opencv(img, limit_mode="average", strength=1.0, screen_channel=screen_channel)
        else:
            img_t = torch.from_numpy(img)
            result = cu.despill_torch(img_t, strength=1.0, screen_channel=screen_channel).numpy()
        assert result[0, screen_channel] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("screen_channel", [1, 2])
    def test_pure_screen_reduced_max_mode(self, screen_channel):
        """With 'max' mode, screen clamped to max(other_a, other_b) = 0 for pure screen pixel."""
        img = _make_pixel(screen_channel, 1.0, 0.0, 0.0)
        result = cu.despill_opencv(img, limit_mode="max", strength=1.0, screen_channel=screen_channel)
        assert result[0, screen_channel] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("screen_channel", [1, 2])
    def test_pure_off_screen_unchanged(self, backend, screen_channel):
        """A pixel with no screen excess should not be modified."""
        # Strong red, zero in the other channels.
        img = _make_pixel(screen_channel, 0.0, 1.0, 0.0)
        if backend == "openCV":
            result = cu.despill_opencv(img, limit_mode="average", strength=1.0, screen_channel=screen_channel)
        else:
            img_t = torch.from_numpy(img)
            result = cu.despill_torch(img_t, strength=1.0, screen_channel=screen_channel).numpy()
        np.testing.assert_allclose(result, img, atol=1e-6)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("screen_channel", [1, 2])
    def test_strength_zero_is_noop(self, backend, screen_channel):
        """strength=0 should return the input unchanged."""
        img = _make_pixel(screen_channel, 0.9, 0.2, 0.1)
        if backend == "openCV":
            result = cu.despill_opencv(img, strength=0.0, screen_channel=screen_channel)
        else:
            img_t = torch.from_numpy(img)
            result = cu.despill_torch(img_t, strength=0.0, screen_channel=screen_channel).numpy()
        np.testing.assert_allclose(result, img, atol=1e-7)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("screen_channel", [1, 2])
    def test_partial_spill_average_mode(self, backend, screen_channel):
        """Screen slightly above (other_a + other_b)/2 should be reduced, not zeroed."""
        img = _make_pixel(screen_channel, 0.8, 0.4, 0.2)
        if backend == "openCV":
            result = cu.despill_opencv(img, limit_mode="average", strength=1.0, screen_channel=screen_channel)
        else:
            img_t = torch.from_numpy(img)
            result = cu.despill_torch(img_t, strength=1.0, screen_channel=screen_channel).numpy()
        limit = (0.4 + 0.2) / 2.0
        assert result[0, screen_channel] == pytest.approx(limit, abs=1e-5)

    def test_max_mode_higher_limit_than_average(self):
        """'max' mode uses max(other_a, other_b) >= average, so less despill."""
        img = _to_np([[0.6, 0.8, 0.1]])  # green screen
        result_avg = cu.despill_opencv(img, limit_mode="average", strength=1.0)
        result_max = cu.despill_opencv(img, limit_mode="max", strength=1.0)
        assert result_max[0, 1] >= result_avg[0, 1]

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    def test_fractional_strength_interpolates(self, backend):
        """strength=0.5 should produce a result between original and fully despilled."""
        img = _to_np([[0.2, 0.9, 0.1]])
        if backend == "openCV":
            full = cu.despill_opencv(img, limit_mode="average", strength=1.0)
            half = cu.despill_opencv(img, limit_mode="average", strength=0.5)
        else:
            img_t = torch.from_numpy(img)
            full = cu.despill_torch(img_t, strength=1.0).numpy()
            half = cu.despill_torch(img_t, strength=0.5).numpy()
        assert half[0, 1] < img[0, 1]
        assert half[0, 1] > full[0, 1]
        expected = img * 0.5 + full * 0.5
        np.testing.assert_allclose(half, expected, atol=1e-6)

    def test_despill_torch_matches_numpy(self):
        """Verify torch path matches numpy path."""
        img_np = _to_np([[0.3, 0.9, 0.2]])
        img_t = _to_torch([[0.3, 0.9, 0.2]])
        result_np = cu.despill_opencv(img_np, limit_mode="average", strength=1.0)
        result_t = cu.despill_opencv(img_t, limit_mode="average", strength=1.0)
        np.testing.assert_allclose(result_np, result_t.numpy(), atol=1e-5)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    @pytest.mark.parametrize("screen_channel", [1, 2])
    def test_screen_below_limit_unchanged(self, backend, screen_channel):
        """spill_amount is clamped to zero when screen < (other_a + other_b)/2 — pixel returned unchanged.

        When a pixel has less in the screen channel than the luminance limit it
        carries no spill.  The max(..., 0) clamp on spill_amount ensures the
        pixel is left untouched.  Without that clamp despill would corrupt
        non-spill regions.
        """
        # screen=0.3 is well below the average limit (0.8+0.6)/2 = 0.7
        img = _make_pixel(screen_channel, 0.3, 0.8, 0.6)
        if backend == "openCV":
            result = cu.despill_opencv(img, limit_mode="average", strength=1.0, screen_channel=screen_channel)
        else:
            img_t = torch.from_numpy(img)
            result = cu.despill_torch(img_t, strength=1.0, screen_channel=screen_channel).numpy()
        np.testing.assert_allclose(result, img, atol=1e-6)

    def test_default_remains_green(self):
        """Calling despill_opencv without screen_channel must still target green (regression guard)."""
        img = _to_np([[0.0, 1.0, 0.0]])
        result = cu.despill_opencv(img, strength=1.0)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_legacy_green_limit_mode_kwarg(self):
        """Old callers (Nuke, Houdini) pass green_limit_mode= — must still work AND emit DeprecationWarning."""
        img = _to_np([[0.4, 0.8, 0.2]])
        with pytest.warns(DeprecationWarning, match="green_limit_mode"):
            result_legacy = cu.despill_opencv(img, green_limit_mode="average", strength=1.0)
        result_new = cu.despill_opencv(img, limit_mode="average", strength=1.0)
        np.testing.assert_allclose(result_legacy, result_new, atol=1e-7)

    def test_invalid_channel_raises(self):
        img = _to_np([[0.5, 0.5, 0.5]])
        with pytest.raises(ValueError):
            cu.despill_opencv(img, strength=1.0, screen_channel=3)
        with pytest.raises(ValueError):
            cu.despill_torch(torch.from_numpy(img), strength=1.0, screen_channel=-1)


class TestEstimateScreenColor:
    """Auto-detect screen color from background pixels."""

    @staticmethod
    def _make_scene(screen_rgb: tuple[float, float, float], subject_size: int = 20):
        """Build a 100x100 image with given screen color and a centered white subject."""
        h = w = 100
        img = np.full((h, w, 3), screen_rgb, dtype=np.float32)
        cy, cx = h // 2, w // 2
        s = subject_size // 2
        img[cy - s : cy + s, cx - s : cx + s] = 1.0  # white subject
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[cy - s : cy + s, cx - s : cx + s] = 1.0
        return img, alpha

    def test_detects_green(self):
        img, alpha = self._make_scene((0.05, 0.85, 0.10))
        assert cu.estimate_screen_color(img, alpha) == "green"

    def test_detects_blue(self):
        img, alpha = self._make_scene((0.05, 0.10, 0.85))
        assert cu.estimate_screen_color(img, alpha) == "blue"

    def test_ambiguous_defaults_to_green(self):
        img, alpha = self._make_scene((0.10, 0.50, 0.51))  # G ≈ B
        assert cu.estimate_screen_color(img, alpha) == "green"

    def test_no_background_defaults_to_green(self):
        img = np.full((100, 100, 3), (0.05, 0.10, 0.85), dtype=np.float32)
        alpha = np.ones((100, 100), dtype=np.float32)  # everything is foreground
        assert cu.estimate_screen_color(img, alpha) == "green"

    def test_alpha_with_channel_dim(self):
        img, alpha = self._make_scene((0.05, 0.10, 0.85))
        alpha_3d = alpha[..., np.newaxis]
        assert cu.estimate_screen_color(img, alpha_3d) == "blue"

    def test_rejects_image_wrong_ndim(self):
        """A 2D 'image' (missing channel dim) must fail fast, not crash later in indexing."""
        with pytest.raises(ValueError, match="HxWx3"):
            cu.estimate_screen_color(np.zeros((100, 100), dtype=np.float32), np.zeros((100, 100)))

    def test_rejects_image_too_few_channels(self):
        """A grayscale-stacked image with 2 channels must fail fast."""
        with pytest.raises(ValueError, match="HxWx3"):
            cu.estimate_screen_color(np.zeros((100, 100, 2), dtype=np.float32), np.zeros((100, 100)))

    def test_rejects_alpha_wrong_ndim(self):
        """A 4D alpha (e.g. accidentally batched) must fail fast."""
        with pytest.raises(ValueError, match="HxW or HxWx1"):
            cu.estimate_screen_color(
                np.zeros((100, 100, 3), dtype=np.float32),
                np.zeros((1, 100, 100, 1), dtype=np.float32),
            )

    def test_rejects_shape_mismatch(self):
        """Image and alpha must agree on H,W."""
        with pytest.raises(ValueError, match="must agree on H,W"):
            cu.estimate_screen_color(np.zeros((100, 100, 3), dtype=np.float32), np.zeros((50, 50), dtype=np.float32))


class TestScreenChannelForColor:
    """Single-source-of-truth helper: 'green' → 1, 'blue' → 2."""

    def test_known_colors(self):
        assert cu.screen_channel_for_color("green") == 1
        assert cu.screen_channel_for_color("blue") == 2

    def test_auto_is_rejected(self):
        """'auto' is the unresolved sentinel — callers must resolve before mapping."""
        with pytest.raises(ValueError, match="auto"):
            cu.screen_channel_for_color("auto")

    def test_unknown_is_rejected(self):
        with pytest.raises(ValueError, match="red"):
            cu.screen_channel_for_color("red")

    def test_constants_are_aligned(self):
        """SCREEN_COLOR_CHOICES must be exactly the keys of SCREEN_CHANNEL_BY_COLOR,
        and SCREEN_COLOR_CHOICES_WITH_AUTO must add only the 'auto' sentinel."""
        assert set(cu.SCREEN_COLOR_CHOICES) == set(cu.SCREEN_CHANNEL_BY_COLOR.keys())
        assert set(cu.SCREEN_COLOR_CHOICES_WITH_AUTO) - set(cu.SCREEN_COLOR_CHOICES) == {cu.SCREEN_COLOR_AUTO}


# ---------------------------------------------------------------------------
# clean_matte
# ---------------------------------------------------------------------------


class TestCleanMatte:
    """Connected-component cleanup of alpha mattes.

    Small disconnected blobs (tracking markers, noise) should be removed
    while large foreground regions are preserved.
    """

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    def test_large_blob_preserved(self, backend):
        """A single large opaque region should survive cleanup."""
        matte = np.zeros((100, 100), dtype=np.float32)
        matte[20:80, 20:80] = 1.0  # 60x60 = 3600 pixels
        if backend == "openCV":
            result = cu.clean_matte_opencv(matte, area_threshold=300)
        else:
            matte = torch.from_numpy(matte).unsqueeze(0).unsqueeze(0)
            result = cu.clean_matte_torch(matte, area_threshold=300).squeeze(0).squeeze(0).numpy()
        # Center of the blob should still be opaque
        assert result[50, 50] > 0.9

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    def test_small_blob_removed(self, backend):
        """A tiny blob below the threshold should be removed."""
        matte = np.zeros((100, 100), dtype=np.float32)
        matte[5:8, 5:8] = 1.0  # 3x3 = 9 pixels
        if backend == "openCV":
            result = cu.clean_matte_opencv(matte, area_threshold=300)  #
        else:
            matte = torch.from_numpy(matte).unsqueeze(0).unsqueeze(0)
            result = cu.clean_matte_torch(matte, area_threshold=300).squeeze(0).squeeze(0).numpy()
        assert result[6, 6] == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("backend", ["openCV", "torch"])
    def test_mixed_blobs(self, backend):
        """Large blob kept, small blob removed."""
        matte = np.zeros((200, 200), dtype=np.float32)
        # Large blob: 50x50 = 2500 px
        matte[10:60, 10:60] = 1.0
        # Small blob: 5x5 = 25 px
        matte[150:155, 150:155] = 1.0

        if backend == "openCV":
            result = cu.clean_matte_opencv(matte, area_threshold=100)
        else:
            matte = torch.from_numpy(matte).unsqueeze(0).unsqueeze(0)
            result = cu.clean_matte_torch(matte, area_threshold=100).squeeze(0).squeeze(0).numpy()
        assert result[35, 35] > 0.9  # large blob center preserved
        assert result[152, 152] < 0.01  # small blob removed

    def test_3d_input_preserved(self):
        """[H, W, 1] input should return [H, W, 1] output."""
        matte = np.zeros((50, 50, 1), dtype=np.float32)
        matte[10:40, 10:40, 0] = 1.0
        result = cu.clean_matte_opencv(matte, area_threshold=100)
        assert result.ndim == 3
        assert result.shape[2] == 1


# ---------------------------------------------------------------------------
# create_checkerboard
# ---------------------------------------------------------------------------


class TestCheckerboard:
    """Checkerboard pattern generator used for QC composites."""

    def test_output_shape(self):
        result = cu.create_checkerboard(640, 480)
        assert result.shape == (480, 640, 3)

    def test_output_range(self):
        result = cu.create_checkerboard(100, 100, color1=0.2, color2=0.4)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uses_specified_colors(self):
        result = cu.create_checkerboard(128, 128, checker_size=64, color1=0.1, color2=0.9)
        unique_vals = np.unique(result[:, :, 0])
        np.testing.assert_allclose(sorted(unique_vals), [0.1, 0.9], atol=1e-6)


# ---------------------------------------------------------------------------
# rgb_to_yuv
# ---------------------------------------------------------------------------


class TestRgbToYuv:
    """RGB to YUV (Rec. 601) conversion.

    Three layout branches: BCHW (4D), CHW (3D channel-first), and
    last-dim (3D/2D channel-last). Each independently indexes channels,
    so a wrong index silently swaps color information.

    Known Rec. 601 coefficients: Y = 0.299R + 0.587G + 0.114B
    """

    def test_pure_white_bchw(self):
        """Pure white (1,1,1) → Y=1, U=0, V=0 in any colorspace."""
        img = torch.ones(1, 3, 2, 2)  # BCHW
        result = cu.rgb_to_yuv(img)
        assert result.shape == (1, 3, 2, 2)
        # Y channel should be 1.0
        torch.testing.assert_close(result[:, 0], torch.ones(1, 2, 2), atol=1e-5, rtol=1e-5)
        # U and V should be ~0 for achromatic input
        assert result[:, 1].abs().max() < 1e-5
        assert result[:, 2].abs().max() < 1e-5

    def test_pure_red_known_values(self):
        """Pure red (1,0,0) → known Y, U, V from Rec. 601 coefficients."""
        img = torch.zeros(1, 3, 1, 1)
        img[0, 0, 0, 0] = 1.0  # R=1, G=0, B=0
        result = cu.rgb_to_yuv(img)
        expected_y = 0.299
        expected_u = 0.492 * (0.0 - expected_y)  # 0.492 * (B - Y)
        expected_v = 0.877 * (1.0 - expected_y)  # 0.877 * (R - Y)
        assert result[0, 0, 0, 0].item() == pytest.approx(expected_y, abs=1e-5)
        assert result[0, 1, 0, 0].item() == pytest.approx(expected_u, abs=1e-5)
        assert result[0, 2, 0, 0].item() == pytest.approx(expected_v, abs=1e-5)

    def test_chw_layout(self):
        """3D CHW input (channel-first) should produce CHW output."""
        img = torch.zeros(3, 4, 4)
        img[1, :, :] = 1.0  # Pure green
        result = cu.rgb_to_yuv(img)
        assert result.shape == (3, 4, 4)
        expected_y = 0.587  # 0.299*0 + 0.587*1 + 0.114*0
        assert result[0, 0, 0].item() == pytest.approx(expected_y, abs=1e-5)

    def test_last_dim_layout(self):
        """2D [N, 3] input (channel-last) should produce [N, 3] output."""
        img = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = cu.rgb_to_yuv(img)
        assert result.shape == (3, 3)
        # Row 0 is pure red: Y = 0.299
        assert result[0, 0].item() == pytest.approx(0.299, abs=1e-5)
        # Row 1 is pure green: Y = 0.587
        assert result[1, 0].item() == pytest.approx(0.587, abs=1e-5)
        # Row 2 is pure blue: Y = 0.114
        assert result[2, 0].item() == pytest.approx(0.114, abs=1e-5)

    def test_rejects_numpy(self):
        """rgb_to_yuv is torch-only — numpy input should raise TypeError."""
        img = np.zeros((3, 4, 4), dtype=np.float32)
        with pytest.raises(TypeError):
            cu.rgb_to_yuv(img)


# ---------------------------------------------------------------------------
# dilate_mask
# ---------------------------------------------------------------------------


class TestDilateMask:
    """Mask dilation via cv2 (numpy) or max_pool2d (torch).

    Both backends should expand the mask outward. radius=0 is a no-op.
    """

    def test_radius_zero_noop_numpy(self):
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[20:30, 20:30] = 1.0
        result = cu.dilate_mask(mask, radius=0)
        np.testing.assert_array_equal(result, mask)

    def test_radius_zero_noop_torch(self):
        mask = torch.zeros(50, 50)
        mask[20:30, 20:30] = 1.0
        result = cu.dilate_mask(mask, radius=0)
        torch.testing.assert_close(result, mask)

    def test_dilation_expands_numpy(self):
        """Dilated mask should be >= original at every pixel."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[20:30, 20:30] = 1.0
        result = cu.dilate_mask(mask, radius=3)
        assert np.all(result >= mask)
        # Pixels just outside the original region should now be 1
        assert result[19, 25] > 0  # above the original box
        assert result[25, 19] > 0  # left of the original box

    def test_dilation_expands_torch(self):
        """Dilated mask should be >= original at every pixel (torch path)."""
        mask = torch.zeros(50, 50)
        mask[20:30, 20:30] = 1.0
        result = cu.dilate_mask(mask, radius=3)
        assert torch.all(result >= mask)
        assert result[19, 25] > 0
        assert result[25, 19] > 0

    def test_preserves_2d_shape_numpy(self):
        mask = np.zeros((40, 60), dtype=np.float32)
        result = cu.dilate_mask(mask, radius=5)
        assert result.shape == (40, 60)

    def test_preserves_2d_shape_torch(self):
        mask = torch.zeros(40, 60)
        result = cu.dilate_mask(mask, radius=5)
        assert result.shape == (40, 60)

    def test_preserves_3d_shape_torch(self):
        """[C, H, W] input should return [C, H, W] output."""
        mask = torch.zeros(1, 40, 60)
        result = cu.dilate_mask(mask, radius=5)
        assert result.shape == (1, 40, 60)


# ---------------------------------------------------------------------------
# apply_garbage_matte
# ---------------------------------------------------------------------------


class TestApplyGarbageMatte:
    """Garbage matte application: multiplies predicted matte by a dilated coarse mask.

    Used to zero out regions outside the coarse matte (rigs, lights, etc.).
    """

    def test_none_input_passthrough(self):
        """If no garbage matte is provided, the predicted matte is returned unchanged."""
        rng = np.random.default_rng(42)
        matte = rng.random((100, 100)).astype(np.float32)
        result = cu.apply_garbage_matte(matte, None)
        np.testing.assert_array_equal(result, matte)

    def test_zeros_outside_garbage_region(self):
        """Regions outside the garbage matte should be zeroed."""
        predicted = np.ones((50, 50), dtype=np.float32)
        garbage = np.zeros((50, 50), dtype=np.float32)
        garbage[10:40, 10:40] = 1.0  # only center is valid
        result = cu.apply_garbage_matte(predicted, garbage, dilation=0)
        # Outside the garbage matte region should be 0
        assert result[0, 0] == pytest.approx(0.0, abs=1e-7)
        # Inside should be preserved
        assert result[25, 25] == pytest.approx(1.0, abs=1e-7)

    def test_3d_matte_with_2d_garbage(self):
        """[H, W, 1] predicted matte with [H, W] garbage matte should broadcast."""
        predicted = np.ones((50, 50, 1), dtype=np.float32)
        garbage = np.zeros((50, 50), dtype=np.float32)
        garbage[10:40, 10:40] = 1.0
        result = cu.apply_garbage_matte(predicted, garbage, dilation=0)
        assert result.shape == (50, 50, 1)
        assert result[0, 0, 0] == pytest.approx(0.0, abs=1e-7)
        assert result[25, 25, 0] == pytest.approx(1.0, abs=1e-7)
