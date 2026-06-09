"""Unit tests for input color-space management.

Covers:
- CorridorKeyModule.core.colorspace: name validation, decode routing, and that
  camera-native encodings decode to scene-linear (sRGB primaries) correctly.
- backend.colorspace_detect: metadata → color-space mapping for video (ffprobe)
  and EXR chromaticities.
- InferenceSettings color_space wiring / legacy bool back-compat.

No GPU or model weights required.
"""

import colour
import numpy as np
import pytest

from CorridorKeyModule.core import colorspace as cs

# ---------------------------------------------------------------------------
# validate_color_space / requires_decode
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("name", cs.SUPPORTED_COLOR_SPACES)
    def test_supported_names_pass(self, name):
        assert cs.validate_color_space(name) == name

    def test_auto_requires_flag(self):
        with pytest.raises(ValueError):
            cs.validate_color_space("auto")
        assert cs.validate_color_space("auto", allow_auto=True) == "auto"

    def test_unknown_raises_with_choices(self):
        with pytest.raises(ValueError, match="Unknown color_space"):
            cs.validate_color_space("rec601")

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("srgb", False),
            ("linear", False),
            ("arri-logc3", True),
            ("arri-logc4", True),
            ("slog3", True),
            ("redlog3g10", True),
            ("acescct", True),
            ("aces2065-1", True),
            ("rec709", True),
            ("rec2020", True),
        ],
    )
    def test_requires_decode(self, name, expected):
        assert cs.requires_decode(name) is expected


# ---------------------------------------------------------------------------
# decode_to_linear
# ---------------------------------------------------------------------------


def _gray(value: float) -> np.ndarray:
    """A 1x1 neutral-gray RGB image."""
    return np.full((1, 1, 3), value, dtype=np.float32)


class TestDecodeToLinear:
    def test_linear_is_identity(self):
        img = _gray(0.42)
        out = cs.decode_to_linear(img, "linear")
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_srgb_applies_eotf(self):
        # sRGB 0.5 -> ~0.214 linear
        out = cs.decode_to_linear(_gray(0.5), "srgb")
        assert float(out[0, 0, 0]) == pytest.approx(0.214, abs=0.002)

    def test_arri_logc3_midgray_roundtrips_to_018(self):
        # Encode 18% scene-linear gray to LogC3 (EI 800), then decode back.
        # White points match (D65), so the neutral gray survives the gamut matrix.
        code = colour.models.log_encoding_ARRILogC3(np.array(0.18), EI=800)
        out = cs.decode_to_linear(_gray(float(code)), "arri-logc3", exposure_index=800)
        np.testing.assert_allclose(out, 0.18, atol=0.01)

    def test_arri_logc3_exposure_index_changes_result(self):
        code = _gray(0.40)
        at_800 = cs.decode_to_linear(code, "arri-logc3", exposure_index=800)
        at_1600 = cs.decode_to_linear(code, "arri-logc3", exposure_index=1600)
        assert not np.allclose(at_800, at_1600, atol=1e-4)

    def test_arri_logc3_unsupported_ei_raises_with_logc4_hint(self):
        # LogC3 has no EI 8000 (that's ALEXA 35 / LogC4 territory).
        with pytest.raises(ValueError, match="arri-logc4"):
            cs.decode_to_linear(_gray(0.40), "arri-logc3", exposure_index=8000)

    def test_arri_logc4_decodes_finite(self):
        out = cs.decode_to_linear(_gray(0.5), "arri-logc4")
        assert out.shape == (1, 1, 3)
        assert np.all(np.isfinite(out))

    def test_slog3_midgray_roundtrips(self):
        code = colour.models.log_encoding_SLog3(np.array(0.18))
        out = cs.decode_to_linear(_gray(float(code)), "slog3")
        np.testing.assert_allclose(out, 0.18, atol=0.02)

    def test_rec709_inverts_oetf(self):
        # Rec.709 shares sRGB primaries; encode linear gray via the 709 OETF.
        code = colour.models.oetf_BT709(np.array(0.25))
        out = cs.decode_to_linear(_gray(float(code)), "rec709")
        np.testing.assert_allclose(out, 0.25, atol=0.01)

    def test_aces2065_1_is_finite_float32(self):
        out = cs.decode_to_linear(_gray(0.18), "aces2065-1")
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_unknown_space_raises(self):
        with pytest.raises(ValueError):
            cs.decode_to_linear(_gray(0.5), "rec601")


# ---------------------------------------------------------------------------
# backend.colorspace_detect
# ---------------------------------------------------------------------------


class TestVideoDetection:
    @pytest.mark.parametrize(
        "tags,expected",
        [
            ({"color_transfer": "bt709", "color_primaries": "bt709"}, "rec709"),
            ({"color_transfer": "bt2020-10", "color_primaries": "bt2020"}, "rec2020"),
            ({"color_transfer": "iec61966-2-1", "color_primaries": "bt709"}, "srgb"),
            ({"color_transfer": "linear", "color_primaries": "bt709"}, "linear"),
            ({"color_transfer": "unknown", "color_primaries": "unknown"}, None),
            ({"color_transfer": None, "color_primaries": None}, None),
            # HDR PQ/HLG: bt2020 primaries but HDR transfer — must not map to rec2020
            ({"color_transfer": "smpte2084", "color_primaries": "bt2020"}, None),
            ({"color_transfer": "arib-std-b67", "color_primaries": "bt2020"}, None),
        ],
    )
    def test_mapping(self, monkeypatch, tags, expected):
        import backend.colorspace_detect as det
        import backend.ffmpeg_tools as ft

        monkeypatch.setattr(ft, "probe_video", lambda path: dict(tags))
        assert det.detect_color_space_from_video("clip.mov") == expected

    def test_probe_failure_returns_none(self, monkeypatch):
        import backend.colorspace_detect as det
        import backend.ffmpeg_tools as ft

        def _boom(path):
            raise RuntimeError("ffprobe not found")

        monkeypatch.setattr(ft, "probe_video", _boom)
        assert det.detect_color_space_from_video("clip.mov") is None


class TestChromaticitiesMatching:
    def test_srgb_primaries_map_to_linear(self):
        from backend.colorspace_detect import _match_chromaticities

        srgb = {"red": (0.640, 0.330), "green": (0.300, 0.600), "blue": (0.150, 0.060)}
        assert _match_chromaticities(srgb) == "linear"

    def test_ap0_primaries_map_to_aces(self):
        from backend.colorspace_detect import _match_chromaticities

        ap0 = {"red": (0.7347, 0.2653), "green": (0.0, 1.0), "blue": (0.0001, -0.077)}
        assert _match_chromaticities(ap0) == "aces2065-1"

    def test_unknown_primaries_return_none(self):
        from backend.colorspace_detect import _match_chromaticities

        weird = {"red": (0.5, 0.5), "green": (0.2, 0.2), "blue": (0.1, 0.1)}
        assert _match_chromaticities(weird) is None

    def test_chromaticities_from_sequence_header(self):
        from backend.colorspace_detect import _chromaticities_from_header

        # (rx, ry, gx, gy, bx, by, wx, wy) — sRGB primaries.
        header = {"chromaticities": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)}
        chroma = _chromaticities_from_header(header)
        assert chroma["red"] == pytest.approx((0.64, 0.33))
        assert chroma["blue"] == pytest.approx((0.15, 0.06))


# ---------------------------------------------------------------------------
# InferenceSettings wiring
# ---------------------------------------------------------------------------


class TestInferenceSettingsColorSpace:
    def test_default_is_srgb(self):
        from clip_manager import InferenceSettings

        s = InferenceSettings()
        assert s.color_space == "srgb"
        assert s.input_is_linear is False
        assert s.exposure_index == cs.DEFAULT_EXPOSURE_INDEX

    def test_legacy_linear_bool_maps_to_linear_space(self):
        from clip_manager import InferenceSettings

        s = InferenceSettings(input_is_linear=True)
        assert s.color_space == "linear"
        assert s.input_is_linear is True

    def test_decoded_space_keeps_input_is_linear_false(self):
        from clip_manager import InferenceSettings

        s = InferenceSettings(color_space="arri-logc3", exposure_index=8000)
        assert s.color_space == "arri-logc3"
        # Decode happens per-frame in the pipeline; the legacy bool stays False.
        assert s.input_is_linear is False
        assert s.exposure_index == 8000

    def test_invalid_space_raises(self):
        from clip_manager import InferenceSettings

        with pytest.raises(ValueError, match="Unknown color_space"):
            InferenceSettings(color_space="rec601")


# ---------------------------------------------------------------------------
# detect_color_space_from_exr
# ---------------------------------------------------------------------------


class TestDetectColorSpaceFromExr:
    """Tests for detect_color_space_from_exr — _read_exr_header is mocked."""

    def _patch(self, monkeypatch, header):
        import backend.colorspace_detect as det

        monkeypatch.setattr(det, "_read_exr_header", lambda _path: header)

    def test_colorspace_attr_linear(self, monkeypatch):
        self._patch(monkeypatch, {"colorSpace": "linear"})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") == "linear"

    def test_colorspace_attr_srgb(self, monkeypatch):
        self._patch(monkeypatch, {"colorSpace": "sRGB"})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") == "srgb"

    def test_colorspace_attr_aces2065(self, monkeypatch):
        self._patch(monkeypatch, {"colorSpace": b"ACES2065-1"})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") == "aces2065-1"

    def test_colorspace_attr_aces_ap0(self, monkeypatch):
        self._patch(monkeypatch, {"colorspace": "aces ap0"})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") == "aces2065-1"

    def test_colorspace_attr_lin_srgb(self, monkeypatch):
        self._patch(monkeypatch, {"colorSpace": "lin_sRGB"})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") == "linear"

    def test_fallback_to_chromaticities_srgb(self, monkeypatch):
        # No colorSpace attr; sRGB chromaticities → "linear"
        self._patch(monkeypatch, {"chromaticities": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.329)})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") == "linear"

    def test_fallback_to_chromaticities_aces(self, monkeypatch):
        self._patch(monkeypatch, {"chromaticities": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767)})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") == "aces2065-1"

    def test_unrecognised_header_returns_none(self, monkeypatch):
        self._patch(monkeypatch, {"software": "DaVinci Resolve"})
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") is None

    def test_missing_header_returns_none(self, monkeypatch):
        self._patch(monkeypatch, None)
        from backend.colorspace_detect import detect_color_space_from_exr

        assert detect_color_space_from_exr("x.exr") is None

    def test_read_exr_header_openexr_not_installed(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def block_openexr(name, *args, **kwargs):
            if name == "OpenEXR":
                raise ImportError("no module named OpenEXR")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_openexr)
        from backend.colorspace_detect import _read_exr_header

        assert _read_exr_header("x.exr") is None
