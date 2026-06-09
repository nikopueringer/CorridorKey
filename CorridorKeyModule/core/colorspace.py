"""Input color-space management — decode camera-native encodings to scene-linear.

The CorridorKey model expects sRGB display-referred input. Camera-native plates
(ARRI LogC, Sony S-Log3, RED Log3G10, ACES, Rec.2020, …) are log-encoded and/or
use wide gamuts, so feeding them straight to the model — or running ``linear_to_srgb``
on them — produces wrong tonality and a degraded key.

This module decodes any supported color space to **scene-linear in sRGB / Rec.709
primaries**. The pipeline then treats the result as linear (resize in linear →
``linear_to_srgb`` → ImageNet normalize), reusing the existing ``input_is_linear``
path with no engine changes.

``srgb`` and ``linear`` keep their existing fast paths (``requires_decode`` is False)
so their output is byte-for-byte unchanged; only the new spaces add a decode step.

The heavy ``colour`` (colour-science) import is deferred to call time so it does not
slow down CLI startup for the common ``srgb`` case.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# --- Color space: single source of truth -----------------------------------
# Mirrors the SCREEN_COLOR_* pattern in color_utils.py. Adding a camera encoding
# requires only registering it in _DECODE_SPECS below and listing it here.

COLOR_SPACE_AUTO: str = "auto"
DEFAULT_COLOR_SPACE: str = "srgb"

# Spaces handled by existing fast paths — no colour-science decode needed.
#   srgb   -> fed to the model as-is (model is trained on sRGB)
#   linear -> existing input_is_linear path applies linear_to_srgb
_PASSTHROUGH_COLOR_SPACES: tuple[str, ...] = ("srgb", "linear")

# Spaces that decode to scene-linear (sRGB primaries) via colour-science.
_DECODED_COLOR_SPACES: tuple[str, ...] = (
    "rec709",
    "rec2020",
    "arri-logc3",
    "arri-logc4",
    "slog3",
    "redlog3g10",
    "acescct",
    "aces2065-1",
)

SUPPORTED_COLOR_SPACES: tuple[str, ...] = _PASSTHROUGH_COLOR_SPACES + _DECODED_COLOR_SPACES
COLOR_SPACE_CHOICES_WITH_AUTO: tuple[str, ...] = (COLOR_SPACE_AUTO,) + SUPPORTED_COLOR_SPACES

# Default ARRI LogC3 Exposure Index (ISO). LogC3 is EI-dependent; LogC4 is not.
DEFAULT_EXPOSURE_INDEX: int = 800

# ARRI LogC3 is only defined for a discrete set of Exposure Indices (these match
# colour-science's conversion table). Higher-EI ALEXA footage (e.g. EI 8000 on an
# ALEXA 35) is LogC4, which is EI-independent — use color_space="arri-logc4".
ARRI_LOGC3_EXPOSURE_INDICES: tuple[int, ...] = (160, 200, 250, 320, 400, 500, 640, 800, 1000, 1280, 1600)


def validate_color_space(name: str, *, allow_auto: bool = False) -> str:
    """Validate a color-space name, returning it unchanged.

    Raises ValueError (listing valid choices) on an unknown value. ``allow_auto``
    permits the sentinel "auto" used before detection resolves a concrete space.
    """
    valid = COLOR_SPACE_CHOICES_WITH_AUTO if allow_auto else SUPPORTED_COLOR_SPACES
    if name not in valid:
        raise ValueError(f"Unknown color_space '{name}'. Valid: {', '.join(valid)}")
    return name


def requires_decode(color_space: str) -> bool:
    """Whether a color space needs a colour-science decode before inference.

    False for ``srgb``/``linear`` (existing fast paths), True for camera-native
    encodings. ``auto`` is conservatively treated as no-decode — callers must
    resolve it to a concrete space first.
    """
    return color_space in _DECODED_COLOR_SPACES


# --- Decode registry --------------------------------------------------------
# Each entry maps a color-space name to:
#   transfer:  name of the colour-science decode (resolved lazily), or None when
#              the data is already scene-linear (only a gamut change is needed).
#   gamut:     source RGB colourspace name in colour.RGB_COLOURSPACES whose
#              primaries the linear data uses; converted to sRGB primaries.
#
# transfer names map to functions inside _decode_transfer() so we can pass the
# EI parameter to ARRI LogC3.
_DECODE_SPECS: dict[str, dict[str, str | None]] = {
    "rec709": {"transfer": "BT.709", "gamut": "ITU-R BT.709"},
    "rec2020": {"transfer": "BT.2020", "gamut": "ITU-R BT.2020"},
    "arri-logc3": {"transfer": "ARRILogC3", "gamut": "ARRI Wide Gamut 3"},
    "arri-logc4": {"transfer": "ARRILogC4", "gamut": "ARRI Wide Gamut 4"},
    "slog3": {"transfer": "SLog3", "gamut": "S-Gamut3"},
    "redlog3g10": {"transfer": "Log3G10", "gamut": "REDWideGamutRGB"},
    "acescct": {"transfer": "ACEScct", "gamut": "ACEScg"},
    "aces2065-1": {"transfer": None, "gamut": "ACES2065-1"},
}


def _decode_transfer(value: np.ndarray, transfer: str, exposure_index: int) -> np.ndarray:
    """Apply the inverse transfer function (encoding → scene-linear)."""
    import colour

    if transfer == "ARRILogC3":
        if exposure_index not in ARRI_LOGC3_EXPOSURE_INDICES:
            raise ValueError(
                f"ARRI LogC3 Exposure Index {exposure_index} is not supported. "
                f"Valid EIs: {', '.join(map(str, ARRI_LOGC3_EXPOSURE_INDICES))}. "
                "For higher-EI ALEXA 35 footage (e.g. EI 8000), use color_space='arri-logc4' instead."
            )
        return colour.models.log_decoding_ARRILogC3(value, EI=exposure_index)
    if transfer == "ARRILogC4":
        return colour.models.log_decoding_ARRILogC4(value)
    if transfer == "SLog3":
        return colour.models.log_decoding_SLog3(value)
    if transfer == "Log3G10":
        return colour.models.log_decoding_Log3G10(value)
    if transfer == "ACEScct":
        return colour.models.log_decoding_ACEScct(value)
    if transfer in ("BT.709", "BT.2020"):
        # Rec.709 and Rec.2020 share the BT.709 OETF; invert it to scene-linear.
        return colour.models.oetf_inverse_BT709(value)
    raise ValueError(f"Unhandled transfer function '{transfer}'")


def decode_to_linear(
    image_rgb: np.ndarray,
    color_space: str,
    exposure_index: int = DEFAULT_EXPOSURE_INDEX,
) -> np.ndarray:
    """Decode native-encoded RGB to scene-linear in sRGB / Rec.709 primaries.

    Args:
        image_rgb: float array [..., 3] in RGB order, native code values
            (normalized 0-1 for integer sources, raw float for EXR).
        color_space: one of SUPPORTED_COLOR_SPACES. ``srgb``/``linear`` are
            passthrough (see requires_decode) — calling decode on them still
            returns a sensible linear result for completeness.
        exposure_index: ARRI LogC3 Exposure Index (ISO). Ignored by other spaces.

    Returns:
        float32 scene-linear array [..., 3], sRGB primaries.
    """
    from CorridorKeyModule.core.color_utils import srgb_to_linear

    if color_space == "linear":
        return np.asarray(image_rgb, dtype=np.float32)
    if color_space == "srgb":
        return np.asarray(srgb_to_linear(np.asarray(image_rgb, dtype=np.float32)), dtype=np.float32)

    spec = _DECODE_SPECS.get(color_space)
    if spec is None:
        raise ValueError(f"Unknown color_space '{color_space}'. Valid: {', '.join(SUPPORTED_COLOR_SPACES)}")

    import colour

    value = np.asarray(image_rgb, dtype=np.float32)

    transfer = spec["transfer"]
    if transfer is not None:
        value = np.asarray(_decode_transfer(value, transfer, exposure_index), dtype=np.float32)

    # Convert from the source gamut primaries to sRGB primaries (linear, no CCTF).
    source_cs = colour.RGB_COLOURSPACES[spec["gamut"]]
    target_cs = colour.RGB_COLOURSPACES["sRGB"]
    linear_srgb = colour.RGB_to_RGB(
        value,
        source_cs,
        target_cs,
        apply_cctf_decoding=False,
        apply_cctf_encoding=False,
    )
    return np.asarray(linear_srgb, dtype=np.float32)
