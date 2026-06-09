"""Best-effort color-space auto-detection from file metadata.

Reads color tags from video streams (via ffprobe) and EXR headers (via OpenEXR)
and maps them to the names in CorridorKeyModule.core.colorspace. Detection is
best-effort: camera-native log encodings (ARRI LogC, S-Log3, …) are frequently
*not* recorded in standard container tags, so callers must fall back to an
explicit ``--color-space`` flag when this returns ``None``.

All functions return a supported color-space name or ``None`` (never raise).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Reference primaries (CIE xy) for matching EXR chromaticities to a gamut.
# Only gamuts we can act on are listed: sRGB/Rec.709 and ACES AP0.
_PRIMARY_SETS: dict[str, dict[str, tuple[float, float]]] = {
    # sRGB / Rec.709 share primaries — a linear EXR in these primaries is just "linear".
    "linear": {
        "red": (0.640, 0.330),
        "green": (0.300, 0.600),
        "blue": (0.150, 0.060),
    },
    # ACES AP0 (ACES2065-1)
    "aces2065-1": {
        "red": (0.7347, 0.2653),
        "green": (0.0000, 1.0000),
        "blue": (0.0001, -0.0770),
    },
}

_PRIMARY_TOLERANCE = 0.01


def _match_chromaticities(chroma: dict[str, tuple[float, float]]) -> str | None:
    """Match a {red,green,blue: (x,y)} dict to a known gamut name, or None."""
    for name, ref in _PRIMARY_SETS.items():
        if all(
            key in chroma
            and abs(chroma[key][0] - ref[key][0]) <= _PRIMARY_TOLERANCE
            and abs(chroma[key][1] - ref[key][1]) <= _PRIMARY_TOLERANCE
            for key in ("red", "green", "blue")
        ):
            return name
    return None


def detect_color_space_from_video(path: str) -> str | None:
    """Map ffprobe color tags of a video to a color-space name, or None.

    Recognizes the standard container tags (bt709, bt2020, sRGB, linear). ARRI/
    Sony/RED log encodings are usually untagged here and return None.
    """
    try:
        from .ffmpeg_tools import probe_video

        info = probe_video(path)
    except Exception as exc:
        logger.debug("Color-space video probe failed for %s: %s", path, exc)
        return None

    transfer = (info.get("color_transfer") or "").lower()
    primaries = (info.get("color_primaries") or "").lower()

    if transfer in ("iec61966-2-1",):
        return "srgb"
    if transfer in ("linear",):
        return "linear"
    # PQ (smpte2084) and HLG (arib-std-b67) are HDR transfers that ffprobe often
    # pairs with bt2020 primaries. We can't correctly decode these as SDR rec2020,
    # so return None and let the caller fall back to srgb with a warning.
    if transfer in ("smpte2084", "smpte2086", "arib-std-b67"):
        return None
    if "bt2020" in transfer or "bt2020" in primaries:
        return "rec2020"
    if transfer in ("bt709", "bt1361") or primaries == "bt709":
        return "rec709"
    return None


def _read_exr_header(path: str) -> dict | None:
    """Read an EXR header as a dict, tolerating OpenEXR API variants. None on failure."""
    try:
        import OpenEXR
    except ImportError:
        logger.debug("OpenEXR not installed — skipping EXR color-space detection.")
        return None

    # OpenEXR v3.x exposes OpenEXR.File with a `.header` dict; the legacy binding
    # exposes OpenEXR.InputFile(path).header(). Try both.
    try:
        if hasattr(OpenEXR, "File"):
            with OpenEXR.File(path) as f:
                header = getattr(f, "header", None)
                if callable(header):
                    header = header()
                # New API stores the header on each part.
                if header is None and getattr(f, "parts", None):
                    header = f.parts[0].header
                return dict(header) if header is not None else None
        in_file = OpenEXR.InputFile(path)
        try:
            return dict(in_file.header())
        finally:
            in_file.close()
    except Exception as exc:
        logger.debug("Could not read EXR header for %s: %s", path, exc)
        return None


def _chromaticities_from_header(header: dict) -> dict[str, tuple[float, float]] | None:
    """Extract {red,green,blue: (x,y)} from an EXR header's chromaticities attr."""
    chroma = header.get("chromaticities")
    if chroma is None:
        return None
    try:
        # Legacy Imath.Chromaticities: attributes .red.x/.red.y, etc.
        if hasattr(chroma, "red"):
            return {
                "red": (float(chroma.red.x), float(chroma.red.y)),
                "green": (float(chroma.green.x), float(chroma.green.y)),
                "blue": (float(chroma.blue.x), float(chroma.blue.y)),
            }
        # Sequence form: (rx, ry, gx, gy, bx, by, wx, wy)
        vals = list(chroma)
        if len(vals) >= 6:
            return {
                "red": (float(vals[0]), float(vals[1])),
                "green": (float(vals[2]), float(vals[3])),
                "blue": (float(vals[4]), float(vals[5])),
            }
    except Exception as exc:
        logger.debug("Could not parse EXR chromaticities: %s", exc)
    return None


def detect_color_space_from_exr(path: str) -> str | None:
    """Detect a color-space name from an EXR header, or None.

    Checks an explicit string ``colorSpace``/``colorspace`` attribute first, then
    falls back to matching the ``chromaticities`` primaries. EXRs are linear by
    convention, so recognized gamuts map to ``linear`` (sRGB/Rec.709 primaries) or
    ``aces2065-1`` (ACES AP0). Returns None when nothing recognizable is present.
    """
    header = _read_exr_header(path)
    if not header:
        return None

    for key in ("colorSpace", "colorspace", "color_space"):
        raw = header.get(key)
        if raw is None:
            continue
        text = (raw.decode() if isinstance(raw, bytes) else str(raw)).strip().lower()
        if "aces" in text and ("2065" in text or "ap0" in text):
            return "aces2065-1"
        if text in ("srgb",):
            return "srgb"
        if text in ("linear", "scene-linear", "lin_srgb", "linear srgb"):
            return "linear"

    chroma = _chromaticities_from_header(header)
    if chroma is not None:
        return _match_chromaticities(chroma)
    return None


def detect_color_space(asset_path: str, asset_type: str, first_frame: str | None = None) -> str | None:
    """Dispatch color-space detection by asset type.

    Args:
        asset_path: Path to the input asset (video file or sequence directory).
        asset_type: "video" or "sequence".
        first_frame: For sequences, the resolved path to the first frame. If
            omitted, it is discovered from ``asset_path``.

    Returns the detected color-space name, or None if undeterminable.
    """
    if asset_type == "video":
        return detect_color_space_from_video(asset_path)

    frame = first_frame
    if frame is None and os.path.isdir(asset_path):
        try:
            from clip_manager import is_image_file

            files = sorted(f for f in os.listdir(asset_path) if is_image_file(f))
            frame = os.path.join(asset_path, files[0]) if files else None
        except Exception as exc:
            logger.debug("Could not list sequence %s for detection: %s", asset_path, exc)
            frame = None

    if frame and frame.lower().endswith(".exr"):
        return detect_color_space_from_exr(frame)
    return None
