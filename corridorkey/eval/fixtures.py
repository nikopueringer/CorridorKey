"""Deterministic diagnostic fixture generation for matting benchmarks.

Generates canonical test cases isolating specific failure classes:
  - opaque_subject: Clean geometric silhouette, sharp boundary.
  - soft_hair: Fine semi-transparent filaments with fractional alpha.
  - motion_blur: Directional motion blur across transition boundary.
  - green_foreground: Chromatic collision (green foreground on green backing).
  - blue_foreground: Chromatic collision (blue foreground on blue backing).
  - bad_hint_eroded: Severely eroded guide hint (undersegmentation recovery).
  - bad_hint_dilated: Severely dilated guide hint with false-positive islands.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _save_png_rgb(path: Path, rgb: np.ndarray) -> None:
    """Save float32 [0, 1] RGB array as 8-bit PNG."""
    bgr = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _save_png_gray(path: Path, gray: np.ndarray) -> None:
    """Save float32 [0, 1] 2D array as 8-bit grayscale PNG."""
    u8 = (np.clip(gray, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), u8)


def generate_opaque_subject(size: int = 256) -> dict[str, np.ndarray | dict]:
    """Generate opaque geometric subject with clean boundary."""
    h, w = size, size
    alpha = np.zeros((h, w), dtype=np.float32)
    # Draw central rounded rectangle / circle
    cv2.circle(alpha, (w // 2, h // 2), int(size * 0.35), 1.0, -1, lineType=cv2.LINE_AA)

    # Foreground: warm orange/tan tone
    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[:, :] = [0.85, 0.45, 0.20]

    # Screen: clean studio green
    screen = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    screen_field = np.tile(screen, (h, w, 1))

    # Composite: alpha * F + (1 - alpha) * B
    comp = alpha[:, :, None] * fg + (1.0 - alpha[:, :, None]) * screen_field

    # Hint: slight 2px Gaussian blur of alpha
    hint = cv2.GaussianBlur(alpha, (5, 5), 1.2)

    meta = {
        "tags": ["opaque", "clean", "baseline"],
        "screen_rgb": [0.0, 1.0, 0.0],
        "description": "Clean opaque circular foreground on standard green backing.",
    }
    return {
        "composite": comp,
        "alpha_hint": hint,
        "target_alpha": alpha,
        "target_foreground": fg,
        "metadata": meta,
    }


def generate_soft_hair(size: int = 256) -> dict[str, np.ndarray | dict]:
    """Generate subject with fine semi-transparent radial hair strands."""
    h, w = size, size
    alpha = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, int(h * 0.55)
    r_core = int(size * 0.22)

    # Draw solid core
    cv2.circle(alpha, (cx, cy), r_core, 1.0, -1, lineType=cv2.LINE_AA)

    # Draw 48 radial hair filaments with varying angles and lengths
    rng = np.random.default_rng(1337)
    for i in range(48):
        angle = np.pi * 0.8 + (np.pi * 1.4) * (i / 48.0) + rng.normal(0, 0.03)
        length = r_core + rng.uniform(int(size * 0.08), int(size * 0.25))
        x2 = int(cx + length * np.cos(angle))
        y2 = int(cy - length * np.sin(angle))
        # Draw line with fractional alpha
        strand = np.zeros((h, w), dtype=np.float32)
        cv2.line(strand, (cx, cy), (x2, y2), 1.0, thickness=1, lineType=cv2.LINE_AA)
        strand = cv2.GaussianBlur(strand, (3, 3), 0.6) * rng.uniform(0.35, 0.75)
        alpha = np.maximum(alpha, strand)

    alpha = np.clip(alpha, 0.0, 1.0)

    # Foreground: dark brown hair
    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[:, :] = [0.22, 0.14, 0.08]

    # Screen: studio green
    screen = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    screen_field = np.tile(screen, (h, w, 1))

    comp = alpha[:, :, None] * fg + (1.0 - alpha[:, :, None]) * screen_field

    # Hint: only covers the solid core (simulating SAM / coarse human segmenter)
    hint = np.zeros((h, w), dtype=np.float32)
    cv2.circle(hint, (cx, cy), int(r_core * 0.95), 1.0, -1, lineType=cv2.LINE_AA)
    hint = cv2.GaussianBlur(hint, (9, 9), 2.5)

    meta = {
        "tags": ["hair", "soft_hair", "fractional_alpha"],
        "screen_rgb": [0.0, 1.0, 0.0],
        "description": "Fine hair filaments extending into backing screen with coarse solid-core hint.",
    }
    return {
        "composite": comp,
        "alpha_hint": hint,
        "target_alpha": alpha,
        "target_foreground": fg,
        "metadata": meta,
    }


def generate_motion_blur(size: int = 256) -> dict[str, np.ndarray | dict]:
    """Generate moving foreground with directional horizontal blur."""
    h, w = size, size
    base_alpha = np.zeros((h, w), dtype=np.float32)
    cv2.circle(base_alpha, (w // 2, h // 2), int(size * 0.22), 1.0, -1)

    # Horizontal motion blur kernel (length 25px)
    k_len = 25
    kernel = np.zeros((k_len, k_len), dtype=np.float32)
    kernel[k_len // 2, :] = 1.0 / k_len
    blurred_alpha = cv2.filter2D(base_alpha, -1, kernel)

    # Foreground: magenta/red prop
    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[:, :] = [0.90, 0.20, 0.35]

    screen = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    screen_field = np.tile(screen, (h, w, 1))

    comp = blurred_alpha[:, :, None] * fg + (1.0 - blurred_alpha[:, :, None]) * screen_field

    # Hint: static unblurred center
    hint = cv2.GaussianBlur(base_alpha, (9, 9), 2.0)

    meta = {
        "tags": ["motion_blur", "translucent", "temporal_spread"],
        "screen_rgb": [0.0, 1.0, 0.0],
        "description": "High-velocity horizontal motion blur across green backing screen.",
    }
    return {
        "composite": comp,
        "alpha_hint": hint,
        "target_alpha": blurred_alpha,
        "target_foreground": fg,
        "metadata": meta,
    }


def generate_green_foreground(size: int = 256) -> dict[str, np.ndarray | dict]:
    """Generate green subject on green screen (severe chroma collision)."""
    h, w = size, size
    alpha = np.zeros((h, w), dtype=np.float32)
    pts = np.array(
        [
            [w * 0.5, h * 0.15],
            [w * 0.8, h * 0.5],
            [w * 0.65, h * 0.85],
            [w * 0.35, h * 0.85],
            [w * 0.2, h * 0.5],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(alpha, [pts], 1.0, lineType=cv2.LINE_AA)

    # Green foreground (e.g. olive / lime shirt)
    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[:, :] = [0.20, 0.78, 0.25]

    # Screen: pure green
    screen = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    screen_field = np.tile(screen, (h, w, 1))

    comp = alpha[:, :, None] * fg + (1.0 - alpha[:, :, None]) * screen_field
    hint = cv2.GaussianBlur(alpha, (7, 7), 1.5)

    meta = {
        "tags": ["green_foreground", "chroma_collision", "adversarial_color"],
        "screen_rgb": [0.0, 1.0, 0.0],
        "description": "Green foliage/fabric subject against green backing screen (delta-E collision).",
    }
    return {
        "composite": comp,
        "alpha_hint": hint,
        "target_alpha": alpha,
        "target_foreground": fg,
        "metadata": meta,
    }


def generate_blue_foreground(size: int = 256) -> dict[str, np.ndarray | dict]:
    """Generate blue subject on blue screen (severe chroma collision)."""
    h, w = size, size
    alpha = np.zeros((h, w), dtype=np.float32)
    pts = np.array(
        [
            [w * 0.5, h * 0.15],
            [w * 0.85, h * 0.55],
            [w * 0.65, h * 0.85],
            [w * 0.35, h * 0.85],
            [w * 0.15, h * 0.55],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(alpha, [pts], 1.0, lineType=cv2.LINE_AA)

    # Blue foreground (navy/cobalt subject)
    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[:, :] = [0.15, 0.25, 0.88]

    # Screen: blue backing
    screen = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    screen_field = np.tile(screen, (h, w, 1))

    comp = alpha[:, :, None] * fg + (1.0 - alpha[:, :, None]) * screen_field
    hint = cv2.GaussianBlur(alpha, (7, 7), 1.5)

    meta = {
        "tags": ["blue_foreground", "chroma_collision", "adversarial_color"],
        "screen_rgb": [0.0, 0.0, 1.0],
        "description": "Cobalt blue foreground object against blue backing screen.",
    }
    return {
        "composite": comp,
        "alpha_hint": hint,
        "target_alpha": alpha,
        "target_foreground": fg,
        "metadata": meta,
    }


def generate_bad_hint_eroded(size: int = 256) -> dict[str, np.ndarray | dict]:
    """Generate subject with severely eroded hint (14px undersegmentation)."""
    h, w = size, size
    alpha = np.zeros((h, w), dtype=np.float32)
    # Head and torso
    cv2.circle(alpha, (w // 2, int(h * 0.35)), int(size * 0.18), 1.0, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(
        alpha,
        (int(w * 0.25), int(h * 0.45)),
        (int(w * 0.75), int(h * 0.90)),
        1.0,
        -1,
        lineType=cv2.LINE_AA,
    )

    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[:, :] = [0.82, 0.55, 0.35]

    screen = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    screen_field = np.tile(screen, (h, w, 1))
    comp = alpha[:, :, None] * fg + (1.0 - alpha[:, :, None]) * screen_field

    # Heavily eroded hint: erode by 15px structuring element
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    hint_binary = cv2.erode(alpha, k)
    hint = cv2.GaussianBlur(hint_binary, (9, 9), 3.0)

    meta = {
        "tags": ["bad_hint", "bad_hint_eroded", "undersegmentation"],
        "screen_rgb": [0.0, 1.0, 0.0],
        "description": "Severely eroded alpha hint missing major perimeter regions.",
    }
    return {
        "composite": comp,
        "alpha_hint": hint,
        "target_alpha": alpha,
        "target_foreground": fg,
        "metadata": meta,
    }


def generate_bad_hint_dilated(size: int = 256) -> dict[str, np.ndarray | dict]:
    """Generate subject with severely dilated hint and spurious false-positive islands."""
    h, w = size, size
    alpha = np.zeros((h, w), dtype=np.float32)
    cv2.circle(alpha, (w // 2, int(h * 0.35)), int(size * 0.18), 1.0, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(
        alpha,
        (int(w * 0.25), int(h * 0.45)),
        (int(w * 0.75), int(h * 0.90)),
        1.0,
        -1,
        lineType=cv2.LINE_AA,
    )

    fg = np.zeros((h, w, 3), dtype=np.float32)
    fg[:, :] = [0.82, 0.55, 0.35]

    screen = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    screen_field = np.tile(screen, (h, w, 1))
    comp = alpha[:, :, None] * fg + (1.0 - alpha[:, :, None]) * screen_field

    # Dilate by 15px structuring element
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    hint_dilated = cv2.dilate(alpha, k)

    # Add false-positive islands in backing screen
    cv2.circle(hint_dilated, (int(w * 0.15), int(h * 0.2)), int(size * 0.08), 1.0, -1)
    cv2.circle(hint_dilated, (int(w * 0.85), int(h * 0.75)), int(size * 0.06), 1.0, -1)
    hint = cv2.GaussianBlur(hint_dilated, (9, 9), 3.0)

    meta = {
        "tags": ["bad_hint", "bad_hint_dilated", "oversegmentation", "false_islands"],
        "screen_rgb": [0.0, 1.0, 0.0],
        "description": "Severely dilated alpha hint with extraneous false-positive background islands.",
    }
    return {
        "composite": comp,
        "alpha_hint": hint,
        "target_alpha": alpha,
        "target_foreground": fg,
        "metadata": meta,
    }


FIXTURE_GENERATORS = {
    "opaque_subject": generate_opaque_subject,
    "soft_hair": generate_soft_hair,
    "motion_blur": generate_motion_blur,
    "green_foreground": generate_green_foreground,
    "blue_foreground": generate_blue_foreground,
    "bad_hint_eroded": generate_bad_hint_eroded,
    "bad_hint_dilated": generate_bad_hint_dilated,
}


def create_diagnostic_fixtures(target_root: Path | str, size: int = 256) -> list[Path]:
    """Generate all 7 canonical benchmark fixtures into target_root.

    Args:
        target_root: Destination directory for fixture folders.
        size: Spatial square dimension for fixtures (default 256).

    Returns:
        List of paths to created fixture directories.
    """
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    for name, gen_fn in FIXTURE_GENERATORS.items():
        case_dir = root / name
        case_dir.mkdir(parents=True, exist_ok=True)

        data = gen_fn(size=size)
        _save_png_rgb(case_dir / "composite.png", data["composite"])
        _save_png_gray(case_dir / "alpha_hint.png", data["alpha_hint"])
        _save_png_gray(case_dir / "target_alpha.png", data["target_alpha"])
        _save_png_rgb(case_dir / "target_foreground.png", data["target_foreground"])

        with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(data["metadata"], f, indent=2)

        created_paths.append(case_dir)

    return created_paths
