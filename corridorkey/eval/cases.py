"""Dataset cases and sample contracts for matting evaluation.

Defines the canonical MattingEvalSample and functions to load diagnostic
challenge sets from disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class MattingEvalSample:
    """Canonical evaluation sample contract for matting quality assessment.

    Attributes:
        composite: Observed composite image tensor [3, H, W], float32 in [0, 1], sRGB.
        alpha_hint: Coarse/guide alpha hint tensor [1, H, W], float32 in [0, 1].
        target_alpha: Ground truth alpha matte tensor [1, H, W], float32 in [0, 1].
        target_foreground: Ground truth unmultiplied straight foreground [3, H, W], float32 in [0, 1].
        screen_rgb: Optional known backing screen RGB tensor [3] or None.
        tags: Frozenset of descriptive failure-mode / category tags.
        name: Unique sample identifier or case name.
    """

    composite: torch.Tensor
    alpha_hint: torch.Tensor
    target_alpha: torch.Tensor
    target_foreground: torch.Tensor
    screen_rgb: torch.Tensor | None
    tags: frozenset[str]
    name: str = ""

    def __post_init__(self) -> None:
        # Validate shapes and dimensions
        if self.composite.ndim != 3 or self.composite.shape[0] != 3:
            raise ValueError(f"composite must have shape [3, H, W], got {self.composite.shape}")

        h, w = self.composite.shape[1], self.composite.shape[2]

        if self.alpha_hint.ndim == 2:
            object.__setattr__(self, "alpha_hint", self.alpha_hint.unsqueeze(0))
        elif self.alpha_hint.ndim != 3 or self.alpha_hint.shape[0] != 1:
            raise ValueError(f"alpha_hint must have shape [1, H, W], got {self.alpha_hint.shape}")

        if self.target_alpha.ndim == 2:
            object.__setattr__(self, "target_alpha", self.target_alpha.unsqueeze(0))
        elif self.target_alpha.ndim != 3 or self.target_alpha.shape[0] != 1:
            raise ValueError(f"target_alpha must have shape [1, H, W], got {self.target_alpha.shape}")

        if self.target_foreground.ndim != 3 or self.target_foreground.shape[0] != 3:
            raise ValueError(f"target_foreground must have shape [3, H, W], got {self.target_foreground.shape}")

        if (
            self.alpha_hint.shape[1:] != (h, w)
            or self.target_alpha.shape[1:] != (h, w)
            or self.target_foreground.shape[1:] != (h, w)
        ):
            raise ValueError(
                f"Spatial dimension mismatch: composite {(h, w)}, "
                f"alpha_hint {tuple(self.alpha_hint.shape[1:])}, "
                f"target_alpha {tuple(self.target_alpha.shape[1:])}, "
                f"target_foreground {tuple(self.target_foreground.shape[1:])}"
            )

        if self.screen_rgb is not None:
            if self.screen_rgb.numel() != 3:
                raise ValueError(f"screen_rgb must have 3 elements, got {self.screen_rgb.shape}")
            object.__setattr__(self, "screen_rgb", self.screen_rgb.view(3).float())


def _read_rgb(path: Path) -> np.ndarray:
    """Read an image file as float32 RGB in range [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _read_grayscale(path: Path) -> np.ndarray:
    """Read an image file as float32 single-channel in range [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read grayscale mask at {path}")
    return img.astype(np.float32) / 255.0


def load_sample_from_directory(dir_path: Path | str) -> MattingEvalSample:
    """Load a MattingEvalSample from a directory containing fixture assets.

    Expected directory contents:
      - composite.png (or .jpg, .exr)
      - alpha_hint.png (or .jpg)
      - target_alpha.png (or .jpg)
      - target_foreground.png (or .jpg, .exr)
      - metadata.json (optional, containing 'tags' and 'screen_rgb')

    Args:
        dir_path: Path to the case directory.

    Returns:
        Populated MattingEvalSample.
    """
    case_dir = Path(dir_path)
    if not case_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {case_dir}")

    # Find composite
    comp_candidates = list(case_dir.glob("composite.*"))
    if not comp_candidates:
        raise FileNotFoundError(f"composite image missing in {case_dir}")
    comp_np = _read_rgb(comp_candidates[0])

    # Find alpha hint
    hint_candidates = list(case_dir.glob("alpha_hint.*"))
    if not hint_candidates:
        raise FileNotFoundError(f"alpha_hint image missing in {case_dir}")
    hint_np = _read_grayscale(hint_candidates[0])

    # Find target alpha
    alpha_candidates = list(case_dir.glob("target_alpha.*"))
    if not alpha_candidates:
        raise FileNotFoundError(f"target_alpha image missing in {case_dir}")
    alpha_np = _read_grayscale(alpha_candidates[0])

    # Find target foreground
    fg_candidates = list(case_dir.glob("target_foreground.*"))
    if not fg_candidates:
        raise FileNotFoundError(f"target_foreground image missing in {case_dir}")
    fg_np = _read_rgb(fg_candidates[0])

    # Load metadata if present
    metadata_path = case_dir / "metadata.json"
    tags: set[str] = set()
    screen_rgb: torch.Tensor | None = None

    if metadata_path.is_file():
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        tags = set(meta.get("tags", []))
        if "screen_rgb" in meta and meta["screen_rgb"] is not None:
            screen_rgb = torch.tensor(meta["screen_rgb"], dtype=torch.float32)

    # Always include directory name as default tag if no tags defined
    tags.add(case_dir.name)

    comp_tensor = torch.from_numpy(comp_np).permute(2, 0, 1).float()
    hint_tensor = torch.from_numpy(hint_np).unsqueeze(0).float()
    alpha_tensor = torch.from_numpy(alpha_np).unsqueeze(0).float()
    fg_tensor = torch.from_numpy(fg_np).permute(2, 0, 1).float()

    return MattingEvalSample(
        composite=comp_tensor,
        alpha_hint=hint_tensor,
        target_alpha=alpha_tensor,
        target_foreground=fg_tensor,
        screen_rgb=screen_rgb,
        tags=frozenset(tags),
        name=case_dir.name,
    )


def load_dataset(dataset_dir: Path | str) -> list[MattingEvalSample]:
    """Load all evaluation cases from a dataset root directory.

    If dataset_dir itself is a single case directory (has composite.*), returns [sample].
    Otherwise, scans all subdirectories containing composite.*.

    Args:
        dataset_dir: Path to directory containing case folders or a single case folder.

    Returns:
        Sorted list of MattingEvalSample objects.
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # Check if single case
    if list(root.glob("composite.*")):
        return [load_sample_from_directory(root)]

    samples: list[MattingEvalSample] = []
    # Discover child directories containing composite.*
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and list(sub.glob("composite.*")):
            samples.append(load_sample_from_directory(sub))

    if not samples:
        raise ValueError(f"No valid evaluation cases found under {root}")

    return samples
