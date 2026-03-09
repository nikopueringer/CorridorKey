"""
GreenScreenMattingDataset — training data loader for torchdistill distillation.

Expected directory layout
-------------------------
<root>/
  train/
    input/          RGB green-screen frames  (.png / .jpg / .exr)
    alpha_hint/     Coarse alpha hint masks  (.png / .exr, single-channel)
    alpha_gt/       Ground-truth alpha mattes (.png / .exr, single-channel)
    fg_gt/          Ground-truth foreground   (.png / .exr, 3-channel) [optional]
  val/
    input/
    alpha_hint/
    alpha_gt/
    fg_gt/          [optional]

Each subdirectory must contain files with matching stem names so that
``sorted()`` on ``input/`` aligns with the other directories.

Sample returned
---------------
The dataset returns ``(sample, target)`` tuples that are then wrapped by
torchdistill's ``BaseDatasetWrapper`` into ``(sample, target, supp_dict)``.

* ``sample``: ``torch.Tensor`` of shape ``[4, H, W]``  float32 — concatenation
  of the normalised RGB frame and the coarse alpha hint.
* ``target``: ``dict`` with keys

  * ``"alpha"`` — ``[1, H, W]`` float32 GT alpha, values in ``[0, 1]``
  * ``"fg"``    — ``[3, H, W]`` float32 GT foreground, values in ``[0, 1]``
    (present only when ``fg_gt/`` exists, otherwise an all-ones tensor)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet statistics used by GreenFormer's inference engine
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".exr"}


def _load_image(path: Path) -> np.ndarray:
    """Load an image from *path* and return an ``uint8`` HWC or HW ndarray."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def _to_float_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR/BGRA uint8 image to float32 RGB in [0, 1]."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _to_float_alpha(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale or single-channel image to float32 in [0, 1]."""
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32) / 255.0


def _list_images(directory: Path) -> list[Path]:
    """Return a sorted list of image files in *directory*."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTS)


class GreenScreenMattingDataset(Dataset):
    """
    Paired dataset for green-screen matting distillation training.

    Parameters
    ----------
    root:
        Path to the split directory (e.g. ``data/train``).
    img_size:
        Square resolution to resize all images to before returning.
        Must match the student model's training resolution.
    """

    def __init__(self, root: str | Path, img_size: int = 512) -> None:
        self.root = Path(root)
        self.img_size = img_size

        self.input_paths = _list_images(self.root / "input")
        self.alpha_hint_paths = _list_images(self.root / "alpha_hint")
        self.alpha_gt_paths = _list_images(self.root / "alpha_gt")

        fg_gt_dir = self.root / "fg_gt"
        self.fg_gt_paths: list[Path] | None = _list_images(fg_gt_dir) if fg_gt_dir.is_dir() else None

        if len(self.input_paths) != len(self.alpha_hint_paths):
            raise ValueError(
                f"Mismatch: {len(self.input_paths)} input frames vs "
                f"{len(self.alpha_hint_paths)} alpha_hint frames in {self.root}"
            )
        if len(self.input_paths) != len(self.alpha_gt_paths):
            raise ValueError(
                f"Mismatch: {len(self.input_paths)} input frames vs "
                f"{len(self.alpha_gt_paths)} alpha_gt frames in {self.root}"
            )
        if self.fg_gt_paths is not None and len(self.fg_gt_paths) != len(self.input_paths):
            raise ValueError(
                f"Mismatch: {len(self.input_paths)} input frames vs "
                f"{len(self.fg_gt_paths)} fg_gt frames in {self.root}"
            )

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        size = (self.img_size, self.img_size)

        # --- Load and resize input frame ---
        rgb = _to_float_rgb(_load_image(self.input_paths[index]))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]
        rgb_norm = (rgb_t - _IMAGENET_MEAN) / _IMAGENET_STD

        # --- Load and resize alpha hint ---
        hint = _to_float_alpha(_load_image(self.alpha_hint_paths[index]))
        hint = cv2.resize(hint, size, interpolation=cv2.INTER_LINEAR)
        hint_t = torch.from_numpy(hint).unsqueeze(0)  # [1, H, W]

        # Model input: [4, H, W]
        sample = torch.cat([rgb_norm, hint_t], dim=0)

        # --- Load ground-truth alpha ---
        alpha_gt = _to_float_alpha(_load_image(self.alpha_gt_paths[index]))
        alpha_gt = cv2.resize(alpha_gt, size, interpolation=cv2.INTER_LINEAR)
        alpha_gt_t = torch.from_numpy(alpha_gt).unsqueeze(0)  # [1, H, W]

        # --- Load ground-truth foreground (or use ones as placeholder) ---
        if self.fg_gt_paths is not None:
            fg_gt = _to_float_rgb(_load_image(self.fg_gt_paths[index]))
            fg_gt = cv2.resize(fg_gt, size, interpolation=cv2.INTER_LINEAR)
            fg_gt_t = torch.from_numpy(fg_gt).permute(2, 0, 1)  # [3, H, W]
        else:
            fg_gt_t = torch.ones(3, self.img_size, self.img_size, dtype=torch.float32)

        target: dict[str, torch.Tensor] = {
            "alpha": alpha_gt_t,
            "fg": fg_gt_t,
        }
        return sample, target
