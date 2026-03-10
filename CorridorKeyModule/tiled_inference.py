"""Tiled inference utilities for CorridorKey's GreenFormer model.

Provides tile decomposition, cosine blend ramps, and grid computation
to enable memory-efficient inference on consumer GPUs.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def align_to_patch_stride(tile_size: int, patch_stride: int = 224) -> int:
    """Round *tile_size* down to the nearest multiple of *patch_stride*.

    The default stride of 224 is the LCM of Hiera's requirements:
      - patch_embed stride = 4
      - q_stride = 2 applied 3 times → token grid divisible by 8 → 4 × 8 = 32
      - backbone patch stride = 7
      - LCM(32, 7) = 224

    This ensures tile dimensions are compatible with Hiera's patch embedding,
    position embeddings, and the Unroll/Reroll token reordering.
    """
    return (tile_size // patch_stride) * patch_stride


def build_cosine_ramp_2d(tile_size: int, overlap_size: int) -> torch.Tensor:
    """Build a 2-D weight mask with cosine ramps in the overlap borders.

    Returns a ``[1, 1, tile_size, tile_size]`` tensor where:
    * The core region (pixels more than *overlap_size* from any edge) is 1.0.
    * The overlap border uses a cosine ramp ``0.5 - 0.5 * cos(π * t)`` where
      *t* goes from 0 at the outer edge to 1 at the core boundary.
    * The 2-D mask is the outer product of horizontal and vertical 1-D ramps.
    """
    ramp_1d = torch.ones(tile_size)

    for i in range(overlap_size):
        t = (i + 0.5) / overlap_size
        w = 0.5 - 0.5 * math.cos(math.pi * t)
        ramp_1d[i] = w
        ramp_1d[tile_size - 1 - i] = w

    # Outer product of horizontal and vertical ramps → 2-D mask
    mask = ramp_1d.unsqueeze(0) * ramp_1d.unsqueeze(1)  # [tile_size, tile_size]
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tile_size, tile_size]


def validate_overlap(tile_size: int, overlap_size: int) -> int:
    """Validate and possibly clamp *overlap_size* for the given *tile_size*.

    The default ``overlap_size`` used by callers is **128** pixels.

    Validation rules:

    * If *overlap_size* is less than 65 (the CNNRefinerModule receptive field),
      it is clamped to 65 and a :func:`warnings.warn` is emitted.
    * If *overlap_size* is greater than or equal to half of *tile_size*, a
      :class:`ValueError` is raised because the overlap would consume the
      entire tile leaving no unique core region.

    Parameters
    ----------
    tile_size:
        Side length of each square tile in pixels.
    overlap_size:
        Requested overlap in pixels between adjacent tiles.

    Returns
    -------
    int
        The validated (and possibly clamped) overlap size.

    Raises
    ------
    ValueError
        If *overlap_size* >= *tile_size* / 2.
    """
    if overlap_size >= tile_size / 2:
        raise ValueError(f"overlap_size ({overlap_size}) must be less than half tile_size ({tile_size})")

    if overlap_size < 65:
        warnings.warn(
            f"overlap_size ({overlap_size}) is below the CNNRefinerModule "
            f"receptive field of 65 pixels. Clamping to 65 to prevent "
            f"seam artifacts.",
            stacklevel=2,
        )
        overlap_size = 65

    return overlap_size


@dataclass
class TilePosition:
    """Grid position and pixel coordinates of a single tile."""

    row: int
    col: int
    y_start: int
    x_start: int


def compute_tile_grid(
    img_size: int,
    tile_size: int,
    overlap_size: int,
) -> tuple[list[TilePosition], int]:
    """Compute tile positions and the padded image size.

    Parameters
    ----------
    img_size:
        Original (square) image dimension in pixels.
    tile_size:
        Side length of each square tile in pixels.
    overlap_size:
        Number of overlapping pixels between adjacent tiles on each edge.

    Returns
    -------
    tiles:
        List of :class:`TilePosition` describing every tile in the grid.
    padded_size:
        The padded image dimension that exactly fits the tile grid.
    """
    stride = tile_size - overlap_size
    n_tiles = math.ceil((img_size - overlap_size) / stride)
    padded_size = n_tiles * stride + overlap_size

    tiles: list[TilePosition] = []
    for row in range(n_tiles):
        for col in range(n_tiles):
            tiles.append(
                TilePosition(
                    row=row,
                    col=col,
                    y_start=row * stride,
                    x_start=col * stride,
                )
            )

    return tiles, padded_size


logger = logging.getLogger(__name__)


class VRAMDetector:
    """Auto-detect available VRAM and recommend tile size."""

    @staticmethod
    def recommend_tile_size(device: torch.device, img_size: int = 2048) -> int | None:
        """Return a recommended tile size, or ``None`` if tiling is not needed.

        CUDA thresholds (total VRAM):
        * ≥ 24 GB  → ``None`` (no tiling needed)
        * 12–24 GB → 1344
        * 8–12 GB  → 896
        * < 8 GB   → 672 + warning about reduced quality

        MPS and CPU devices always return ``None``.

        All returned tile sizes are aligned to 224 (Hiera compatibility)
        via :func:`align_to_patch_stride`.
        """
        device = torch.device(device) if not isinstance(device, torch.device) else device

        if device.type == "mps":
            logger.info("MPS uses unified memory — tiling not needed.")
            return None

        if device.type == "cpu":
            logger.info("CPU mode — tiling not applicable.")
            return None

        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            total_gb = props.total_mem / (1024**3)

            if total_gb >= 24:
                return None

            if total_gb >= 12:
                return align_to_patch_stride(1536)

            if total_gb >= 8:
                return align_to_patch_stride(1024)

            # < 8 GB
            warnings.warn(
                f"GPU has only {total_gb:.1f} GB VRAM. Using tile size 672 — quality may be reduced.",
                stacklevel=2,
            )
            return align_to_patch_stride(768)

        # Unknown device type — no tiling
        return None


class TiledInferenceEngine:
    """Wraps a GreenFormer model to run tiled inference with overlap blending.

    The engine decomposes the model's internal tensor space into overlapping
    square tiles, dispatches each tile through the model independently, blends
    overlapping regions with cosine ramps, and stitches the result.  This
    reduces peak VRAM from ~22.7 GB (full 2048² pass) to ~6–8 GB.

    Parameters
    ----------
    model:
        The GreenFormer model instance used for per-tile forward passes.
    device:
        Torch device the model lives on (e.g. ``torch.device("cuda")``).
    tile_size:
        Side length of each square tile in pixels.  Will be aligned down to
        the nearest multiple of the Hiera patch stride (7).
    overlap_size:
        Number of overlapping pixels between adjacent tiles on each edge.
        Clamped to a minimum of 65 (CNNRefinerModule receptive field).
    img_size:
        The full model input resolution (default 2048).
    """

    def __init__(
        self,
        model: object,
        device: torch.device,
        tile_size: int = 1024,
        overlap_size: int = 128,
        img_size: int = 2048,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.img_size = img_size

        # Align tile_size to Hiera-compatible stride (224 pixels)
        aligned_tile_size = align_to_patch_stride(tile_size)
        if aligned_tile_size != tile_size:
            logger.info(
                "Tile size %d is not a multiple of 224; rounded down to %d.",
                tile_size,
                aligned_tile_size,
            )
        self.tile_size = aligned_tile_size

        # Check if tiling can be skipped
        if self.tile_size >= img_size:
            self.skip_tiling = True
            self.overlap_size = overlap_size
            self.tiles: list[TilePosition] = []
            self.padded_size = img_size
            self.blend_ramp: torch.Tensor | None = None
            logger.info(
                "Tile size (%d) >= image size (%d), running full-resolution pass.",
                self.tile_size,
                img_size,
            )
            return

        # Validate overlap_size (clamp if < 65, reject if >= tile_size / 2)
        self.overlap_size = validate_overlap(self.tile_size, overlap_size)

        # Compute tile grid positions and padded image size
        self.tiles, self.padded_size = compute_tile_grid(img_size, self.tile_size, self.overlap_size)

        # Build the cosine blend ramp for overlap blending
        self.blend_ramp = build_cosine_ramp_2d(self.tile_size, self.overlap_size)

        self.skip_tiling = False
        self._first_frame_logged = False

        logger.info(
            "Tiled inference: %d tiles (%dx%d grid), tile_size=%d, overlap=%d, padded_size=%d.",
            len(self.tiles),
            int(math.sqrt(len(self.tiles))),
            int(math.sqrt(len(self.tiles))),
            self.tile_size,
            self.overlap_size,
            self.padded_size,
        )

    @torch.inference_mode()
    def forward_tiled(
        self,
        inp_tensor: torch.Tensor,
        refiner_scale: float | None = None,
        mixed_precision: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run tiled inference on a ``[1, 4, H, W]`` input tensor.

        Parameters
        ----------
        inp_tensor:
            Input tensor of shape ``[1, 4, H, W]``.
        refiner_scale:
            Optional refiner scale to apply per-tile.  When provided,
            ``self.model.refiner_scale`` is set before each forward pass.
        mixed_precision:
            Whether to use torch.autocast for mixed precision inference.

        Returns
        -------
        dict[str, torch.Tensor]
            Dict with ``'alpha'`` ``[1, 1, H, W]`` and ``'fg'`` ``[1, 3, H, W]``.
        """
        # --- Skip-tiling fast path ---
        if self.skip_tiling:
            if refiner_scale is not None:
                self.model.refiner_scale = refiner_scale
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=mixed_precision):
                return self.model(inp_tensor)

        # --- Tiled path ---
        rows = max(t.row for t in self.tiles) + 1
        cols = max(t.col for t in self.tiles) + 1
        logger.info(
            "Tiled inference: %d×%d grid, tile_size=%d, overlap=%d",
            rows,
            cols,
            self.tile_size,
            self.overlap_size,
        )
        t_start = time.perf_counter()

        _, _, H, W = inp_tensor.shape
        pad_h = self.padded_size - H
        pad_w = self.padded_size - W
        padded = F.pad(inp_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # Accumulators
        output_sum = torch.zeros(
            1,
            4,
            self.padded_size,
            self.padded_size,
            device=self.device,
        )
        weight_sum = torch.zeros(
            1,
            1,
            self.padded_size,
            self.padded_size,
            device=self.device,
        )

        self.blend_ramp = self.blend_ramp.to(self.device)

        tile_size = self.tile_size
        for tile in self.tiles:
            y, x = tile.y_start, tile.x_start
            tile_input = padded[:, :, y : y + tile_size, x : x + tile_size]

            if refiner_scale is not None:
                self.model.refiner_scale = refiner_scale

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=mixed_precision):
                tile_pred = self.model(tile_input)

            tile_out = torch.cat([tile_pred["alpha"], tile_pred["fg"]], dim=1)

            tile_out = tile_out * self.blend_ramp
            output_sum[:, :, y : y + tile_size, x : x + tile_size] += tile_out
            weight_sum[:, :, y : y + tile_size, x : x + tile_size] += self.blend_ramp

            del tile_input, tile_out
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if self.device.type == "cuda" and not self._first_frame_logged:
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            logger.info("Peak VRAM: %.1f MB", peak_mb)
            self._first_frame_logged = True

        result = output_sum / weight_sum.clamp(min=1e-8)
        result = result[:, :, : self.img_size, : self.img_size]

        elapsed = time.perf_counter() - t_start
        logger.info("Tiled pass completed in %.2f s", elapsed)

        return {"alpha": result[:, :1], "fg": result[:, 1:]}
