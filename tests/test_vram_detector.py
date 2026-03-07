"""Property-based tests for VRAM detection and tile size recommendation."""

from unittest.mock import MagicMock, patch

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from CorridorKeyModule.tiled_inference import VRAMDetector, align_to_patch_stride


# Feature: tiled-inference, Property 7: VRAM tier recommendation
@given(vram_gb=st.floats(min_value=0.5, max_value=48.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_vram_tier_recommendation(vram_gb: float) -> None:
    """For any positive VRAM float (0.5–48 GB) on a CUDA device,
    VRAMDetector.recommend_tile_size() returns the correct tier:
      - >= 24 GB  -> None (no tiling)
      - 12–24 GB  -> align_to_patch_stride(1536) = 1344
      - 8–12 GB   -> align_to_patch_stride(1024) = 896
      - < 8 GB    -> align_to_patch_stride(768) = 672
    Each returned tile size is a multiple of 224 (Hiera compatibility).

    **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
    """
    vram_bytes = int(vram_gb * (1024**3))

    mock_props = MagicMock()
    mock_props.total_mem = vram_bytes

    device = torch.device("cuda:0")

    with patch("CorridorKeyModule.tiled_inference.torch.cuda.get_device_properties", return_value=mock_props):
        result = VRAMDetector.recommend_tile_size(device)

    if vram_gb >= 24:
        assert result is None, f"Expected None for {vram_gb:.1f} GB, got {result}"
    elif vram_gb >= 12:
        expected = align_to_patch_stride(1536)
        assert result == expected, f"Expected {expected} for {vram_gb:.1f} GB, got {result}"
        assert result % 224 == 0, f"{result} is not a multiple of 224"
    elif vram_gb >= 8:
        expected = align_to_patch_stride(1024)
        assert result == expected, f"Expected {expected} for {vram_gb:.1f} GB, got {result}"
        assert result % 224 == 0, f"{result} is not a multiple of 224"
    else:
        expected = align_to_patch_stride(768)
        assert result == expected, f"Expected {expected} for {vram_gb:.1f} GB, got {result}"
        assert result % 224 == 0, f"{result} is not a multiple of 224"
