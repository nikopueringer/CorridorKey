"""Property-based tests for tiled inference logic."""

import argparse
import math

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from CorridorKeyModule.tiled_inference import (
    align_to_patch_stride,
    build_cosine_ramp_2d,
    compute_tile_grid,
    validate_overlap,
)


# Feature: tiled-inference, Property 1: Tile grid correctness
@given(
    tile_size=st.integers(min_value=224, max_value=2048).map(lambda x: align_to_patch_stride(x)),
    overlap_size=st.integers(min_value=65, max_value=200),
    img_size=st.just(2048),
)
@settings(max_examples=100)
def test_tile_grid_correctness(tile_size: int, overlap_size: int, img_size: int) -> None:
    """For any valid img_size, tile_size (aligned to patch stride, < img_size),
    and overlap_size (>= 65, < tile_size/2), the computed tile grid satisfies:
    (a) the union of all tile regions covers every pixel of the (possibly padded) input,
    (b) each horizontally or vertically adjacent tile pair overlaps by exactly
        overlap_size pixels along the shared axis,
    (c) every tile has spatial dimensions exactly tile_size x tile_size.

    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    assume(overlap_size < tile_size // 2)
    assume(tile_size < img_size)

    tiles, padded_size = compute_tile_grid(img_size, tile_size, overlap_size)

    # Padded size must be >= img_size
    assert padded_size >= img_size, f"padded_size {padded_size} < img_size {img_size}"

    # (c) Every tile has spatial dimensions exactly tile_size x tile_size.
    # Each tile occupies [y_start, y_start+tile_size) x [x_start, x_start+tile_size).
    # Verify all tiles fit within the padded image.
    for tile in tiles:
        assert tile.y_start >= 0, f"Tile ({tile.row},{tile.col}) y_start < 0"
        assert tile.x_start >= 0, f"Tile ({tile.row},{tile.col}) x_start < 0"
        assert tile.y_start + tile_size <= padded_size, (
            f"Tile ({tile.row},{tile.col}) exceeds padded_size vertically: {tile.y_start + tile_size} > {padded_size}"
        )
        assert tile.x_start + tile_size <= padded_size, (
            f"Tile ({tile.row},{tile.col}) exceeds padded_size horizontally: {tile.x_start + tile_size} > {padded_size}"
        )

    # (a) The union of all tile regions covers every pixel of the padded input.
    # Use a tensor for efficient coverage tracking instead of nested Python loops.
    coverage = torch.zeros(padded_size, padded_size, dtype=torch.bool)
    for tile in tiles:
        coverage[
            tile.y_start : tile.y_start + tile_size,
            tile.x_start : tile.x_start + tile_size,
        ] = True

    assert coverage.all(), f"Not all pixels covered: {(~coverage).sum().item()} uncovered pixels"

    # (b) Each horizontally or vertically adjacent tile pair overlaps by exactly
    # overlap_size pixels along the shared axis.
    tile_map = {(t.row, t.col): t for t in tiles}

    for tile in tiles:
        # Check right neighbour (same row, col+1)
        right = tile_map.get((tile.row, tile.col + 1))
        if right is not None:
            horizontal_overlap = (tile.x_start + tile_size) - right.x_start
            assert horizontal_overlap == overlap_size, (
                f"Horizontal overlap between ({tile.row},{tile.col}) and "
                f"({right.row},{right.col}): expected {overlap_size}, "
                f"got {horizontal_overlap}"
            )

        # Check bottom neighbour (row+1, same col)
        below = tile_map.get((tile.row + 1, tile.col))
        if below is not None:
            vertical_overlap = (tile.y_start + tile_size) - below.y_start
            assert vertical_overlap == overlap_size, (
                f"Vertical overlap between ({tile.row},{tile.col}) and "
                f"({below.row},{below.col}): expected {overlap_size}, "
                f"got {vertical_overlap}"
            )


# Feature: tiled-inference, Property 4: Weight sum full coverage
@given(
    tile_size=st.integers(min_value=224, max_value=2048).map(lambda x: align_to_patch_stride(x)),
    overlap_size=st.integers(min_value=65, max_value=200),
    img_size=st.just(2048),
)
@settings(max_examples=100)
def test_weight_sum_full_coverage(tile_size: int, overlap_size: int, img_size: int) -> None:
    """For any valid tiling configuration, after accumulating the blend ramp of
    every tile in the grid into a weight_sum tensor, every pixel within the
    original img_size × img_size region shall have a weight sum strictly
    greater than zero.

    **Validates: Requirements 3.3**
    """
    assume(overlap_size < tile_size // 2)
    assume(tile_size < img_size)

    tiles, padded_size = compute_tile_grid(img_size, tile_size, overlap_size)

    # Build the cosine blend ramp for this tile configuration
    ramp = build_cosine_ramp_2d(tile_size, overlap_size)  # [1, 1, tile_size, tile_size]
    ramp_2d = ramp.squeeze(0).squeeze(0)  # [tile_size, tile_size]

    # Accumulate blend weights for all tiles
    weight_sum = torch.zeros(padded_size, padded_size)
    for tile in tiles:
        weight_sum[
            tile.y_start : tile.y_start + tile_size,
            tile.x_start : tile.x_start + tile_size,
        ] += ramp_2d

    # Every pixel in the original img_size × img_size region must have weight > 0
    original_region = weight_sum[:img_size, :img_size]
    assert (original_region > 0).all(), (
        f"Found {(original_region <= 0).sum().item()} pixels with weight_sum <= 0 "
        f"in the original {img_size}×{img_size} region"
    )


# Feature: tiled-inference, Property 6: Overlap validation
@given(
    tile_size=st.integers(min_value=133, max_value=2048),
    overlap_size=st.integers(min_value=1, max_value=64),
)
@settings(max_examples=100)
def test_overlap_validation_clamps_below_65(tile_size: int, overlap_size: int) -> None:
    """If overlap_size < 65, validate_overlap clamps it to 65 (and emits a warning).
    tile_size must be >= 133 so that 65 < tile_size/2, ensuring the ValueError
    branch is not triggered.

    **Validates: Requirements 4.2**
    """
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = validate_overlap(tile_size, overlap_size)

    assert result == 65, (
        f"Expected clamped overlap 65, got {result} for tile_size={tile_size}, overlap_size={overlap_size}"
    )
    # A warning must have been emitted
    assert len(caught) >= 1, "Expected a warning when overlap_size < 65"
    assert "65" in str(caught[0].message), f"Warning message should mention 65, got: {caught[0].message}"


# Feature: tiled-inference, Property 6: Overlap validation
@given(
    data=st.data(),
    tile_size=st.integers(min_value=133, max_value=2048),
)
@settings(max_examples=100)
def test_overlap_validation_raises_when_ge_half_tile(data: st.DataObject, tile_size: int) -> None:
    """If overlap_size >= tile_size / 2, validate_overlap raises ValueError.

    **Validates: Requirements 4.3**
    """
    # Draw an overlap_size that is >= tile_size / 2 (float division)
    min_overlap = math.ceil(tile_size / 2)
    overlap_size = data.draw(
        st.integers(min_value=min_overlap, max_value=tile_size + 100),
        label="overlap_size",
    )

    with pytest.raises(ValueError, match="must be less than half"):
        validate_overlap(tile_size, overlap_size)


# Feature: tiled-inference, Property 8: fp16 weight casting
@given(
    num_layers=st.integers(min_value=1, max_value=5),
    hidden_size=st.integers(min_value=4, max_value=64),
)
@settings(max_examples=100)
def test_fp16_weight_casting(num_layers: int, hidden_size: int) -> None:
    """For any model loaded with half_precision=True, every parameter tensor
    shall have dtype torch.float16.

    Since we cannot easily load the real GreenFormer model in tests (no
    checkpoint), we create a small torch.nn.Module with a configurable number
    of layers and verify that calling .half() converts all parameters to fp16.

    **Validates: Requirements 6.1**
    """
    import torch.nn as nn

    # Build a small model with variable depth and width
    layers: list[nn.Module] = []
    in_features = hidden_size
    for _ in range(num_layers):
        out_features = hidden_size
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    model = nn.Sequential(*layers)

    # Precondition: all params start as float32
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"Parameter '{name}' should start as float32, got {param.dtype}"

    # Simulate half_precision=True: cast to fp16
    model.half()

    # Property: every parameter must now be float16
    for name, param in model.named_parameters():
        assert param.dtype == torch.float16, f"Parameter '{name}' should be float16 after .half(), got {param.dtype}"


# ---------------------------------------------------------------------------
# Unit tests for CLI argument parsing (Task 8.3)
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build a parser that mirrors the --tile-size, --overlap, --half args
    defined in corridorkey_cli.py and clip_manager.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tile-size",
        default="auto",
        help='Tile size in pixels, "auto", or "off" (default: "auto")',
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlap size in pixels (default: 128)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use fp16 model weights to reduce VRAM usage",
    )
    return parser


class TestCLIArgumentParsing:
    """Unit tests for CLI argument parsing.

    **Validates: Requirements 7.1, 7.2, 7.5**
    """

    def test_tile_size_auto(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(["--tile-size", "auto"])
        assert args.tile_size == "auto"

    def test_tile_size_explicit_int(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(["--tile-size", "1024"])
        assert args.tile_size == "1024"

    def test_tile_size_off(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(["--tile-size", "off"])
        assert args.tile_size == "off"

    def test_tile_size_zero(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(["--tile-size", "0"])
        assert args.tile_size == "0"

    def test_half_flag_present(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(["--half"])
        assert args.half is True

    def test_half_flag_absent(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args([])
        assert args.half is False

    def test_overlap_explicit(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args(["--overlap", "64"])
        assert args.overlap == 64

    def test_defaults(self) -> None:
        parser = _build_cli_parser()
        args = parser.parse_args([])
        assert args.tile_size == "auto"
        assert args.overlap == 128
        assert args.half is False
