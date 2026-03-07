"""Property-based tests for blend ramp utilities."""

import math

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from CorridorKeyModule.tiled_inference import align_to_patch_stride, build_cosine_ramp_2d


# Feature: tiled-inference, Property 2: Patch stride alignment
@given(tile_size=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=100)
def test_align_to_patch_stride_property(tile_size: int) -> None:
    """For any positive tile_size, align_to_patch_stride returns a value that is
    (a) a multiple of 224, (b) ≤ tile_size, and (c) within 223 of tile_size.

    The alignment of 224 is the LCM of Hiera's patch_embed stride (4),
    q_stride pooling requirements (32), and backbone patch stride (7).

    **Validates: Requirements 1.4**
    """
    result = align_to_patch_stride(tile_size)

    # (a) result is a multiple of 224
    assert result % 224 == 0, f"{result} is not a multiple of 224"

    # (b) result ≤ tile_size
    assert result <= tile_size, f"{result} > {tile_size}"

    # (c) tile_size - result < 224
    assert tile_size - result < 224, f"gap {tile_size - result} >= 224"


# Feature: tiled-inference, Property 3: Cosine ramp correctness
@given(
    tile_size=st.integers(min_value=224, max_value=2048).map(lambda x: (x // 224) * 224),
    overlap_size=st.integers(min_value=65, max_value=500),
)
@settings(max_examples=100)
def test_cosine_ramp_correctness(tile_size: int, overlap_size: int) -> None:
    """For any valid tile_size and overlap_size, build_cosine_ramp_2d produces a
    mask where:
    (a) all values lie in [0.0, 1.0],
    (b) the core region (pixels more than overlap_size from any edge) has weight
        exactly 1.0,
    (c) the outermost pixel row/column has weight approaching 0.0,
    (d) for any position i in the overlap region, the weight equals
        0.5 - 0.5*cos(π*(i+0.5)/overlap_size) within floating-point tolerance.

    The output shape must be [1, 1, tile_size, tile_size].

    **Validates: Requirements 3.1, 3.2**
    """
    assume(overlap_size < tile_size // 2)

    mask = build_cosine_ramp_2d(tile_size, overlap_size)

    # Shape check
    assert mask.shape == (1, 1, tile_size, tile_size), (
        f"Expected shape [1, 1, {tile_size}, {tile_size}], got {mask.shape}"
    )

    m = mask[0, 0]  # [tile_size, tile_size]

    # (a) All values in [0.0, 1.0]
    assert m.min() >= 0.0, f"Min value {m.min().item()} < 0.0"
    assert m.max() <= 1.0, f"Max value {m.max().item()} > 1.0"

    # (b) Core region has weight exactly 1.0
    core = m[overlap_size : tile_size - overlap_size, overlap_size : tile_size - overlap_size]
    if core.numel() > 0:
        assert torch.all(core == 1.0), "Core region contains values != 1.0"

    # (c) Outermost pixel row/column has weight approaching 0.0
    edge_1d = 0.5 - 0.5 * math.cos(math.pi * 0.5 / overlap_size)
    assert m[0, 0].item() < 0.05, f"Corner weight {m[0, 0].item()} not approaching 0.0"
    assert m[0, :].max().item() <= edge_1d + 1e-7, "Top row has unexpectedly large values"

    # (d) Overlap region follows cosine formula within floating-point tolerance
    center = tile_size // 2
    for i in range(overlap_size):
        expected = 0.5 - 0.5 * math.cos(math.pi * (i + 0.5) / overlap_size)
        actual = m[center, i].item()
        assert abs(actual - expected) < 1e-6, f"At overlap pos {i}: expected {expected}, got {actual}"
        actual_v = m[i, center].item()
        assert abs(actual_v - expected) < 1e-6, f"At vertical overlap pos {i}: expected {expected}, got {actual_v}"
