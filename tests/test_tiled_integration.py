"""Integration tests for tiled inference output contract preservation.

# Feature: tiled-inference, Property 9: Output contract preservation
"""

import torch
import torch.nn as nn

from CorridorKeyModule.tiled_inference import TiledInferenceEngine


class MockGreenFormer(nn.Module):
    """Minimal mock model that mimics GreenFormer's forward contract.

    Accepts [1, 4, H, W] input and returns {'alpha': [1, 1, H, W], 'fg': [1, 3, H, W]}
    with values in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.refiner = None
        # A dummy parameter so the model is a valid nn.Module
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, _, H, W = x.shape
        alpha = torch.sigmoid(x[:, :1, :, :])  # [1, 1, H, W], values in [0, 1]
        fg = torch.sigmoid(x[:, 1:, :, :])  # [1, 3, H, W], values in [0, 1]
        return {"alpha": alpha, "fg": fg}


# Feature: tiled-inference, Property 9: Output contract preservation
def test_forward_tiled_output_keys_and_shapes() -> None:
    """Tiled output dict from forward_tiled contains exactly keys {'alpha', 'fg'}
    with shapes matching non-tiled output.

    **Validates: Requirements 8.1, 8.4**
    """
    model = MockGreenFormer()
    device = torch.device("cpu")
    img_size = 2048
    tile_size = 896  # multiple of 224 (4 * 224)
    overlap_size = 128

    engine = TiledInferenceEngine(
        model=model,
        device=device,
        tile_size=tile_size,
        overlap_size=overlap_size,
        img_size=img_size,
    )

    inp = torch.randn(1, 4, img_size, img_size)

    # --- Tiled output ---
    tiled_out = engine.forward_tiled(inp)

    # Output must contain exactly {'alpha', 'fg'}
    assert set(tiled_out.keys()) == {"alpha", "fg"}, f"Expected keys {{'alpha', 'fg'}}, got {set(tiled_out.keys())}"

    # Shape checks
    assert tiled_out["alpha"].shape == (1, 1, img_size, img_size), f"alpha shape mismatch: {tiled_out['alpha'].shape}"
    assert tiled_out["fg"].shape == (1, 3, img_size, img_size), f"fg shape mismatch: {tiled_out['fg'].shape}"

    # --- Non-tiled (direct model) output ---
    with torch.no_grad():
        direct_out = model(inp)

    # Shapes from tiled and non-tiled must match
    assert tiled_out["alpha"].shape == direct_out["alpha"].shape, (
        f"alpha shape mismatch: tiled={tiled_out['alpha'].shape} vs direct={direct_out['alpha'].shape}"
    )
    assert tiled_out["fg"].shape == direct_out["fg"].shape, (
        f"fg shape mismatch: tiled={tiled_out['fg'].shape} vs direct={direct_out['fg'].shape}"
    )


# Feature: tiled-inference, Property 5: Tiled vs non-tiled equivalence (round trip)
def test_tiled_vs_non_tiled_equivalence() -> None:
    """Tiled inference output matches non-tiled output within tolerance.

    For a MockGreenFormer (smooth sigmoid activations), the per-pixel absolute
    difference between tiled and non-tiled outputs should be very small because
    the cosine ramp blending preserves smooth functions well.

    **Validates: Requirements 3.5, 8.4**
    """
    model = MockGreenFormer()
    device = torch.device("cpu")
    # Use img_size = 672 (3 * 224) for speed, tile_size = 448 (2 * 224)
    img_size = 672
    tile_size = 448  # 2 * 224, multiple of Hiera alignment
    overlap_size = 65

    engine = TiledInferenceEngine(
        model=model,
        device=device,
        tile_size=tile_size,
        overlap_size=overlap_size,
        img_size=img_size,
    )

    # Deterministic input for reproducibility
    torch.manual_seed(42)
    inp = torch.randn(1, 4, img_size, img_size)

    # --- Non-tiled (direct model) output ---
    with torch.no_grad():
        direct_out = model(inp)

    # --- Tiled output ---
    tiled_out = engine.forward_tiled(inp)

    # Shape equivalence
    assert tiled_out["alpha"].shape == direct_out["alpha"].shape, (
        f"alpha shape mismatch: tiled={tiled_out['alpha'].shape} vs direct={direct_out['alpha'].shape}"
    )
    assert tiled_out["fg"].shape == direct_out["fg"].shape, (
        f"fg shape mismatch: tiled={tiled_out['fg'].shape} vs direct={direct_out['fg'].shape}"
    )

    # Per-pixel absolute difference < tolerance
    alpha_diff = (tiled_out["alpha"] - direct_out["alpha"]).abs()
    fg_diff = (tiled_out["fg"] - direct_out["fg"]).abs()

    atol = 0.02  # Slightly relaxed for cosine ramp blending numerical differences
    assert alpha_diff.max().item() < atol, f"alpha max abs diff {alpha_diff.max().item():.6f} >= {atol}"
    assert fg_diff.max().item() < atol, f"fg max abs diff {fg_diff.max().item():.6f} >= {atol}"
