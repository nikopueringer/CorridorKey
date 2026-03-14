"""Tests for corridorkey_core.model_transformer.

Instantiates GreenFormer at a small img_size to verify architecture shapes
without requiring a checkpoint or GPU. Marked slow because timm model
creation and a forward pass take a few seconds.
"""

import pytest
import torch
from corridorkey_core.model_transformer import (
    MLP,
    CNNRefinerModule,
    DecoderHead,
    GreenFormer,
    RefinerBlock,
)

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(input_dim=128, embed_dim=64)
        x = torch.randn(2, 10, 128)
        assert mlp(x).shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# DecoderHead
# ---------------------------------------------------------------------------


class TestDecoderHead:
    def _make_features(self, batch: int = 1, spatial: int = 8):
        """Synthetic feature maps matching Hiera Base Plus channel counts."""
        return [
            torch.randn(batch, 112, spatial * 4, spatial * 4),
            torch.randn(batch, 224, spatial * 2, spatial * 2),
            torch.randn(batch, 448, spatial, spatial),
            torch.randn(batch, 896, spatial // 2, spatial // 2),
        ]

    def test_alpha_head_output_shape(self):
        head = DecoderHead(output_dim=1)
        features = self._make_features()
        out = head(features)
        assert out.shape[1] == 1

    def test_fg_head_output_shape(self):
        head = DecoderHead(output_dim=3)
        features = self._make_features()
        out = head(features)
        assert out.shape[1] == 3


# ---------------------------------------------------------------------------
# RefinerBlock
# ---------------------------------------------------------------------------


class TestRefinerBlock:
    def test_output_shape_preserved(self):
        block = RefinerBlock(channels=32, dilation=2)
        x = torch.randn(1, 32, 16, 16)
        assert block(x).shape == x.shape


# ---------------------------------------------------------------------------
# CNNRefinerModule
# ---------------------------------------------------------------------------


class TestCNNRefinerModule:
    def test_output_shape(self):
        refiner = CNNRefinerModule(in_channels=7, hidden_channels=32, out_channels=4)
        img = torch.randn(1, 3, 64, 64)
        coarse = torch.randn(1, 4, 64, 64)
        out = refiner(img, coarse)
        assert out.shape == (1, 4, 64, 64)


# ---------------------------------------------------------------------------
# GreenFormer
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGreenFormer:
    """Full model instantiation and forward pass. Slow due to timm model creation."""

    IMG_SIZE = 64  # smallest valid size to keep the test fast

    def test_instantiation(self):
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        assert model is not None

    def test_forward_output_keys(self):
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert "alpha" in out
        assert "fg" in out

    def test_forward_alpha_shape(self):
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["alpha"].shape == (1, 1, self.IMG_SIZE, self.IMG_SIZE)

    def test_forward_fg_shape(self):
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["fg"].shape == (1, 3, self.IMG_SIZE, self.IMG_SIZE)

    def test_forward_output_range(self):
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.rand(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["alpha"].min() >= 0.0
        assert out["alpha"].max() <= 1.0
        assert out["fg"].min() >= 0.0
        assert out["fg"].max() <= 1.0

    def test_forward_with_refiner(self):
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=True)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["alpha"].shape == (1, 1, self.IMG_SIZE, self.IMG_SIZE)
