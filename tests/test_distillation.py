"""
Tests for the torchdistill distillation integration.

All tests run without a GPU and without real model weights.
They validate:
  - GreenFormerSmall can be instantiated and forward-passed with random input.
  - GreenFormerSmall is strictly smaller than GreenFormer (parameter count).
  - MatteTaskLoss and MatteResponseKDLoss compute scalars with the correct shape.
  - GreenScreenMattingDataset raises on a mismatched directory (contract check).
  - The distillation losses are importable from the distillation package.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_io_dict(output: dict) -> dict:
    """Build a minimal torchdistill io_dict from a model output dict."""
    return {".": {"output": output}}


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# GreenFormerSmall architecture
# ---------------------------------------------------------------------------


class TestGreenFormerSmall:
    """Tests for the lightweight student model."""

    @pytest.fixture(scope="class")
    def small_model(self):
        """Instantiate a tiny GreenFormerSmall (CPU, no pretrained weights)."""
        from CorridorKeyModule.core.model_transformer import GreenFormerSmall

        return GreenFormerSmall(img_size=64, use_refiner=True)

    @pytest.fixture(scope="class")
    def teacher_model(self):
        """Instantiate a GreenFormer teacher for comparison (CPU)."""
        from CorridorKeyModule.core.model_transformer import GreenFormer

        return GreenFormer(img_size=64, use_refiner=True)

    def test_instantiation(self, small_model):
        """Model should be instantiated without error."""
        assert small_model is not None

    def test_output_keys(self, small_model):
        """Forward pass must return a dict with 'alpha' and 'fg' keys."""
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out = small_model(x)
        assert set(out.keys()) == {"alpha", "fg"}

    def test_output_shapes(self, small_model):
        """Output tensors must match the input spatial resolution."""
        B, H, W = 2, 64, 64
        x = torch.randn(B, 4, H, W)
        with torch.no_grad():
            out = small_model(x)
        assert out["alpha"].shape == (B, 1, H, W), f"alpha shape mismatch: {out['alpha'].shape}"
        assert out["fg"].shape == (B, 3, H, W), f"fg shape mismatch: {out['fg'].shape}"

    def test_output_range(self, small_model):
        """Sigmoid activation must keep values in [0, 1]."""
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out = small_model(x)
        assert out["alpha"].min() >= 0.0
        assert out["alpha"].max() <= 1.0
        assert out["fg"].min() >= 0.0
        assert out["fg"].max() <= 1.0

    def test_smaller_than_teacher(self, small_model, teacher_model):
        """GreenFormerSmall must have fewer parameters than GreenFormer."""
        n_student = _count_params(small_model)
        n_teacher = _count_params(teacher_model)
        assert n_student < n_teacher, (
            f"Student ({n_student:,}) should be smaller than teacher ({n_teacher:,})"
        )

    def test_no_refiner_mode(self):
        """use_refiner=False should disable the refiner without error."""
        from CorridorKeyModule.core.model_transformer import GreenFormerSmall

        model = GreenFormerSmall(img_size=64, use_refiner=False)
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["alpha"].shape == (1, 1, 64, 64)

    def test_gradients_flow(self, small_model):
        """A backward pass must not raise and must produce non-zero gradients."""
        x = torch.randn(1, 4, 64, 64)
        out = small_model(x)
        loss = out["alpha"].mean() + out["fg"].mean()
        loss.backward()
        grad_found = any(p.grad is not None and p.grad.abs().sum() > 0 for p in small_model.parameters())
        assert grad_found, "No non-zero gradients found after backward pass"


# ---------------------------------------------------------------------------
# Custom torchdistill losses
# ---------------------------------------------------------------------------


class TestMatteTaskLoss:
    @pytest.fixture
    def loss_fn(self):
        from distillation.losses import MatteTaskLoss

        return MatteTaskLoss(alpha_weight=1.0, fg_weight=1.0)

    def test_returns_scalar(self, loss_fn):
        B, H, W = 2, 32, 32
        student_out = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        # MatteTaskLoss ignores teacher io_dict; pass a distinct placeholder to match the interface
        teacher_placeholder = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        targets = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        loss = loss_fn(_make_io_dict(student_out), _make_io_dict(teacher_placeholder), targets)
        assert loss.ndim == 0, "Loss must be a scalar"

    def test_zero_loss_on_identical_inputs(self, loss_fn):
        B, H, W = 1, 16, 16
        pred = {"alpha": torch.ones(B, 1, H, W) * 0.5, "fg": torch.ones(B, 3, H, W) * 0.5}
        targets = {"alpha": torch.ones(B, 1, H, W) * 0.5, "fg": torch.ones(B, 3, H, W) * 0.5}
        # Teacher dict is unused by MatteTaskLoss; pass a distinct placeholder
        teacher_placeholder = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        loss = loss_fn(_make_io_dict(pred), _make_io_dict(teacher_placeholder), targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_on_different_inputs(self, loss_fn):
        B, H, W = 1, 16, 16
        pred = {"alpha": torch.zeros(B, 1, H, W), "fg": torch.zeros(B, 3, H, W)}
        targets = {"alpha": torch.ones(B, 1, H, W), "fg": torch.ones(B, 3, H, W)}
        teacher_placeholder = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        loss = loss_fn(_make_io_dict(pred), _make_io_dict(teacher_placeholder), targets)
        assert loss.item() > 0


class TestMatteResponseKDLoss:
    @pytest.fixture
    def loss_fn(self):
        from distillation.losses import MatteResponseKDLoss

        return MatteResponseKDLoss(alpha_weight=0.5, fg_weight=0.5)

    def test_returns_scalar(self, loss_fn):
        B, H, W = 2, 32, 32
        s_out = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        t_out = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        targets = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        loss = loss_fn(_make_io_dict(s_out), _make_io_dict(t_out), targets)
        assert loss.ndim == 0

    def test_zero_loss_when_student_matches_teacher(self, loss_fn):
        B, H, W = 1, 16, 16
        shared = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        targets = {"alpha": torch.rand(B, 1, H, W), "fg": torch.rand(B, 3, H, W)}
        loss = loss_fn(_make_io_dict(shared), _make_io_dict(shared), targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Loss registry integration
# ---------------------------------------------------------------------------


class TestLossRegistry:
    """Check that our custom losses are registered in torchdistill's MIDDLE_LEVEL_LOSS_DICT."""

    def test_task_loss_registered(self):
        from torchdistill.losses.registry import MIDDLE_LEVEL_LOSS_DICT

        import distillation.losses  # noqa: F401

        assert "MatteTaskLoss" in MIDDLE_LEVEL_LOSS_DICT

    def test_kd_loss_registered(self):
        from torchdistill.losses.registry import MIDDLE_LEVEL_LOSS_DICT

        import distillation.losses  # noqa: F401

        assert "MatteResponseKDLoss" in MIDDLE_LEVEL_LOSS_DICT


# ---------------------------------------------------------------------------
# GreenScreenMattingDataset contract
# ---------------------------------------------------------------------------


class TestGreenScreenMattingDataset:
    """Smoke-test the dataset on a synthetically generated tiny split."""

    @pytest.fixture
    def synthetic_split(self, tmp_path):
        """Create a minimal dataset split with 3 samples."""
        import cv2
        import numpy as np

        split = tmp_path / "train"
        for sub in ("input", "alpha_hint", "alpha_gt", "fg_gt"):
            (split / sub).mkdir(parents=True)

        rng = np.random.default_rng(0)
        for i in range(3):
            frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            mask = rng.integers(0, 256, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(split / "input" / f"{i:04d}.png"), frame)
            cv2.imwrite(str(split / "alpha_hint" / f"{i:04d}.png"), mask)
            cv2.imwrite(str(split / "alpha_gt" / f"{i:04d}.png"), mask)
            cv2.imwrite(str(split / "fg_gt" / f"{i:04d}.png"), frame)

        return split

    def test_len(self, synthetic_split):
        from distillation.dataset import GreenScreenMattingDataset

        ds = GreenScreenMattingDataset(synthetic_split, img_size=32)
        assert len(ds) == 3

    def test_sample_shape(self, synthetic_split):
        from distillation.dataset import GreenScreenMattingDataset

        ds = GreenScreenMattingDataset(synthetic_split, img_size=32)
        sample, target = ds[0]
        assert sample.shape == (4, 32, 32)
        assert target["alpha"].shape == (1, 32, 32)
        assert target["fg"].shape == (3, 32, 32)

    def test_alpha_range(self, synthetic_split):
        from distillation.dataset import GreenScreenMattingDataset

        ds = GreenScreenMattingDataset(synthetic_split, img_size=32)
        _, target = ds[0]
        assert target["alpha"].min() >= 0.0
        assert target["alpha"].max() <= 1.0

    def test_missing_fg_gt(self, tmp_path):
        """Dataset should work when fg_gt/ is absent, returning ones."""
        import cv2
        import numpy as np

        split = tmp_path / "train"
        for sub in ("input", "alpha_hint", "alpha_gt"):
            (split / sub).mkdir(parents=True)

        rng = np.random.default_rng(1)
        frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        cv2.imwrite(str(split / "input" / "0000.png"), frame)
        cv2.imwrite(str(split / "alpha_hint" / "0000.png"), mask)
        cv2.imwrite(str(split / "alpha_gt" / "0000.png"), mask)

        from distillation.dataset import GreenScreenMattingDataset

        ds = GreenScreenMattingDataset(split, img_size=16)
        _, target = ds[0]
        assert target["fg"].shape == (3, 16, 16)
        assert (target["fg"] == 1.0).all()

    def test_mismatch_raises(self, tmp_path):
        """ValueError should be raised when input and alpha_hint counts differ."""
        import cv2
        import numpy as np

        split = tmp_path / "train"
        for sub in ("input", "alpha_hint", "alpha_gt"):
            (split / sub).mkdir(parents=True)

        rng = np.random.default_rng(2)
        for i in range(3):
            frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(split / "input" / f"{i:04d}.png"), frame)
            mask = rng.integers(0, 256, (32, 32), dtype=np.uint8)
            cv2.imwrite(str(split / "alpha_gt" / f"{i:04d}.png"), mask)

        # Write only 1 alpha_hint (mismatch!)
        mask = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        cv2.imwrite(str(split / "alpha_hint" / "0000.png"), mask)

        from distillation.dataset import GreenScreenMattingDataset

        with pytest.raises(ValueError, match="Mismatch"):
            GreenScreenMattingDataset(split, img_size=16)
