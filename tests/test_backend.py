"""Unit tests for CorridorKeyModule.backend — no GPU/MLX required."""

import os
import sys
from types import ModuleType
from unittest import mock

import numpy as np
import pytest

from CorridorKeyModule.backend import (
    BACKEND_ENV_VAR,
    MLX_EXT,
    TORCH_EXT,
    _discover_checkpoint,
    _wrap_mlx_output,
    create_engine,
    resolve_backend,
)

# --- resolve_backend ---


class TestResolveBackend:
    def test_explicit_torch(self):
        assert resolve_backend("torch") == "torch"

    def test_explicit_mlx_on_non_apple_raises(self):
        with mock.patch("CorridorKeyModule.backend.sys") as mock_sys:
            mock_sys.platform = "linux"
            with pytest.raises(RuntimeError, match="Apple Silicon"):
                resolve_backend("mlx")

    def test_env_var_torch(self):
        with mock.patch.dict(os.environ, {BACKEND_ENV_VAR: "torch"}):
            assert resolve_backend(None) == "torch"
            assert resolve_backend("auto") == "torch"

    def test_auto_non_darwin(self):
        with mock.patch("CorridorKeyModule.backend.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert resolve_backend("auto") == "torch"

    def test_auto_darwin_no_mlx_package(self):
        with (
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
            mock.patch("CorridorKeyModule.backend.platform") as mock_platform,
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"

            # corridorkey_mlx not importable
            import builtins

            real_import = builtins.__import__

            def fail_mlx(name, *args, **kwargs):
                if name == "corridorkey_mlx":
                    raise ImportError
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=fail_mlx):
                assert resolve_backend("auto") == "torch"

    def test_unknown_backend_raises(self):
        with pytest.raises(RuntimeError, match="Unknown backend"):
            resolve_backend("tensorrt")


# --- _discover_checkpoint ---


class TestDiscoverCheckpoint:
    def test_exactly_one(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        ckpt.touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            result = _discover_checkpoint(TORCH_EXT)
            assert result == ckpt

    def test_zero_raises(self, tmp_path):
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="No .pth checkpoint"):
                _discover_checkpoint(TORCH_EXT)

    def test_zero_with_cross_reference(self, tmp_path):
        (tmp_path / "model.safetensors").touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="--backend=mlx"):
                _discover_checkpoint(TORCH_EXT)

    def test_multiple_raises(self, tmp_path):
        (tmp_path / "a.pth").touch()
        (tmp_path / "b.pth").touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="Multiple"):
                _discover_checkpoint(TORCH_EXT)

    def test_safetensors(self, tmp_path):
        ckpt = tmp_path / "model.safetensors"
        ckpt.touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            result = _discover_checkpoint(MLX_EXT)
            assert result == ckpt


class TestCreateEngineBootstrap:
    def test_auto_backend_bootstraps_assets_before_engine_load(self, tmp_path):
        ensured_kwargs = {}

        def fake_ensure(**kwargs):
            ensured_kwargs.update(kwargs)
            (tmp_path / "model.pth").touch()

        inference_mod = ModuleType("CorridorKeyModule.inference_engine")

        class DummyEngine:
            def __init__(self, checkpoint_path: str, device: str, img_size: int):
                self.checkpoint_path = checkpoint_path
                self.device = device
                self.img_size = img_size

        inference_mod.CorridorKeyEngine = DummyEngine

        with (
            mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)),
            mock.patch("CorridorKeyModule.backend.ensure_corridorkey_assets", side_effect=fake_ensure),
            mock.patch("CorridorKeyModule.backend.resolve_backend", return_value="torch"),
            mock.patch.dict(sys.modules, {"CorridorKeyModule.inference_engine": inference_mod}),
        ):
            engine = create_engine(backend="auto", device="cpu", img_size=1024)

        assert ensured_kwargs == {
            "ensure_torch": True,
            "ensure_mlx": False,
            "download_mlx_if_available": True,
            "checkpoint_dir": str(tmp_path),
        }
        assert engine.checkpoint_path == str(tmp_path / "model.pth")
        assert engine.device == "cpu"
        assert engine.img_size == 1024


# --- _wrap_mlx_output ---


class TestWrapMlxOutput:
    @pytest.fixture
    def mlx_raw_output(self):
        """Simulated MLX engine output: uint8."""
        h, w = 64, 64
        rng = np.random.default_rng(42)
        return {
            "alpha": rng.integers(0, 256, (h, w), dtype=np.uint8),
            "fg": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
            "comp": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
            "processed": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        }

    def test_output_keys(self, mlx_raw_output):
        result = _wrap_mlx_output(mlx_raw_output, despill_strength=1.0, auto_despeckle=True, despeckle_size=400)
        assert set(result.keys()) == {"alpha", "fg", "comp", "processed"}

    def test_alpha_shape_dtype(self, mlx_raw_output):
        result = _wrap_mlx_output(mlx_raw_output, despill_strength=1.0, auto_despeckle=False, despeckle_size=400)
        assert result["alpha"].shape == (64, 64, 1)
        assert result["alpha"].dtype == np.float32
        assert result["alpha"].min() >= 0.0
        assert result["alpha"].max() <= 1.0

    def test_fg_shape_dtype(self, mlx_raw_output):
        result = _wrap_mlx_output(mlx_raw_output, despill_strength=0.0, auto_despeckle=False, despeckle_size=400)
        assert result["fg"].shape == (64, 64, 3)
        assert result["fg"].dtype == np.float32

    def test_processed_shape_dtype(self, mlx_raw_output):
        result = _wrap_mlx_output(mlx_raw_output, despill_strength=1.0, auto_despeckle=False, despeckle_size=400)
        assert result["processed"].shape == (64, 64, 4)
        assert result["processed"].dtype == np.float32

    def test_comp_shape_dtype(self, mlx_raw_output):
        result = _wrap_mlx_output(mlx_raw_output, despill_strength=1.0, auto_despeckle=False, despeckle_size=400)
        assert result["comp"].shape == (64, 64, 3)
        assert result["comp"].dtype == np.float32

    def test_value_ranges(self, mlx_raw_output):
        result = _wrap_mlx_output(mlx_raw_output, despill_strength=1.0, auto_despeckle=False, despeckle_size=400)
        # alpha and fg come from uint8 / 255 so strictly 0-1
        for key in ("alpha", "fg"):
            assert result[key].min() >= 0.0, f"{key} has negative values"
            assert result[key].max() <= 1.0, f"{key} exceeds 1.0"
        # comp/processed can slightly exceed 1.0 due to sRGB conversion + despill redistribution
        # (same behavior as Torch engine — linear_to_srgb doesn't clamp)
        for key in ("comp", "processed"):
            assert result[key].min() >= 0.0, f"{key} has negative values"
