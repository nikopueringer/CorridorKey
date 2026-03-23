"""Unit tests for CorridorKeyModule.backend — no GPU/MLX required."""

import errno
import logging
import os
from unittest import mock

import numpy as np
import pytest

from CorridorKeyModule.backend import (
    BACKEND_ENV_VAR,
    HF_CHECKPOINT_FILENAME,
    HF_REPO_ID,
    MLX_EXT,
    MLX_MODEL_FILENAME,
    TORCH_EXT,
    _discover_checkpoint,
    _ensure_torch_checkpoint,
    _migrate_pth_to_safetensors,
    _wrap_mlx_output,
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

    def test_zero_torch_triggers_auto_download(self, tmp_path):
        """Empty dir + TORCH_EXT now calls _ensure_torch_checkpoint (auto-download)."""
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
                # Simulate hf_hub_download returning a cached file
                cached = tmp_path / "hf_cache" / "CorridorKey.pth"
                cached.parent.mkdir()
                cached.write_bytes(b"fake-checkpoint")
                mock_dl.return_value = str(cached)

                result = _discover_checkpoint(TORCH_EXT)
                assert result.name == "CorridorKey.pth"
                assert result.exists()
                mock_dl.assert_called_once()

    def test_zero_torch_download_failure_raises_runtime_error(self, tmp_path):
        """When auto-download fails, RuntimeError is raised with HF URL."""
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=ConnectionError("no network"),
            ):
                with pytest.raises(RuntimeError, match="huggingface.co"):
                    _discover_checkpoint(TORCH_EXT)

    def test_zero_safetensors_with_cross_reference(self, tmp_path):
        """MLX ext with no .safetensors but .pth present gives cross-reference hint."""
        (tmp_path / "model.pth").touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="--backend=torch"):
                _discover_checkpoint(MLX_EXT)

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

    def test_ensure_torch_checkpoint_happy_path(self, tmp_path):
        """Mock hf_hub_download, verify copy to CHECKPOINT_DIR/CorridorKey.pth."""
        cached = tmp_path / "hf_cache" / "CorridorKey.pth"
        cached.parent.mkdir()
        cached.write_bytes(b"fake-checkpoint-data")

        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as mock_dl:
                result = _ensure_torch_checkpoint()

                assert result == tmp_path / HF_CHECKPOINT_FILENAME
                assert result.exists()
                assert result.read_bytes() == b"fake-checkpoint-data"
                mock_dl.assert_called_once_with(
                    repo_id=HF_REPO_ID,
                    filename=HF_CHECKPOINT_FILENAME,
                )

    def test_skip_when_present(self, tmp_path):
        """Existing .pth file means hf_hub_download is never called."""
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"existing")
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
                result = _discover_checkpoint(TORCH_EXT)
                assert result == ckpt
                mock_dl.assert_not_called()

    def test_safetensors_preferred_over_pth(self, tmp_path):
        """When both .safetensors and .pth are present, .safetensors is returned."""
        (tmp_path / "model.pth").write_bytes(b"legacy")
        st = tmp_path / "model.safetensors"
        st.write_bytes(b"converted")
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
                result = _discover_checkpoint(TORCH_EXT)
                assert result == st
                mock_dl.assert_not_called()

    def test_safetensors_torch_found_directly(self, tmp_path):
        """A lone .safetensors (non-MLX) is returned for TORCH_EXT discovery."""
        st = tmp_path / "CorridorKey.safetensors"
        st.write_bytes(b"converted")
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            result = _discover_checkpoint(TORCH_EXT)
            assert result == st

    def test_mlx_safetensors_excluded_from_torch_discovery(self, tmp_path):
        """The MLX checkpoint is not returned when discovering Torch checkpoints."""
        mlx_ckpt = tmp_path / MLX_MODEL_FILENAME
        mlx_ckpt.write_bytes(b"mlx-weights")
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            # No Torch .safetensors → should fall through to .pth, find none, auto-download
            with mock.patch("CorridorKeyModule.backend._ensure_torch_checkpoint") as mock_dl:
                mock_dl.return_value = tmp_path / "CorridorKey.pth"
                _discover_checkpoint(TORCH_EXT)
                mock_dl.assert_called_once()

    def test_multiple_safetensors_torch_raises(self, tmp_path):
        """More than one non-MLX .safetensors raises ValueError."""
        (tmp_path / "a.safetensors").write_bytes(b"x")
        (tmp_path / "b.safetensors").write_bytes(b"y")
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="Multiple"):
                _discover_checkpoint(TORCH_EXT)

    def test_mlx_not_triggered(self, tmp_path):
        """MLX ext with empty dir raises FileNotFoundError, no download attempted."""
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
                with pytest.raises(FileNotFoundError):
                    _discover_checkpoint(MLX_EXT)
                mock_dl.assert_not_called()

    def test_network_error_wrapping(self, tmp_path):
        """ConnectionError from hf_hub_download becomes RuntimeError with HF URL."""
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=ConnectionError("connection refused"),
            ) as mock_dl:
                with pytest.raises(RuntimeError, match=r"huggingface\.co/nikopueringer/CorridorKey_v1\.0"):
                    _ensure_torch_checkpoint()
                mock_dl.assert_called_once()

    def test_disk_space_error(self, tmp_path):
        """OSError ENOSPC from copy2 produces message mentioning ~300 MB."""
        cached = tmp_path / "hf_cache" / "CorridorKey.pth"
        cached.parent.mkdir()
        cached.write_bytes(b"data")

        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
                with mock.patch(
                    "CorridorKeyModule.backend.shutil.copy2",
                    side_effect=OSError(errno.ENOSPC, "No space left on device"),
                ):
                    with pytest.raises(OSError, match="300 MB"):
                        _ensure_torch_checkpoint()

    def test_logging_on_download(self, tmp_path, caplog):
        """Info-level log messages emitted at download start and completion."""
        cached = tmp_path / "hf_cache" / "CorridorKey.pth"
        cached.parent.mkdir()
        cached.write_bytes(b"data")

        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
                with caplog.at_level(logging.INFO, logger="CorridorKeyModule.backend"):
                    _ensure_torch_checkpoint()

        assert any("Downloading" in msg for msg in caplog.messages)
        assert any("saved" in msg.lower() for msg in caplog.messages)


# --- _migrate_pth_to_safetensors ---


class TestMigratePthToSafetensors:
    def test_converts_and_deletes_pth(self, tmp_path):
        """Successful migration saves .safetensors and removes the .pth."""
        import torch
        from safetensors.torch import load_file

        src = tmp_path / "model.pth"
        state_dict = {"weight": torch.zeros(4, 4)}
        torch.save({"state_dict": state_dict}, src)

        dst = _migrate_pth_to_safetensors(src)

        assert dst == src.with_suffix(".safetensors")
        assert dst.exists()
        assert not src.exists()
        loaded = load_file(dst)
        assert "weight" in loaded

    def test_flat_state_dict(self, tmp_path):
        """Handles a .pth that is a plain flat dict (no 'state_dict' wrapper)."""
        import torch

        src = tmp_path / "flat.pth"
        torch.save({"bias": torch.ones(3)}, src)
        dst = _migrate_pth_to_safetensors(src)
        assert dst.exists()

    def test_returns_safetensors_path(self, tmp_path):
        import torch

        src = tmp_path / "ck.pth"
        torch.save({"w": torch.eye(2)}, src)
        result = _migrate_pth_to_safetensors(src)
        assert result.suffix == ".safetensors"


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
