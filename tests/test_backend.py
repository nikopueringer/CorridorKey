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
    HF_CHECKPOINT_FILENAME_SAFETENSORS,
    HF_REPO_ID,
    MLX_EXT,
    SAFETENSORS_EXT,
    TORCH_EXT,
    _discover_checkpoint,
    _ensure_torch_checkpoint,
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
        """Empty dir + TORCH_EXT calls _ensure_torch_checkpoint — primary path is safetensors."""
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
                # Simulate hf_hub_download returning a cached safetensors file
                cached = tmp_path / "hf_cache" / HF_CHECKPOINT_FILENAME_SAFETENSORS
                cached.parent.mkdir()
                cached.write_bytes(b"fake-checkpoint")
                mock_dl.return_value = str(cached)

                result = _discover_checkpoint(TORCH_EXT)
                assert result.name == HF_CHECKPOINT_FILENAME_SAFETENSORS
                assert result.exists()
                mock_dl.assert_called_once()
                # Must have requested the safetensors filename
                _, kwargs = mock_dl.call_args
                assert kwargs["filename"] == HF_CHECKPOINT_FILENAME_SAFETENSORS

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

    def test_torch_prefers_safetensors_over_pth(self, tmp_path):
        """When both .safetensors and .pth exist, Torch discovery picks safetensors."""
        safetensors_ckpt = tmp_path / "model.safetensors"
        pth_ckpt = tmp_path / "model.pth"
        safetensors_ckpt.touch()
        pth_ckpt.touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            result = _discover_checkpoint(TORCH_EXT)
            assert result == safetensors_ckpt

    def test_torch_falls_back_to_pth_when_only_pth_present(self, tmp_path):
        """Legacy installs with only .pth cached: Torch discovery returns it, no download."""
        pth_ckpt = tmp_path / "legacy.pth"
        pth_ckpt.touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
                result = _discover_checkpoint(TORCH_EXT)
                assert result == pth_ckpt
                mock_dl.assert_not_called()

    def test_torch_finds_safetensors_when_only_safetensors_present(self, tmp_path):
        """New installs with only .safetensors: Torch discovery returns it, no download."""
        sft_ckpt = tmp_path / "model.safetensors"
        sft_ckpt.touch()
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
                result = _discover_checkpoint(TORCH_EXT)
                assert result == sft_ckpt
                mock_dl.assert_not_called()

    def test_ensure_torch_checkpoint_falls_back_to_pth_on_entry_not_found(self, tmp_path):
        """When HF does not yet host the .safetensors, _ensure_torch_checkpoint downloads the .pth."""
        from huggingface_hub.utils import EntryNotFoundError

        cached_pth = tmp_path / "hf_cache" / HF_CHECKPOINT_FILENAME
        cached_pth.parent.mkdir()
        cached_pth.write_bytes(b"legacy-pth-bytes")

        def hf_side_effect(*, repo_id, filename):
            assert repo_id == HF_REPO_ID
            if filename == HF_CHECKPOINT_FILENAME_SAFETENSORS:
                raise EntryNotFoundError("not uploaded yet")
            if filename == HF_CHECKPOINT_FILENAME:
                return str(cached_pth)
            raise AssertionError(f"unexpected filename: {filename}")

        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download", side_effect=hf_side_effect) as mock_dl:
                result = _ensure_torch_checkpoint()

                # Fallback path: landed as .pth in the checkpoint dir
                assert result == tmp_path / HF_CHECKPOINT_FILENAME
                assert result.read_bytes() == b"legacy-pth-bytes"
                # Both filenames must have been attempted (safetensors first, then pth)
                assert mock_dl.call_count == 2
                assert mock_dl.call_args_list[0].kwargs["filename"] == HF_CHECKPOINT_FILENAME_SAFETENSORS
                assert mock_dl.call_args_list[1].kwargs["filename"] == HF_CHECKPOINT_FILENAME

    def test_ensure_torch_checkpoint_network_error_does_not_fall_back_to_pth(self, tmp_path):
        """Regression guard: a generic network error during .safetensors download must surface as a
        RuntimeError with an actionable message — it must NOT silently fall back to downloading the
        legacy .pth. Only an HF-specific EntryNotFoundError (file missing from the repo) triggers
        the fallback; everything else (transient 5xx, DNS, timeout, disk, auth) propagates.

        Without this test, a future refactor could widen the `except` and degrade users to the
        less-safe .pth format whenever the network hiccups.
        """
        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=ConnectionError("DNS resolution failed"),
            ) as mock_dl:
                with pytest.raises(RuntimeError, match=r"huggingface\.co"):
                    _ensure_torch_checkpoint()

                # The .pth fallback download must NOT have been attempted.
                assert mock_dl.call_count == 1
                assert mock_dl.call_args.kwargs["filename"] == HF_CHECKPOINT_FILENAME_SAFETENSORS

    def test_discover_uses_safetensors_constant(self):
        """SAFETENSORS_EXT must equal MLX_EXT — both point at the .safetensors format."""
        assert SAFETENSORS_EXT == ".safetensors"
        assert SAFETENSORS_EXT == MLX_EXT

    def test_ensure_torch_checkpoint_happy_path(self, tmp_path):
        """Mock hf_hub_download, verify copy to CHECKPOINT_DIR/CorridorKey.safetensors."""
        cached = tmp_path / "hf_cache" / HF_CHECKPOINT_FILENAME_SAFETENSORS
        cached.parent.mkdir()
        cached.write_bytes(b"fake-checkpoint-data")

        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as mock_dl:
                result = _ensure_torch_checkpoint()

                assert result == tmp_path / HF_CHECKPOINT_FILENAME_SAFETENSORS
                assert result.exists()
                assert result.read_bytes() == b"fake-checkpoint-data"
                mock_dl.assert_called_once_with(
                    repo_id=HF_REPO_ID,
                    filename=HF_CHECKPOINT_FILENAME_SAFETENSORS,
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
        cached = tmp_path / "hf_cache" / HF_CHECKPOINT_FILENAME_SAFETENSORS
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
        cached = tmp_path / "hf_cache" / HF_CHECKPOINT_FILENAME_SAFETENSORS
        cached.parent.mkdir()
        cached.write_bytes(b"data")

        with mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)):
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
                with caplog.at_level(logging.INFO, logger="CorridorKeyModule.backend"):
                    _ensure_torch_checkpoint()

        assert any("Downloading" in msg for msg in caplog.messages)
        assert any("saved" in msg.lower() for msg in caplog.messages)


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
