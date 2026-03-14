"""Tests for corridorkey_core.engine_factory.

Backend resolution and checkpoint discovery are tested without GPU or model
files. Tests that require hardware are marked accordingly.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey_core.engine_factory import (
    BACKEND_ENV_VAR,
    MLX_EXT,
    TORCH_EXT,
    discover_checkpoint,
    resolve_backend,
)

# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------


class TestResolveBackend:
    def test_none_triggers_auto_detect(self):
        # On non-Apple-Silicon CI, auto-detect always returns torch
        with patch("corridorkey_core.engine_factory.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = resolve_backend(None)
        assert result == "torch"

    def test_explicit_torch(self):
        result = resolve_backend("torch")
        assert result == "torch"

    def test_explicit_auto_triggers_auto_detect(self):
        with patch("corridorkey_core.engine_factory.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = resolve_backend("auto")
        assert result == "torch"

    def test_case_insensitive(self):
        result = resolve_backend("TORCH")
        assert result == "torch"

    def test_unknown_backend_raises(self):
        with pytest.raises(RuntimeError, match="Unknown backend"):
            resolve_backend("tpu")

    def test_env_var_torch(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "torch")
        result = resolve_backend(None)
        assert result == "torch"

    def test_explicit_arg_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "mlx")
        # Explicit "torch" should win even if env says mlx
        result = resolve_backend("torch")
        assert result == "torch"

    def test_mlx_on_non_apple_raises(self):
        with (
            patch("corridorkey_core.engine_factory.sys") as mock_sys,
            patch("corridorkey_core.engine_factory.platform") as mock_platform,
        ):
            mock_sys.platform = "linux"
            mock_platform.machine.return_value = "x86_64"
            with pytest.raises(RuntimeError, match="Apple Silicon"):
                resolve_backend("mlx")

    def test_mlx_on_apple_without_package_raises(self):
        with (
            patch("corridorkey_core.engine_factory.sys") as mock_sys,
            patch("corridorkey_core.engine_factory.platform") as mock_platform,
            patch("corridorkey_core.engine_factory._mlx_available", return_value=False),
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"
            with pytest.raises(RuntimeError, match="corridorkey_mlx is not installed"):
                resolve_backend("mlx")

    def test_mlx_on_apple_with_package_returns_mlx(self):
        with (
            patch("corridorkey_core.engine_factory.sys") as mock_sys,
            patch("corridorkey_core.engine_factory.platform") as mock_platform,
            patch("corridorkey_core.engine_factory._mlx_available", return_value=True),
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"
            result = resolve_backend("mlx")
        assert result == "mlx"


# ---------------------------------------------------------------------------
# discover_checkpoint
# ---------------------------------------------------------------------------


class TestDiscoverCheckpoint:
    def test_finds_single_pth(self, tmp_path: Path):
        ckpt = tmp_path / "model.pth"
        ckpt.touch()
        result = discover_checkpoint(tmp_path, TORCH_EXT)
        assert result == ckpt

    def test_finds_single_safetensors(self, tmp_path: Path):
        ckpt = tmp_path / "model.safetensors"
        ckpt.touch()
        result = discover_checkpoint(tmp_path, MLX_EXT)
        assert result == ckpt

    def test_accepts_str_path(self, tmp_path: Path):
        ckpt = tmp_path / "model.pth"
        ckpt.touch()
        result = discover_checkpoint(str(tmp_path), TORCH_EXT)
        assert result == ckpt

    def test_returns_path_object(self, tmp_path: Path):
        (tmp_path / "model.pth").touch()
        result = discover_checkpoint(tmp_path, TORCH_EXT)
        assert isinstance(result, Path)

    def test_no_match_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            discover_checkpoint(tmp_path, TORCH_EXT)

    def test_multiple_matches_raises_value_error(self, tmp_path: Path):
        (tmp_path / "model_a.pth").touch()
        (tmp_path / "model_b.pth").touch()
        with pytest.raises(ValueError, match="Multiple"):
            discover_checkpoint(tmp_path, TORCH_EXT)

    def test_wrong_ext_hint_suggests_correct_backend(self, tmp_path: Path):
        # Only a .safetensors file present, but asking for .pth
        (tmp_path / "model.safetensors").touch()
        with pytest.raises(FileNotFoundError, match="mlx"):
            discover_checkpoint(tmp_path, TORCH_EXT)

    def test_ignores_files_with_different_extension(self, tmp_path: Path):
        (tmp_path / "model.bin").touch()
        (tmp_path / "model.pth").touch()
        result = discover_checkpoint(tmp_path, TORCH_EXT)
        assert result.suffix == TORCH_EXT
