"""Unit tests for CorridorKeyModule.backend — no GPU/MLX required."""

import errno
import logging
import os
import sys
from types import ModuleType
from unittest import mock

import numpy as np
import pytest

from CorridorKeyModule.backend import (
    BACKEND_ENV_VAR,
    HF_CHECKPOINT_FILENAME,
    HF_REPO_ID,
    MLX_EXT,
    TORCH_EXT,
    _discover_checkpoint,
    _ensure_mlx_checkpoint,
    _install_commands,
    _ensure_torch_checkpoint,
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

    def test_auto_darwin_no_mlx_package_and_install_fails(self):
        with (
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
            mock.patch("CorridorKeyModule.backend.platform") as mock_platform,
            mock.patch("CorridorKeyModule.backend._can_import_mlx_runtime", return_value=False),
            mock.patch("CorridorKeyModule.backend._install_mlx_runtime", return_value=False) as mock_install,
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"

            assert resolve_backend("auto") == "torch"
            mock_install.assert_called_once_with()

    def test_auto_darwin_missing_mlx_package_installs_and_prefers_mlx(self):
        with (
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
            mock.patch("CorridorKeyModule.backend.platform") as mock_platform,
            mock.patch("CorridorKeyModule.backend._can_import_mlx_runtime", return_value=False),
            mock.patch("CorridorKeyModule.backend._install_mlx_runtime", return_value=True) as mock_install,
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"

            assert resolve_backend("auto") == "mlx"
            mock_install.assert_called_once_with()

    def test_auto_darwin_with_mlx_package_prefers_mlx(self):
        fake_mlx = ModuleType("corridorkey_mlx")
        with (
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
            mock.patch("CorridorKeyModule.backend.platform") as mock_platform,
            mock.patch.dict(sys.modules, {"corridorkey_mlx": fake_mlx}, clear=False),
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"

            assert resolve_backend("auto") == "mlx"

    def test_explicit_mlx_attempts_install_before_raising(self):
        with (
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
            mock.patch("CorridorKeyModule.backend.platform") as mock_platform,
            mock.patch("CorridorKeyModule.backend._can_import_mlx_runtime", return_value=False),
            mock.patch("CorridorKeyModule.backend._install_mlx_runtime", return_value=False) as mock_install,
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"
            mock_sys.executable = "/tmp/fake-python"

            with pytest.raises(RuntimeError, match="automatic installation failed"):
                resolve_backend("mlx")

            mock_install.assert_called_once_with()

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


class TestInstallCommands:
    def test_prefers_uv_for_active_interpreter(self):
        with (
            mock.patch("CorridorKeyModule.backend.shutil.which", return_value="/opt/homebrew/bin/uv"),
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
        ):
            mock_sys.executable = "/tmp/project/.venv/bin/python"

            assert _install_commands() == [
                [
                    "/opt/homebrew/bin/uv",
                    "pip",
                    "install",
                    "--python",
                    "/tmp/project/.venv/bin/python",
                    "corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git",
                ],
                [
                    "/tmp/project/.venv/bin/python",
                    "-m",
                    "pip",
                    "install",
                    "corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git",
                ],
            ]

    def test_falls_back_to_pip_when_uv_missing(self):
        with (
            mock.patch("CorridorKeyModule.backend.shutil.which", return_value=None),
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
        ):
            mock_sys.executable = "/tmp/project/.venv/bin/python"

            assert _install_commands() == [
                [
                    "/tmp/project/.venv/bin/python",
                    "-m",
                    "pip",
                    "install",
                    "corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git",
                ]
            ]


class TestEnsureMlxCheckpoint:
    def test_returns_existing_mlx_checkpoint_without_conversion(self, tmp_path):
        ckpt = tmp_path / "existing.safetensors"
        ckpt.touch()

        with (
            mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)),
            mock.patch("CorridorKeyModule.backend._convert_torch_checkpoint_to_mlx") as mock_convert,
        ):
            result = _ensure_mlx_checkpoint()

        assert result == ckpt
        mock_convert.assert_not_called()

    def test_converts_torch_checkpoint_when_mlx_weights_missing(self, tmp_path):
        torch_ckpt = tmp_path / "CorridorKey_v1.0.pth"
        torch_ckpt.touch()

        repo_dir = tmp_path / "corridorkey-mlx"
        output_ckpt = tmp_path / "corridorkey_mlx.safetensors"
        command_calls: list[tuple[list[str], str]] = []

        def fake_run_checked_command(cmd, *, cwd):
            command_calls.append((cmd, str(cwd)))
            if cmd[0] == "/usr/bin/git":
                (repo_dir / "scripts").mkdir(parents=True, exist_ok=True)
                (repo_dir / "scripts" / "convert_weights.py").touch()
            if cmd[:5] == ["/usr/local/bin/uv", "run", "--group", "reference", "python"]:
                output_ckpt.touch()

        with (
            mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)),
            mock.patch("CorridorKeyModule.backend.PROJECT_ROOT", tmp_path),
            mock.patch("CorridorKeyModule.backend.MLX_CONVERTER_REPO_DIR", repo_dir),
            mock.patch("CorridorKeyModule.backend.shutil.which") as mock_which,
            mock.patch("CorridorKeyModule.backend._run_checked_command", side_effect=fake_run_checked_command),
        ):
            mock_which.side_effect = lambda name: {
                "git": "/usr/bin/git",
                "uv": "/usr/local/bin/uv",
            }.get(name)

            result = _ensure_mlx_checkpoint()

        assert result == output_ckpt
        assert command_calls == [
            (
                [
                    "/usr/bin/git",
                    "clone",
                    "https://github.com/nikopueringer/corridorkey-mlx.git",
                    str(repo_dir),
                ],
                str(tmp_path),
            ),
            (
                [
                    "/usr/local/bin/uv",
                    "sync",
                    "--group",
                    "reference",
                ],
                str(repo_dir),
            ),
            (
                [
                    "/usr/local/bin/uv",
                    "run",
                    "--group",
                    "reference",
                    "python",
                    "scripts/convert_weights.py",
                    "--checkpoint",
                    str(torch_ckpt),
                    "--output",
                    str(output_ckpt),
                ],
                str(repo_dir),
            ),
        ]


class TestCreateEngine:
    def test_auto_on_apple_silicon_uses_mlx_engine(self, tmp_path):
        ckpt = tmp_path / "model.safetensors"
        ckpt.touch()

        fake_raw_engine = mock.Mock()
        fake_mlx = ModuleType("corridorkey_mlx")
        fake_mlx.CorridorKeyMLXEngine = mock.Mock(return_value=fake_raw_engine)

        with (
            mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(tmp_path)),
            mock.patch("CorridorKeyModule.backend.sys") as mock_sys,
            mock.patch("CorridorKeyModule.backend.platform") as mock_platform,
            mock.patch.dict(sys.modules, {"corridorkey_mlx": fake_mlx}, clear=False),
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"

            engine = create_engine(backend="auto", device="mps", img_size=1024)

        fake_mlx.CorridorKeyMLXEngine.assert_called_once_with(
            str(ckpt),
            img_size=1024,
            tile_size=512,
            overlap=64,
        )
        assert engine._engine is fake_raw_engine


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
