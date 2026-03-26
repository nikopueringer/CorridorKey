"""Tests for runtime model asset bootstrap/download helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

from CorridorKeyModule import model_assets as assets


def _write_text(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestEnsureCorridorKeyAssets:
    def test_empty_checkpoint_dir_downloads_torch_and_mlx_when_available(self, tmp_path):
        torch_downloads: list[Path] = []
        mlx_downloads: list[Path] = []

        def fake_torch_download(checkpoint_dir: Path) -> Path:
            torch_downloads.append(checkpoint_dir)
            return _write_text(checkpoint_dir / assets.CORRIDORKEY_TORCH_FILENAME)

        def fake_mlx_download(checkpoint_dir: Path) -> Path:
            mlx_downloads.append(checkpoint_dir)
            return _write_text(checkpoint_dir / assets.CORRIDORKEY_MLX_FILENAME)

        with (
            mock.patch.object(assets, "_download_corridorkey_torch", side_effect=fake_torch_download),
            mock.patch.object(assets, "_download_corridorkey_mlx", side_effect=fake_mlx_download),
            mock.patch.object(assets, "mlx_runtime_available", return_value=True),
        ):
            assets.ensure_corridorkey_assets(
                checkpoint_dir=tmp_path,
                ensure_torch=True,
                ensure_mlx=False,
                download_mlx_if_available=True,
            )

        assert torch_downloads == [tmp_path]
        assert mlx_downloads == [tmp_path]
        assert (tmp_path / assets.CORRIDORKEY_TORCH_FILENAME).is_file()
        assert (tmp_path / assets.CORRIDORKEY_MLX_FILENAME).is_file()

    def test_existing_torch_checkpoint_skips_opportunistic_mlx_download(self, tmp_path):
        _write_text(tmp_path / "existing_model.pth")

        with (
            mock.patch.object(assets, "_download_corridorkey_torch") as mock_torch,
            mock.patch.object(assets, "_download_corridorkey_mlx") as mock_mlx,
            mock.patch.object(assets, "mlx_runtime_available", return_value=True),
        ):
            assets.ensure_corridorkey_assets(
                checkpoint_dir=tmp_path,
                ensure_torch=True,
                ensure_mlx=False,
                download_mlx_if_available=True,
            )

        mock_torch.assert_not_called()
        mock_mlx.assert_not_called()

    def test_explicit_mlx_request_downloads_mlx_even_when_torch_exists(self, tmp_path):
        _write_text(tmp_path / "existing_model.pth")

        with mock.patch.object(
            assets,
            "_download_corridorkey_mlx",
            side_effect=lambda checkpoint_dir: _write_text(checkpoint_dir / assets.CORRIDORKEY_MLX_FILENAME),
        ) as mock_mlx:
            assets.ensure_corridorkey_assets(
                checkpoint_dir=tmp_path,
                ensure_torch=False,
                ensure_mlx=True,
                download_mlx_if_available=False,
            )

        mock_mlx.assert_called_once_with(tmp_path)
        assert (tmp_path / assets.CORRIDORKEY_MLX_FILENAME).is_file()


class TestDownloadCorridorKeyMlx:
    def test_cli_download_uses_expected_repo_override_and_release_tag(self, tmp_path):
        cached = _write_text(tmp_path / "cache" / assets.CORRIDORKEY_MLX_FILENAME)

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=str(cached),
            stderr="",
        )

        with (
            mock.patch.object(assets, "mlx_runtime_available", return_value=True),
            mock.patch.object(assets, "subprocess") as mock_subprocess,
        ):
            mock_subprocess.run.return_value = completed
            result = assets._download_corridorkey_mlx(tmp_path)

        assert result == tmp_path / assets.CORRIDORKEY_MLX_FILENAME
        assert result.is_file()

        _, kwargs = mock_subprocess.run.call_args
        command = kwargs["args"] if "args" in kwargs else mock_subprocess.run.call_args.args[0]
        env = kwargs["env"]

        assert "--tag" in command
        assert assets.CORRIDORKEY_MLX_TAG in command
        assert env["CORRIDORKEY_MLX_WEIGHTS_REPO"] == assets.CORRIDORKEY_MLX_REPO

    def test_cli_failure_falls_back_to_direct_release_download(self, tmp_path):
        with (
            mock.patch.object(assets, "mlx_runtime_available", return_value=True),
            mock.patch.object(
                assets.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["corridorkey_mlx"]),
            ),
            mock.patch.object(assets.urllib.request, "urlretrieve") as mock_urlretrieve,
        ):
            mock_urlretrieve.side_effect = lambda url, dest: Path(dest).write_bytes(b"mlx-weights")
            result = assets._download_corridorkey_mlx(tmp_path)

        assert result == tmp_path / assets.CORRIDORKEY_MLX_FILENAME
        assert result.read_bytes() == b"mlx-weights"
        download_url = mock_urlretrieve.call_args.args[0]
        assert assets.CORRIDORKEY_MLX_REPO in download_url
        assert assets.CORRIDORKEY_MLX_TAG in download_url
        assert download_url.endswith(f"/{assets.CORRIDORKEY_MLX_FILENAME}")


class TestEnsureOptionalStepWeights:
    def test_gvm_weights_download_once_and_reuse(self, tmp_path):
        def fake_snapshot_download(*, local_dir: str, **kwargs):
            weights_dir = Path(local_dir)
            _write_text(weights_dir / "vae" / "diffusion_pytorch_model.safetensors")
            _write_text(weights_dir / "scheduler" / "scheduler_config.json", "{}")
            _write_text(weights_dir / "unet" / "diffusion_pytorch_model.safetensors")
            return str(weights_dir)

        with mock.patch.object(assets, "snapshot_download", side_effect=fake_snapshot_download) as mock_snapshot:
            assets.ensure_gvm_weights(tmp_path)
            assets.ensure_gvm_weights(tmp_path)

        assert mock_snapshot.call_count == 1

    def test_videomama_weights_download_base_and_unet_once(self, tmp_path):
        def fake_snapshot_download(*, repo_id: str, local_dir: str, **kwargs):
            target_dir = Path(local_dir)
            if repo_id == assets.VIDEOMAMA_BASE_REPO_ID:
                _write_text(target_dir / "feature_extractor" / "preprocessor_config.json", "{}")
                _write_text(target_dir / "image_encoder" / "model.safetensors")
                _write_text(target_dir / "vae" / "diffusion_pytorch_model.safetensors")
            elif repo_id == assets.VIDEOMAMA_REPO_ID:
                _write_text(target_dir / "unet" / "diffusion_pytorch_model.safetensors")
            else:
                raise AssertionError(f"Unexpected repo_id: {repo_id}")
            return str(target_dir)

        with mock.patch.object(assets, "snapshot_download", side_effect=fake_snapshot_download) as mock_snapshot:
            assets.ensure_videomama_weights(tmp_path)
            assets.ensure_videomama_weights(tmp_path)

        assert mock_snapshot.call_count == 2
