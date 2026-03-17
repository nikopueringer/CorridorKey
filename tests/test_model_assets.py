"""Tests for runtime model asset bootstrap/download helpers."""

from __future__ import annotations

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
