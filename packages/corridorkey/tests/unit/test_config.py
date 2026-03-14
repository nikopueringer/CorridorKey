"""Unit tests for config.py."""

from __future__ import annotations

from pathlib import Path

from corridorkey.config import CorridorKeyConfig, export_config


def _config(tmp_path: Path, **kwargs) -> CorridorKeyConfig:
    """Build a CorridorKeyConfig with tmp_path-based dirs that exist on disk."""
    app_dir = tmp_path / "app"
    checkpoint_dir = tmp_path / "models"
    app_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return CorridorKeyConfig(app_dir=app_dir, checkpoint_dir=checkpoint_dir, **kwargs)


class TestCorridorKeyConfig:
    def test_defaults(self):
        config = CorridorKeyConfig()
        assert config.device == "auto"
        assert config.despill_strength == 1.0
        assert config.auto_despeckle is True
        assert config.despeckle_size == 400
        assert config.refiner_scale == 1.0
        assert config.input_is_linear is False
        assert config.fg_format == "exr"
        assert config.matte_format == "exr"
        assert config.comp_format == "png"
        assert config.processed_format == "png"

    def test_path_fields_are_path_objects(self):
        config = CorridorKeyConfig()
        assert isinstance(config.app_dir, Path)
        assert isinstance(config.checkpoint_dir, Path)

    def test_override_device(self, tmp_path: Path):
        config = _config(tmp_path, device="cpu")
        assert config.device == "cpu"

    def test_override_despill_strength(self, tmp_path: Path):
        config = _config(tmp_path, despill_strength=0.5)
        assert config.despill_strength == 0.5

    def test_path_fields_resolve_to_provided_dirs(self, tmp_path: Path):
        config = _config(tmp_path)
        assert config.app_dir == tmp_path / "app"
        assert config.checkpoint_dir == tmp_path / "models"


class TestExportConfig:
    def test_writes_toml_file(self, tmp_path: Path):
        config = _config(tmp_path)
        dest = export_config(config, path=tmp_path / "out.toml")
        assert dest.exists()
        content = dest.read_text()
        assert "device" in content
        assert "despill_strength" in content

    def test_default_path_is_inside_app_dir(self, tmp_path: Path):
        config = _config(tmp_path)
        dest = export_config(config)
        assert dest.parent == config.app_dir

    def test_bool_written_as_lowercase(self, tmp_path: Path):
        config = _config(tmp_path, auto_despeckle=True, input_is_linear=False)
        dest = export_config(config, path=tmp_path / "out.toml")
        content = dest.read_text()
        assert "true" in content
        assert "false" in content
        assert "True" not in content
        assert "False" not in content

    def test_string_values_quoted(self, tmp_path: Path):
        config = _config(tmp_path, device="cpu")
        dest = export_config(config, path=tmp_path / "out.toml")
        content = dest.read_text()
        assert 'device = "cpu"' in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        config = _config(tmp_path)
        dest = export_config(config, path=tmp_path / "nested" / "deep" / "out.toml")
        assert dest.exists()

    def test_roundtrip_contains_all_fields(self, tmp_path: Path):
        config = _config(tmp_path)
        dest = export_config(config, path=tmp_path / "out.toml")
        content = dest.read_text()
        for field_name in config.__class__.model_fields:
            assert field_name in content
