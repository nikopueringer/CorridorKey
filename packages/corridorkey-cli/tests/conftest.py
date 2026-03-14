"""Pytest configuration for corridorkey-cli tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """CliRunner for invoking CLI commands in-process."""
    return CliRunner()


@pytest.fixture
def clips_dir(tmp_path: Path) -> Path:
    """A temporary directory pre-populated with one READY clip."""
    clip = tmp_path / "shot1"
    clip.mkdir()
    frames = clip / "Frames"
    frames.mkdir()
    alpha = clip / "AlphaHint"
    alpha.mkdir()

    # Write minimal PNG stubs so frame_count > 0
    for i in range(3):
        (frames / f"frame_{i:06d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (alpha / f"frame_{i:06d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """A temporary directory with no clips."""
    return tmp_path


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A temporary app_dir with models subdirectory."""
    app_dir = tmp_path / "config"
    models_dir = app_dir / "models"
    models_dir.mkdir(parents=True)
    return app_dir
