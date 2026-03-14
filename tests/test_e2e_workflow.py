"""End-to-end workflow integration tests for CorridorKey.

These tests exercise the full pipeline from ClipEntry asset discovery through
run_inference output file creation.  The neural network engine is mocked so
no model weights or GPU are required.

Why integration-test run_inference?
  Unit tests cover individual math functions.  This file verifies that the
  orchestration layer (reading frames from disk, calling the engine, writing
  output files to the right directories) works end-to-end on realistic
  directory structures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_result(h: int = 4, w: int = 4) -> dict:
    """Return a minimal but valid process_frame result dict sized to (h, w)."""
    return {
        "alpha": np.full((h, w, 1), 0.8, dtype=np.float32),
        "fg": np.full((h, w, 3), 0.6, dtype=np.float32),
        "comp": np.full((h, w, 3), 0.5, dtype=np.float32),
        "processed": np.full((h, w, 4), 0.4, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# End-to-end: ClipEntry discovery → run_inference → files on disk
# ---------------------------------------------------------------------------


class TestE2EInferenceWorkflow:
    """End-to-end: ClipEntry discovery → run_inference → output files on disk.

    Uses the ``tmp_clip_dir`` fixture (shot_a: 2 frames, shot_b: 1 frame /
    no alpha) and a mocked engine.  Verifies directory creation, frame I/O,
    and file writing without a real engine or checkpoint.
    """

    def test_output_directories_created(self, tmp_clip_dir, monkeypatch):
        """run_inference creates Output/{FG,Matte,Comp,Processed} for each clip."""
        from clip_manager import ClipEntry, run_inference

        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()

        # Supply blank answers to all interactive prompts inside run_inference
        monkeypatch.setattr("builtins.input", lambda prompt="": "")

        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = _fake_result()

        with patch("CorridorKeyModule.backend.create_engine", return_value=mock_engine):
            run_inference([entry], device="cpu")

        out_root = tmp_clip_dir / "shot_a" / "Output"
        assert (out_root / "FG").is_dir()
        assert (out_root / "Matte").is_dir()
        assert (out_root / "Comp").is_dir()
        assert (out_root / "Processed").is_dir()

    def test_output_files_written_per_frame(self, tmp_clip_dir, monkeypatch):
        """run_inference writes exactly one output file per input frame.

        shot_a has 2 input frames and 2 alpha frames, so each output
        subdirectory should contain exactly 2 files after inference.
        """
        from clip_manager import ClipEntry, run_inference

        entry = ClipEntry("shot_a", str(tmp_clip_dir / "shot_a"))
        entry.find_assets()

        monkeypatch.setattr("builtins.input", lambda prompt="": "")

        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = _fake_result()

        with patch("CorridorKeyModule.backend.create_engine", return_value=mock_engine):
            run_inference([entry], device="cpu")

        out_root = tmp_clip_dir / "shot_a" / "Output"
        # shot_a has 2 frames → 2 files per output directory
        assert len(list((out_root / "FG").glob("*.exr"))) == 2
        assert len(list((out_root / "Matte").glob("*.exr"))) == 2
        assert len(list((out_root / "Comp").glob("*.png"))) == 2
        assert len(list((out_root / "Processed").glob("*.exr"))) == 2

    def test_clip_without_alpha_skipped(self, tmp_clip_dir, monkeypatch):
        """Clips missing an alpha asset are silently skipped by run_inference.

        shot_b has Input but an empty AlphaHint, so it has no alpha_asset.
        run_inference should process zero frames and create no Output directory.
        """
        from clip_manager import ClipEntry, run_inference

        entry = ClipEntry("shot_b", str(tmp_clip_dir / "shot_b"))
        entry.find_assets()
        assert entry.alpha_asset is None  # precondition

        monkeypatch.setattr("builtins.input", lambda prompt="": "")

        mock_engine = MagicMock()
        mock_engine.process_frame.return_value = _fake_result()

        with patch("CorridorKeyModule.backend.create_engine", return_value=mock_engine):
            run_inference([entry], device="cpu")

        # No engine calls — clip was filtered out before inference
        mock_engine.process_frame.assert_not_called()
        assert not (tmp_clip_dir / "shot_b" / "Output").exists()
