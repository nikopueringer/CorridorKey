"""Tests for backend.service screen-color behaviour.

Focused on the helpers added for the CorridorKeyBlue feature:

  * _resolve_screen_color short-circuits when not "auto" (no disk read).
  * _resolve_screen_color logs the explicit choice and the auto-detected choice.
  * _get_engine reuses the cached engine when the screen color matches and
    swaps + reclaims VRAM when it changes.

CorridorKeyService loads heavy ML models on demand; we instantiate it and
mock the loaders so the tests stay pure-Python and run on CPU.
"""

from __future__ import annotations

import logging
from unittest import mock

import numpy as np
import pytest

from backend.service import CorridorKeyService, InferenceParams

# ---------------------------------------------------------------------------
# InferenceParams validation
# ---------------------------------------------------------------------------


class TestInferenceParamsValidation:
    def test_default_is_auto(self):
        assert InferenceParams().screen_color == "auto"

    def test_accepts_known_colors(self):
        for c in ("auto", "green", "blue"):
            assert InferenceParams(screen_color=c).screen_color == c

    def test_rejects_unknown_color(self):
        with pytest.raises(ValueError, match="screen_color"):
            InferenceParams(screen_color="red")


# ---------------------------------------------------------------------------
# _resolve_screen_color
# ---------------------------------------------------------------------------


class _FakeAsset:
    """Bare-minimum stand-in for backend.clip_state.ClipAsset."""

    def __init__(self, asset_type: str = "sequence", path: str = "/dev/null"):
        self.asset_type = asset_type
        self.path = path
        self.frame_count = 1

    def get_frame_files(self):
        return []


class _FakeClip:
    def __init__(self, name: str = "fake"):
        self.name = name
        self.input_asset = _FakeAsset()
        self.alpha_asset = _FakeAsset()
        self.root_path = "/dev/null"


class TestResolveScreenColorService:
    """The service-layer mirror of clip_manager._resolve_screen_color.

    The contract is the same: explicit choices short-circuit (no peek I/O),
    "auto" peeks the clip and runs estimate_screen_color, and a peek failure
    falls back to green with a logged warning.
    """

    def test_explicit_green_does_not_peek(self, caplog):
        svc = CorridorKeyService()
        clip = _FakeClip()
        with (
            mock.patch.object(svc, "_peek_first_frame_for_color") as peek,
            caplog.at_level(logging.INFO, logger="backend.service"),
        ):
            assert svc._resolve_screen_color("green", clip) == "green"
            peek.assert_not_called()
        assert any("explicitly" in m and "green" in m for m in caplog.messages)

    def test_explicit_blue_does_not_peek(self):
        svc = CorridorKeyService()
        clip = _FakeClip()
        with mock.patch.object(svc, "_peek_first_frame_for_color") as peek:
            assert svc._resolve_screen_color("blue", clip) == "blue"
            peek.assert_not_called()

    def test_auto_with_failed_peek_defaults_to_green(self, caplog):
        svc = CorridorKeyService()
        clip = _FakeClip()
        with (
            mock.patch.object(svc, "_peek_first_frame_for_color", return_value=(None, None)),
            caplog.at_level(logging.WARNING, logger="backend.service"),
        ):
            assert svc._resolve_screen_color("auto", clip) == "green"
        assert any("no sample frame" in m for m in caplog.messages)

    def test_auto_calls_estimate_screen_color(self, caplog):
        svc = CorridorKeyService()
        clip = _FakeClip("blue_clip")
        # Synthesize a blue background scene the estimator will pick up.
        h = w = 32
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[..., 2] = 0.85  # blue dominant
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[10:20, 10:20] = 1.0  # subject

        with (
            mock.patch.object(svc, "_peek_first_frame_for_color", return_value=(img, alpha)),
            caplog.at_level(logging.INFO, logger="backend.service"),
        ):
            assert svc._resolve_screen_color("auto", clip) == "blue"
        assert any("auto-detected" in m and "blue_clip" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# _get_engine cache invalidation
# ---------------------------------------------------------------------------


class TestGetEngineColorCache:
    """Engine cache must:

    * reuse the engine when the requested color matches the cached one
    * unload + reload (and reset _engine_screen_color) when the color changes
    """

    def test_reuses_engine_on_same_color(self):
        svc = CorridorKeyService()
        sentinel = object()
        svc._engine = sentinel
        svc._engine_screen_color = "green"

        with mock.patch.object(svc, "_ensure_model"):
            got = svc._get_engine(screen_color="green")
        assert got is sentinel

    def test_swaps_engine_when_color_changes(self):
        """Asking for 'blue' while a 'green' engine is loaded must offload, GC, reload."""
        svc = CorridorKeyService()
        old_engine = mock.MagicMock(name="green_engine")
        svc._engine = old_engine
        svc._engine_screen_color = "green"

        # Patch the import target so we don't drag in CorridorKeyEngine + a real checkpoint.
        new_engine = mock.MagicMock(name="blue_engine")
        with (
            mock.patch.object(svc, "_ensure_model"),
            mock.patch.object(svc, "_safe_offload") as off,
            mock.patch("CorridorKeyModule.backend._discover_checkpoint", return_value="/tmp/blue.safetensors"),
            mock.patch(
                "CorridorKeyModule.inference_engine.CorridorKeyEngine",
                return_value=new_engine,
            ),
        ):
            got = svc._get_engine(screen_color="blue")

        assert got is new_engine
        assert svc._engine_screen_color == "blue"
        off.assert_called_once_with(old_engine)
