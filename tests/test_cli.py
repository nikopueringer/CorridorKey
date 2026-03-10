"""Tests for the typer-based CLI in corridorkey_cli.py."""

from __future__ import annotations

import re
from unittest.mock import patch

from typer.testing import CliRunner

from clip_manager import InferenceSettings
from corridorkey_cli import app

runner = CliRunner()

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Help output
# ---------------------------------------------------------------------------


class TestHelpOutput:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "list-clips" in result.output
        assert "generate-alphas" in result.output
        assert "run-inference" in result.output
        assert "wizard" in result.output

    def test_list_clips_help(self):
        result = runner.invoke(app, ["list-clips", "--help"])
        assert result.exit_code == 0

    def test_generate_alphas_help(self):
        result = runner.invoke(app, ["generate-alphas", "--help"])
        assert result.exit_code == 0

    def test_run_inference_help(self):
        result = runner.invoke(app, ["run-inference", "--help"])
        assert result.exit_code == 0

    def test_wizard_help(self):
        result = runner.invoke(app, ["wizard", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------


class TestInvalidArgs:
    def test_wizard_requires_path(self):
        result = runner.invoke(app, ["wizard"])
        assert result.exit_code != 0

    def test_unknown_subcommand(self):
        result = runner.invoke(app, ["nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# InferenceSettings defaults
# ---------------------------------------------------------------------------


class TestInferenceSettings:
    def test_defaults(self):
        s = InferenceSettings()
        assert s.input_is_linear is False
        assert s.despill_strength == 0.5
        assert s.auto_despeckle is True
        assert s.despeckle_size == 400
        assert s.refiner_scale == 1.0

    def test_custom_values(self):
        s = InferenceSettings(
            input_is_linear=True,
            despill_strength=0.8,
            auto_despeckle=False,
            despeckle_size=200,
            refiner_scale=1.5,
        )
        assert s.input_is_linear is True
        assert s.despill_strength == 0.8
        assert s.auto_despeckle is False
        assert s.despeckle_size == 200
        assert s.refiner_scale == 1.5


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class TestCallbackProtocol:
    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    @patch("corridorkey_cli._prompt_inference_settings")
    def test_run_inference_passes_callbacks(self, mock_prompt, mock_run, mock_scan):
        """run-inference subcommand passes on_clip_start and on_frame_complete."""
        mock_scan.return_value = []
        mock_prompt.return_value = InferenceSettings()

        result = runner.invoke(app, ["run-inference"])
        assert result.exit_code == 0

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert "on_clip_start" in kwargs
        assert "on_frame_complete" in kwargs
        assert callable(kwargs["on_clip_start"])
        assert callable(kwargs["on_frame_complete"])

    def test_callback_signatures(self):
        """Callbacks accept the documented (name, count) / (idx, total) args."""
        from corridorkey_cli import ProgressContext

        ctx = ProgressContext()
        ctx.__enter__()
        try:
            # Should not raise
            ctx.on_clip_start("test_clip", 100)
            ctx.on_frame_complete(0, 100)
            ctx.on_frame_complete(99, 100)
        finally:
            ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# list-clips subcommand
# ---------------------------------------------------------------------------


class TestListClips:
    @patch("corridorkey_cli.scan_clips")
    def test_list_clips_calls_scan(self, mock_scan):
        mock_scan.return_value = []
        result = runner.invoke(app, ["list-clips"])
        assert result.exit_code == 0
        mock_scan.assert_called_once()


# ---------------------------------------------------------------------------
# Non-interactive flags for run-inference
# ---------------------------------------------------------------------------


class TestNonInteractiveFlags:
    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_all_flags_skips_prompts(self, mock_run, mock_scan):
        """When all settings flags are provided, no interactive prompts fire."""
        mock_scan.return_value = []

        result = runner.invoke(
            app,
            [
                "run-inference",
                "--linear",
                "--despill",
                "7",
                "--despeckle",
                "--despeckle-size",
                "200",
                "--refiner",
                "1.5",
            ],
        )
        assert result.exit_code == 0

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        settings = kwargs["settings"]
        assert settings.input_is_linear is True
        assert settings.despill_strength == 0.7
        assert settings.auto_despeckle is True
        assert settings.despeckle_size == 200
        assert settings.refiner_scale == 1.5

    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_srgb_flag(self, mock_run, mock_scan):
        """--srgb sets input_is_linear=False."""
        mock_scan.return_value = []

        result = runner.invoke(
            app,
            [
                "run-inference",
                "--srgb",
                "--despill",
                "5",
                "--no-despeckle",
                "--refiner",
                "1.0",
            ],
        )
        assert result.exit_code == 0

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        settings = kwargs["settings"]
        assert settings.input_is_linear is False
        assert settings.auto_despeckle is False

    @patch("corridorkey_cli.scan_clips")
    @patch("corridorkey_cli.run_inference")
    def test_despill_clamped_to_range(self, mock_run, mock_scan):
        """Despill values outside 0-10 are clamped."""
        mock_scan.return_value = []

        result = runner.invoke(
            app,
            [
                "run-inference",
                "--srgb",
                "--despill",
                "15",
                "--despeckle",
                "--refiner",
                "1.0",
            ],
        )
        assert result.exit_code == 0

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        settings = kwargs["settings"]
        assert settings.despill_strength == 1.0  # clamped 15→10, then /10

    def test_run_inference_help_shows_flags(self):
        """run-inference --help lists the settings flags."""
        result = runner.invoke(app, ["run-inference", "--help"])
        assert result.exit_code == 0
        plain = ANSI_ESCAPE.sub("", result.output)
        assert "--despill" in plain
        assert "--linear" in plain
        assert "--refiner" in plain
        assert "--despeckle-size" in plain
