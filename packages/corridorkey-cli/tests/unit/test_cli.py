"""Unit tests for the CorridorKey CLI.

Tests use Typer's CliRunner to invoke commands in-process without spawning
subprocesses. Heavy operations (inference, model download) are mocked so
tests run without GPU or network access.

Each test class covers one command. Tests verify:
- Exit codes (0 = success, 1 = failure)
- Key strings in stdout/stderr
- That the correct Application Layer functions are called with the right args
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from corridorkey_cli import app
from typer.testing import CliRunner


class TestHelp:
    """Top-level --help must list all commands so users can discover the CLI."""

    def test_help_lists_all_commands(self, runner: CliRunner) -> None:
        """All six commands must appear in the top-level help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("init", "doctor", "wizard", "process", "scan", "config"):
            assert cmd in result.output

    def test_no_args_shows_help(self, runner: CliRunner) -> None:
        """Invoking with no arguments must show help, not crash."""
        result = runner.invoke(app, [])
        # Typer returns exit code 2 when no_args_is_help=True
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output


class TestScan:
    """``corridorkey scan`` - reads clip states without processing."""

    def test_scan_missing_dir_exits_1(self, runner: CliRunner, tmp_path: Path) -> None:
        """A non-existent path must exit 1 with a clear error message."""
        result = runner.invoke(app, ["scan", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_scan_empty_dir_reports_no_clips(self, runner: CliRunner, empty_dir: Path) -> None:
        """An empty directory must exit 0 and report no clips found."""
        result = runner.invoke(app, ["scan", str(empty_dir)])
        assert result.exit_code == 0
        assert "No clips found" in result.output

    def test_scan_shows_clip_table(self, runner: CliRunner, clips_dir: Path) -> None:
        """A directory with clips must show a table containing clip names and states."""
        with patch("corridorkey_cli.commands.scan.CorridorKeyService") as mock_svc_cls:
            mock_svc = MagicMock()
            mock_svc_cls.return_value = mock_svc

            clip = MagicMock()
            clip.name = "shot1"
            clip.state.value = "READY"
            clip.input_asset.frame_count = 10
            clip.alpha_asset.frame_count = 10
            clip.error_message = None
            mock_svc.scan_clips.return_value = [clip]

            result = runner.invoke(app, ["scan", str(clips_dir)])

        assert result.exit_code == 0
        assert "shot1" in result.output
        assert "READY" in result.output

    def test_scan_shows_error_message(self, runner: CliRunner, clips_dir: Path) -> None:
        """ERROR state clips must show their error_message in the table."""
        with patch("corridorkey_cli.commands.scan.CorridorKeyService") as mock_svc_cls:
            mock_svc = MagicMock()
            mock_svc_cls.return_value = mock_svc

            clip = MagicMock()
            clip.name = "shot1"
            clip.state.value = "ERROR"
            clip.input_asset = None
            clip.alpha_asset = None
            clip.error_message = "alpha generation failed"
            mock_svc.scan_clips.return_value = [clip]

            result = runner.invoke(app, ["scan", str(clips_dir)])

        assert result.exit_code == 0
        assert "alpha generation failed" in result.output

    def test_scan_summary_shows_count(self, runner: CliRunner, clips_dir: Path) -> None:
        """The summary line must include the total clip count."""
        with patch("corridorkey_cli.commands.scan.CorridorKeyService") as mock_svc_cls:
            mock_svc = MagicMock()
            mock_svc_cls.return_value = mock_svc

            clips = []
            for name in ("shot1", "shot2"):
                c = MagicMock()
                c.name = name
                c.state.value = "READY"
                c.input_asset.frame_count = 5
                c.alpha_asset.frame_count = 5
                c.error_message = None
                clips.append(c)
            mock_svc.scan_clips.return_value = clips

            result = runner.invoke(app, ["scan", str(clips_dir)])

        assert result.exit_code == 0
        assert "2" in result.output


class TestProcess:
    """``corridorkey process`` - non-interactive batch processing."""

    def test_process_missing_dir_exits_1(self, runner: CliRunner, tmp_path: Path) -> None:
        """A non-existent path must exit 1 with a clear error message."""
        result = runner.invoke(app, ["process", str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_process_calls_process_directory(self, runner: CliRunner, clips_dir: Path) -> None:
        """process must delegate to process_directory with the correct clips_dir."""
        with (
            patch("corridorkey_cli.commands.process.process_directory") as mock_pd,
            patch("corridorkey_cli.commands.process.ProgressContext"),
        ):
            mock_result = MagicMock()
            mock_result.clips = []
            mock_result.succeeded = []
            mock_result.failed = []
            mock_result.skipped = []
            mock_pd.return_value = mock_result

            result = runner.invoke(app, ["process", str(clips_dir)])

        assert result.exit_code == 0
        mock_pd.assert_called_once()
        call_kwargs = mock_pd.call_args.kwargs
        assert call_kwargs["clips_dir"] == str(clips_dir)

    def test_process_passes_device_flag(self, runner: CliRunner, clips_dir: Path) -> None:
        """--device flag must be forwarded to process_directory."""
        with (
            patch("corridorkey_cli.commands.process.process_directory") as mock_pd,
            patch("corridorkey_cli.commands.process.ProgressContext"),
        ):
            mock_result = MagicMock()
            mock_result.clips = []
            mock_result.succeeded = []
            mock_result.failed = []
            mock_result.skipped = []
            mock_pd.return_value = mock_result

            runner.invoke(app, ["process", str(clips_dir), "--device", "cpu"])

        call_kwargs = mock_pd.call_args.kwargs
        assert call_kwargs["device"] == "cpu"

    def test_process_passes_inference_params(self, runner: CliRunner, clips_dir: Path) -> None:
        """Inference param flags must be forwarded as InferenceParams fields."""
        with (
            patch("corridorkey_cli.commands.process.process_directory") as mock_pd,
            patch("corridorkey_cli.commands.process.ProgressContext"),
        ):
            mock_result = MagicMock()
            mock_result.clips = []
            mock_result.succeeded = []
            mock_result.failed = []
            mock_result.skipped = []
            mock_pd.return_value = mock_result

            runner.invoke(
                app,
                [
                    "process",
                    str(clips_dir),
                    "--despill",
                    "0.5",
                    "--no-despeckle",
                    "--refiner",
                    "0.8",
                    "--linear",
                ],
            )

        call_kwargs = mock_pd.call_args.kwargs
        params = call_kwargs["params"]
        assert params.despill_strength == 0.5
        assert params.auto_despeckle is False
        assert params.refiner_scale == 0.8
        assert params.input_is_linear is True

    def test_process_exits_1_on_failures(self, runner: CliRunner, clips_dir: Path) -> None:
        """Exit code must be 1 when any clips failed so CI pipelines catch errors."""
        with (
            patch("corridorkey_cli.commands.process.process_directory") as mock_pd,
            patch("corridorkey_cli.commands.process.ProgressContext"),
        ):
            failed_clip = MagicMock()
            failed_clip.name = "shot1"
            failed_clip.state = "READY"
            failed_clip.error = "inference failed"
            failed_clip.skipped = False
            failed_clip.frames_processed = 0
            failed_clip.frames_total = 10

            mock_result = MagicMock()
            mock_result.clips = [failed_clip]
            mock_result.succeeded = []
            mock_result.failed = [failed_clip]
            mock_result.skipped = []
            mock_pd.return_value = mock_result

            result = runner.invoke(app, ["process", str(clips_dir)])

        assert result.exit_code == 1

    def test_process_exits_0_on_success(self, runner: CliRunner, clips_dir: Path) -> None:
        """Exit code must be 0 when all clips succeed."""
        with (
            patch("corridorkey_cli.commands.process.process_directory") as mock_pd,
            patch("corridorkey_cli.commands.process.ProgressContext"),
        ):
            ok_clip = MagicMock()
            ok_clip.name = "shot1"
            ok_clip.state = "COMPLETE"
            ok_clip.error = None
            ok_clip.skipped = False
            ok_clip.frames_processed = 10
            ok_clip.frames_total = 10

            mock_result = MagicMock()
            mock_result.clips = [ok_clip]
            mock_result.succeeded = [ok_clip]
            mock_result.failed = []
            mock_result.skipped = []
            mock_pd.return_value = mock_result

            result = runner.invoke(app, ["process", str(clips_dir)])

        assert result.exit_code == 0


class TestDoctor:
    """``corridorkey doctor`` - read-only environment health check."""

    def test_doctor_renders_table(self, runner: CliRunner, tmp_path: Path) -> None:
        """doctor must always render a table with check results, never crash."""
        with (
            patch("corridorkey_cli.commands.doctor.check_ffmpeg") as mock_ffmpeg,
            patch("corridorkey_cli.commands.doctor.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.doctor.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.doctor.CorridorKeyService"),
        ):
            mock_ffmpeg.return_value = {
                "available": True,
                "ffmpeg_path": "ffmpeg",
                "ffprobe_path": "ffprobe",
                "version": "6.1",
            }
            cfg = MagicMock()
            cfg.app_dir = tmp_path
            cfg.checkpoint_dir = tmp_path / "models"
            mock_cfg.return_value = cfg
            mock_model.return_value = True

            result = runner.invoke(app, ["doctor"])

        assert "ffmpeg" in result.output
        assert "Python" in result.output

    def test_doctor_exits_1_when_model_missing(self, runner: CliRunner, tmp_path: Path) -> None:
        """Missing inference model must cause exit 1 so init can be triggered."""
        with (
            patch("corridorkey_cli.commands.doctor.check_ffmpeg") as mock_ffmpeg,
            patch("corridorkey_cli.commands.doctor.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.doctor.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.doctor.CorridorKeyService"),
        ):
            mock_ffmpeg.return_value = {
                "available": True,
                "ffmpeg_path": "ffmpeg",
                "ffprobe_path": "ffprobe",
                "version": "6.1",
            }
            cfg = MagicMock()
            cfg.app_dir = tmp_path
            cfg.checkpoint_dir = tmp_path / "models"
            mock_cfg.return_value = cfg
            mock_model.return_value = False

            result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1

    def test_doctor_exits_1_when_ffmpeg_missing(self, runner: CliRunner, tmp_path: Path) -> None:
        """Missing FFmpeg must cause exit 1 since it is required for video extraction."""
        with (
            patch("corridorkey_cli.commands.doctor.check_ffmpeg") as mock_ffmpeg,
            patch("corridorkey_cli.commands.doctor.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.doctor.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.doctor.CorridorKeyService"),
        ):
            mock_ffmpeg.return_value = {
                "available": False,
                "ffmpeg_path": "",
                "ffprobe_path": "",
                "version": "",
            }
            cfg = MagicMock()
            cfg.app_dir = tmp_path
            cfg.checkpoint_dir = tmp_path / "models"
            mock_cfg.return_value = cfg
            mock_model.return_value = True

            result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1

    def test_doctor_exits_0_when_all_pass(self, runner: CliRunner, tmp_path: Path) -> None:
        """All checks passing must produce exit 0 and a success message."""
        with (
            patch("corridorkey_cli.commands.doctor.check_ffmpeg") as mock_ffmpeg,
            patch("corridorkey_cli.commands.doctor.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.doctor.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.doctor.CorridorKeyService"),
            patch("corridorkey_cli.commands.doctor.shutil.which") as mock_which,
        ):
            mock_ffmpeg.return_value = {
                "available": True,
                "ffmpeg_path": "ffmpeg",
                "ffprobe_path": "ffprobe",
                "version": "6.1",
            }
            cfg = MagicMock()
            cfg.app_dir = tmp_path
            cfg.checkpoint_dir = tmp_path / "models"
            mock_cfg.return_value = cfg
            mock_model.return_value = True
            mock_which.return_value = "git"

            result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 0
        assert "passed" in result.output.lower()


class TestConfigCommands:
    """``corridorkey config`` subcommands."""

    def test_config_show_prints_fields(self, runner: CliRunner, config_dir: Path) -> None:
        """config show must print all config field names in a table."""
        with patch("corridorkey_cli.commands.config.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.model_fields = {"device": None, "despill_strength": None}
            cfg.device = "auto"
            cfg.despill_strength = 1.0
            mock_cfg.return_value = cfg

            result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        assert "device" in result.output
        assert "despill_strength" in result.output

    def test_config_init_writes_file(self, runner: CliRunner, config_dir: Path) -> None:
        """config init must call export_config and report the written path."""
        with (
            patch("corridorkey_cli.commands.config.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.config.export_config") as mock_export,
        ):
            mock_cfg.return_value = MagicMock()
            mock_export.return_value = config_dir / "corridorkey.toml"

            result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 0
        mock_export.assert_called_once()
        assert "corridorkey.toml" in result.output


class TestInit:
    """``corridorkey init`` - one-time setup command."""

    def test_init_creates_config_and_reports(self, runner: CliRunner, config_dir: Path) -> None:
        """init must create the config file and print its path."""
        with (
            patch("corridorkey_cli.commands.init.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.init.export_config") as mock_export,
            patch("corridorkey_cli.commands.init.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.doctor.doctor"),
        ):
            cfg = MagicMock()
            cfg.app_dir = config_dir
            cfg.checkpoint_dir = config_dir / "models"
            mock_cfg.return_value = cfg
            mock_export.return_value = config_dir / "corridorkey.toml"
            mock_model.return_value = True

            # Config file does not exist yet
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        mock_export.assert_called_once()

    def test_init_skips_download_when_model_present(self, runner: CliRunner, config_dir: Path) -> None:
        """init must not prompt for download when the model is already present."""
        with (
            patch("corridorkey_cli.commands.init.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.init.export_config"),
            patch("corridorkey_cli.commands.init.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.init.download_model") as mock_dl,
            patch("corridorkey_cli.commands.doctor.doctor"),
        ):
            cfg = MagicMock()
            cfg.app_dir = config_dir
            cfg.checkpoint_dir = config_dir / "models"
            mock_cfg.return_value = cfg
            mock_model.return_value = True

            runner.invoke(app, ["init"])

        mock_dl.assert_not_called()

    def test_init_prompts_download_when_model_missing(self, runner: CliRunner, config_dir: Path) -> None:
        """init must ask to download when the model is missing and download on 'y'."""
        with (
            patch("corridorkey_cli.commands.init.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.init.export_config"),
            patch("corridorkey_cli.commands.init.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.init.download_model") as mock_dl,
            patch("corridorkey_cli.commands.doctor.doctor"),
        ):
            cfg = MagicMock()
            cfg.app_dir = config_dir
            cfg.checkpoint_dir = config_dir / "models"
            mock_cfg.return_value = cfg
            mock_model.return_value = False
            mock_dl.return_value = config_dir / "models" / "greenformer_v2.pth"

            # Answer 'y' to the download prompt
            result = runner.invoke(app, ["init"], input="y\n")

        assert result.exit_code == 0
        mock_dl.assert_called_once()

    def test_init_skips_download_on_no(self, runner: CliRunner, config_dir: Path) -> None:
        """Answering 'n' to the download prompt must skip download and print manual instructions."""
        with (
            patch("corridorkey_cli.commands.init.load_config") as mock_cfg,
            patch("corridorkey_cli.commands.init.export_config"),
            patch("corridorkey_cli.commands.init.is_model_present") as mock_model,
            patch("corridorkey_cli.commands.init.download_model") as mock_dl,
            patch("corridorkey_cli.commands.doctor.doctor"),
        ):
            cfg = MagicMock()
            cfg.app_dir = config_dir
            cfg.checkpoint_dir = config_dir / "models"
            mock_cfg.return_value = cfg
            mock_model.return_value = False

            result = runner.invoke(app, ["init"], input="n\n")

        assert result.exit_code == 0
        mock_dl.assert_not_called()
        assert "manually" in result.output.lower() or "download" in result.output.lower()
