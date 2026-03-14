"""``corridorkey doctor`` - read-only environment health check."""

from __future__ import annotations

import platform
import shutil
import sys

import typer
from corridorkey import CorridorKeyService, check_ffmpeg, load_config
from corridorkey.model_manager import is_model_present
from rich.table import Table

from corridorkey_cli._helpers import console

app = typer.Typer(help="Check environment health.")

_PASS = "[green]OK[/green]"
_FAIL = "[red]FAIL[/red]"
_WARN = "[yellow]WARN[/yellow]"


@app.callback(invoke_without_command=True)
def doctor() -> None:
    """Run a read-only environment health check and print a results table."""
    rows: list[tuple[str, str, str]] = []
    all_ok = True

    # Python version
    major, minor = sys.version_info[:2]
    py_ok = major == 3 and minor >= 13
    rows.append((
        "Python >= 3.13",
        _PASS if py_ok else _FAIL,
        f"{major}.{minor}.{sys.version_info.micro}",
    ))
    if not py_ok:
        all_ok = False

    # Git
    git_path = shutil.which("git")
    rows.append((
        "git",
        _PASS if git_path else _WARN,
        git_path or "not found on PATH (needed for some alpha generator installs)",
    ))

    # FFmpeg / ffprobe
    ffmpeg_info = check_ffmpeg()
    ffmpeg_detail: str = str(
        ffmpeg_info.get("version") or ffmpeg_info.get("ffmpeg_path") or "not found - install FFmpeg"
    )
    rows.append(("ffmpeg", _PASS if ffmpeg_info["available"] else _FAIL, ffmpeg_detail))
    rows.append((
        "ffprobe",
        _PASS if ffmpeg_info["ffprobe_path"] else _FAIL,
        str(ffmpeg_info["ffprobe_path"]) or "not found - ships with FFmpeg",
    ))
    if not ffmpeg_info["available"]:
        all_ok = False

    # Compute device
    try:
        from corridorkey.device_utils import detect_best_device

        device = detect_best_device()
        rows.append(("compute device", _PASS, device))
    except Exception as e:
        rows.append(("compute device", _WARN, str(e)))

    # VRAM (CUDA only)
    try:
        service = CorridorKeyService()
        vram = service.get_vram_info()
        if vram:
            rows.append((
                "VRAM",
                _PASS,
                f"{vram['free']:.1f} GB free / {vram['total']:.1f} GB total ({vram['name']})",
            ))
    except Exception:
        pass

    # Config file
    config = load_config()
    config_file = config.app_dir / "corridorkey.toml"
    rows.append((
        "config file",
        _PASS if config_file.exists() else _WARN,
        str(config_file) if config_file.exists() else "not found - run `corridorkey init` to create it",
    ))

    # Checkpoint directory
    rows.append((
        "checkpoint_dir",
        _PASS if config.checkpoint_dir.is_dir() else _WARN,
        str(config.checkpoint_dir),
    ))

    # Inference model
    model_ok = is_model_present(config)
    rows.append((
        "inference model",
        _PASS if model_ok else _FAIL,
        "found" if model_ok else f"not found in {config.checkpoint_dir} - run `corridorkey init`",
    ))
    if not model_ok:
        all_ok = False

    # Platform info
    rows.append(("platform", _PASS, f"{platform.system()} {platform.machine()}"))

    _render_table(rows)

    if all_ok:
        console.print("\n[green]All checks passed. Ready to run.[/green]")
    else:
        console.print("\n[red]Some checks failed. Run `corridorkey init` to fix setup issues.[/red]")
        raise typer.Exit(1)


def _render_table(rows: list[tuple[str, str, str]]) -> None:
    table = Table(title="Environment Check", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    for check, status, detail in rows:
        table.add_row(check, status, detail)
    console.print(table)
