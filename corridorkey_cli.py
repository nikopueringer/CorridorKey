"""CorridorKey command-line interface and interactive wizard.

This module handles CLI subcommands, environment setup, and the
interactive wizard workflow. The pipeline logic lives in clip_manager.py,
which can be imported independently as a library.

Usage:
    uv run corridorkey wizard "V:\\..."
    uv run corridorkey run-inference
    uv run corridorkey generate-alphas
    uv run corridorkey list-clips
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import sys
import warnings
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from clip_manager import (
    LINUX_MOUNT_ROOT,
    ClipEntry,
    InferenceSettings,
    generate_alphas,
    is_video_file,
    map_path,
    organize_target,
    run_inference,
    run_videomama,
    scan_clips,
)
from device_utils import resolve_device

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="corridorkey",
    help="Neural network green screen keying for professional VFX pipelines.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _configure_environment() -> None:
    """Set up logging and warnings for interactive CLI use."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ---------------------------------------------------------------------------
# Readline-safe input helper
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"(\x1b\[[0-9;]*m)")


def _readline_input(markup: str, *, suffix: str = ": ") -> str:
    """Prompt with Rich markup, safe for readline/backspace.

    Renders *markup* to an ANSI string, wraps escape sequences in
    readline ignore markers (``\\x01``/``\\x02``), then passes the
    result to :func:`input`.  This lets readline track cursor position
    correctly so backspace never erases the prompt text.
    """
    with console.capture() as cap:
        console.print(markup, end="")
    ansi = cap.get() + suffix
    safe = _ANSI_RE.sub(lambda m: "\x01" + m.group(1) + "\x02", ansi)
    return input(safe)


# ---------------------------------------------------------------------------
# Progress helpers (callback protocol → rich.progress)
# ---------------------------------------------------------------------------


class ProgressContext:
    """Context manager bridging clip_manager callbacks to Rich progress bars.

    clip_manager's callback protocol doesn't know about Rich, so this class
    owns the Progress instance and exposes bound methods as callbacks.
    ``__exit__`` always cleans up, even if inference raises.
    """

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self._frame_task_id: TaskID | None = None

    def __enter__(self) -> "ProgressContext":
        self._progress.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.__exit__(*exc)

    def on_clip_start(self, clip_name: str, num_frames: int) -> None:
        """Callback: reset the progress bar for a new clip."""
        if self._frame_task_id is not None:
            self._progress.remove_task(self._frame_task_id)
        self._frame_task_id = self._progress.add_task(f"[cyan]{clip_name}", total=num_frames)

    def on_frame_complete(self, frame_idx: int, num_frames: int) -> None:
        """Callback: advance the progress bar by one frame."""
        if self._frame_task_id is not None:
            self._progress.advance(self._frame_task_id)


def _on_clip_start_log_only(clip_name: str, total_clips: int) -> None:
    """Clip-level callback for generate-alphas.

    Unlike ProgressContext.on_clip_start (frame-level granularity with a Rich
    task per clip), GVM has no per-frame progress so we just log.
    """
    console.print(f"  Processing [bold]{clip_name}[/bold] ({total_clips} total)")


# ---------------------------------------------------------------------------
# Inference settings prompt (rich.prompt — CLI layer only)
# ---------------------------------------------------------------------------


def _prompt_inference_settings(
    *,
    default_linear: bool | None = None,
    default_despill: int | None = None,
    default_despeckle: bool | None = None,
    default_despeckle_size: int | None = None,
    default_refiner: float | None = None,
) -> tuple[InferenceSettings | None, int]:
    """Interactively prompt for inference settings, skipping any pre-filled values.

    Returns (settings, lines_printed). *lines_printed* counts terminal lines
    produced so the caller can erase them on cancel.
    """
    lines = 0
    with console.capture() as cap_hdr:
        console.print(Panel("Inference Settings", style="bold cyan"))
    hdr = cap_hdr.get()
    lines += hdr.count("\n")
    sys.stdout.write(hdr)
    sys.stdout.flush()

    try:
        if default_linear is not None:
            input_is_linear = default_linear
        else:
            while True:
                gamma_choice = _readline_input(
                    "Input colorspace"
                    " [bold magenta]\\[[/bold magenta]"
                    "[bold magenta]l[/bold magenta][magenta]inear[/magenta]"
                    "[bold magenta]/[/bold magenta]"
                    "[bold magenta]s[/bold magenta][magenta]rgb[/magenta]"
                    "[bold magenta]][/bold magenta]"
                    " [cyan](srgb)[/cyan]",
                )
                val = gamma_choice.strip().lower()
                if not val:
                    input_is_linear = False
                    lines += 1
                    break
                elif val in ("l", "linear"):
                    input_is_linear = True
                    lines += 1
                    break
                elif val in ("s", "srgb"):
                    input_is_linear = False
                    lines += 1
                    break

        if default_despill is not None:
            despill_int = max(0, min(10, default_despill))
        else:
            while True:
                raw = _readline_input(
                    "Despill strength [cyan](0–10, 10 = max despill)[/cyan]"
                    " [cyan](5)[/cyan]",
                )
                val = raw.strip()
                if not val:
                    despill_int = 5
                    lines += 1
                    break
                try:
                    despill_int = int(val)
                    despill_int = max(0, min(10, despill_int))
                    lines += 1
                    break
                except ValueError:
                    if console.is_terminal:
                        sys.stdout.write("\033[A\r\033[J")
                        sys.stdout.flush()
        despill_strength = despill_int / 10.0

        if default_despeckle is not None:
            auto_despeckle = default_despeckle
        else:
            while True:
                raw = _readline_input(
                    "Enable auto-despeckle (removes tracking dots)?"
                    " [bold magenta]\\[[/bold magenta]"
                    "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
                    "[bold magenta]/[/bold magenta]"
                    "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
                    "[bold magenta]][/bold magenta]"
                    " [cyan](yes)[/cyan]",
                )
                val = raw.strip().lower()
                if not val or val in ("y", "yes"):
                    auto_despeckle = True
                    lines += 1
                    break
                elif val in ("n", "no"):
                    auto_despeckle = False
                    lines += 1
                    break
                if console.is_terminal:
                    sys.stdout.write("\033[A\r\033[J")
                    sys.stdout.flush()

        despeckle_size = default_despeckle_size if default_despeckle_size is not None else 400
        if auto_despeckle and default_despeckle_size is None and default_despeckle is None:
            while True:
                raw = _readline_input(
                    "Despeckle size [cyan](min pixels for a spot)[/cyan]"
                    " [cyan](400)[/cyan]",
                )
                val = raw.strip()
                if not val:
                    despeckle_size = 400
                    lines += 1
                    break
                try:
                    despeckle_size = max(0, int(val))
                    lines += 1
                    break
                except ValueError:
                    if console.is_terminal:
                        sys.stdout.write("\033[A\r\033[J")
                        sys.stdout.flush()

        if default_refiner is not None:
            refiner_scale = default_refiner
        else:
            while True:
                raw = _readline_input(
                    "Refiner strength multiplier [dim](experimental)[/dim]"
                    " [cyan](1.0)[/cyan]",
                )
                val = raw.strip()
                if not val:
                    refiner_scale = 1.0
                    lines += 1
                    break
                try:
                    refiner_scale = float(val)
                    lines += 1
                    break
                except ValueError:
                    if console.is_terminal:
                        sys.stdout.write("\033[A\r\033[J")
                        sys.stdout.flush()

        return InferenceSettings(
            input_is_linear=input_is_linear,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            refiner_scale=refiner_scale,
        ), lines
    except EOFError:
        return None, lines


# ---------------------------------------------------------------------------
# Typer callback (shared options)
# ---------------------------------------------------------------------------


@app.callback()
def app_callback(
    ctx: typer.Context,
    device: Annotated[
        str,
        typer.Option(help="Compute device: auto, cuda, mps, cpu"),
    ] = "auto",
) -> None:
    """Neural network green screen keying for professional VFX pipelines."""
    _configure_environment()
    ctx.ensure_object(dict)
    ctx.obj["device"] = resolve_device(device)
    logger.info("Using device: %s", ctx.obj["device"])


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("list-clips")
def list_clips_cmd(ctx: typer.Context) -> None:
    """List all clips in ClipsForInference and their status."""
    scan_clips()


@app.command("generate-alphas")
def generate_alphas_cmd(ctx: typer.Context) -> None:
    """Generate coarse alpha hints via GVM for clips missing them."""
    clips = scan_clips()
    try:
        generate_alphas(clips, device=ctx.obj["device"], on_clip_start=_on_clip_start_log_only)
    except KeyboardInterrupt:
        console.print("\n[yellow]Alpha generation interrupted.[/yellow]")
        return
    console.print("[bold green]Alpha generation complete.")


@app.command("run-inference")
def run_inference_cmd(
    ctx: typer.Context,
    backend: Annotated[
        str,
        typer.Option(help="Inference backend: auto, torch, mlx"),
    ] = "auto",
    max_frames: Annotated[
        Optional[int],
        typer.Option("--max-frames", help="Limit frames per clip"),
    ] = None,
    linear: Annotated[
        Optional[bool],
        typer.Option("--linear/--srgb", help="Input colorspace (default: prompt)"),
    ] = None,
    despill: Annotated[
        Optional[int],
        typer.Option("--despill", help="Despill strength 0–10 (default: prompt)"),
    ] = None,
    despeckle: Annotated[
        Optional[bool],
        typer.Option("--despeckle/--no-despeckle", help="Auto-despeckle toggle (default: prompt)"),
    ] = None,
    despeckle_size: Annotated[
        Optional[int],
        typer.Option("--despeckle-size", help="Min pixel size for despeckle (default: prompt)"),
    ] = None,
    refiner: Annotated[
        Optional[float],
        typer.Option("--refiner", help="Refiner strength multiplier (default: prompt)"),
    ] = None,
) -> None:
    """Run CorridorKey inference on clips with Input + AlphaHint.

    Settings can be passed as flags for non-interactive use, or omitted to
    prompt interactively.
    """
    clips = scan_clips()

    # despeckle_size excluded — sensible default even in headless mode
    required_flags_set = all(v is not None for v in [linear, despill, despeckle, refiner])
    if required_flags_set:
        assert linear is not None and despill is not None and despeckle is not None and refiner is not None
        despill_clamped = max(0, min(10, despill))
        settings = InferenceSettings(
            input_is_linear=linear,
            despill_strength=despill_clamped / 10.0,
            auto_despeckle=despeckle,
            despeckle_size=despeckle_size if despeckle_size is not None else 400,
            refiner_scale=refiner,
        )
    else:
        try:
            settings, _ = _prompt_inference_settings(
                default_linear=linear,
                default_despill=despill,
                default_despeckle=despeckle,
                default_despeckle_size=despeckle_size,
                default_refiner=refiner,
            )
        except EOFError:
            console.print("[yellow]Aborted.[/yellow]")
            return
        if settings is None:
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        with ProgressContext() as ctx_progress:
            run_inference(
                clips,
                device=ctx.obj["device"],
                backend=backend,
                max_frames=max_frames,
                settings=settings,
                on_clip_start=ctx_progress.on_clip_start,
                on_frame_complete=ctx_progress.on_frame_complete,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Inference interrupted.[/yellow]")
        return

    console.print("[bold green]Inference complete.")


@app.command()
def wizard(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Target path (Windows or local)")],
) -> None:
    """Interactive wizard for organizing clips and running the pipeline."""
    interactive_wizard(path, device=ctx.obj["device"])


# ---------------------------------------------------------------------------
# Wizard (rich-styled)
# ---------------------------------------------------------------------------


def interactive_wizard(win_path: str, device: str | None = None) -> None:
    console.print(Panel("[bold]CORRIDOR KEY — SMART WIZARD[/bold]", style="cyan"))

    # 1. Resolve Path
    console.print(f"Windows Path: {win_path}")

    if os.path.exists(win_path):
        process_path = win_path
        console.print(f"Running locally: [bold]{process_path}[/bold]")
    else:
        process_path = map_path(win_path)
        console.print(f"Linux/Remote Path: [bold]{process_path}[/bold]")

        if not os.path.exists(process_path):
            console.print(
                f"\n[bold red]ERROR:[/bold red] Path does not exist locally OR on Linux mount!\n"
                f"Expected Linux Mount Root: {LINUX_MOUNT_ROOT}"
            )
            raise typer.Exit(code=1)

    # 2. Analyze — shot or project?
    target_is_shot = False
    if os.path.exists(os.path.join(process_path, "Input")) or glob.glob(os.path.join(process_path, "Input.*")):
        target_is_shot = True

    work_dirs: list[str] = []
    # Pipeline output dirs, not clip sources
    excluded_dirs = {"Output", "AlphaHint", "VideoMamaMaskHint", ".ipynb_checkpoints"}
    if target_is_shot:
        work_dirs = [process_path]
    else:
        work_dirs = [
            os.path.join(process_path, d)
            for d in os.listdir(process_path)
            if os.path.isdir(os.path.join(process_path, d)) and d not in excluded_dirs
        ]

    console.print(f"\nFound [bold]{len(work_dirs)}[/bold] potential clip folders.")

    # Files already named Input/AlphaHint/etc are organized, not "loose"
    known_names = {"input", "alphahint", "videomamamaskhint"}
    loose_videos = [
        f
        for f in os.listdir(process_path)
        if is_video_file(f)
        and os.path.isfile(os.path.join(process_path, f))
        and os.path.splitext(f)[0].lower() not in known_names
    ]

    dirs_needing_org = []
    for d in work_dirs:
        has_input = os.path.exists(os.path.join(d, "Input")) or glob.glob(os.path.join(d, "Input.*"))
        has_alpha = os.path.exists(os.path.join(d, "AlphaHint"))
        has_mask = os.path.exists(os.path.join(d, "VideoMamaMaskHint"))
        if not has_input or not has_alpha or not has_mask:
            dirs_needing_org.append(d)

    if loose_videos or dirs_needing_org:
        if loose_videos:
            console.print(f"Found [yellow]{len(loose_videos)}[/yellow] loose video files:")
            for v in loose_videos:
                console.print(f"  • {v}")

        if dirs_needing_org:
            console.print(f"Found [yellow]{len(dirs_needing_org)}[/yellow] folders needing setup:")
            display_limit = 10
            for d in dirs_needing_org[:display_limit]:
                console.print(f"  • {os.path.basename(d)}")
            if len(dirs_needing_org) > display_limit:
                console.print(f"  …and {len(dirs_needing_org) - display_limit} others.")

        # 3. Organize
        organize = _readline_input(
            "\nOrganize clips & create hint folders?"
            " [bold magenta]\\[[/bold magenta]"
            "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
            "[bold magenta]/[/bold magenta]"
            "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
            "[bold magenta]][/bold magenta]"
            " [cyan](no)[/cyan]",
        ).strip().lower()
        if organize in ("y", "yes"):
            for v in loose_videos:
                clip_name = os.path.splitext(v)[0]
                ext = os.path.splitext(v)[1]
                target_folder = os.path.join(process_path, clip_name)

                if os.path.exists(target_folder):
                    logger.warning(f"Skipping loose video '{v}': Target folder '{clip_name}' already exists.")
                    continue

                try:
                    os.makedirs(target_folder)
                    target_file = os.path.join(target_folder, f"Input{ext}")
                    shutil.move(os.path.join(process_path, v), target_file)
                    logger.info(f"Organized: Moved '{v}' to '{clip_name}/Input{ext}'")
                    for hint in ["AlphaHint", "VideoMamaMaskHint"]:
                        os.makedirs(os.path.join(target_folder, hint), exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to organize video '{v}': {e}")

            for d in work_dirs:
                organize_target(d)
            console.print("[green]Organization complete.[/green]")

            if not target_is_shot:
                work_dirs = [
                    os.path.join(process_path, d)
                    for d in os.listdir(process_path)
                    if os.path.isdir(os.path.join(process_path, d)) and d not in excluded_dirs
                ]

    # 4. Status Check Loop
    _erase_lines = 0  # Lines to erase before next menu draw

    def _erase_menu() -> None:
        nonlocal _erase_lines
        if _erase_lines > 0 and console.is_terminal:
            # Move cursor up N lines and clear from cursor to end of screen
            sys.stdout.write(f"\033[{_erase_lines}F\033[J")
            sys.stdout.flush()
        _erase_lines = 0

    try:
        while True:
            ready: list[ClipEntry] = []
            masked: list[ClipEntry] = []
            raw: list[ClipEntry] = []

            for d in work_dirs:
                entry = ClipEntry(os.path.basename(d), d)
                try:
                    entry.find_assets()
                except (FileNotFoundError, ValueError, OSError):
                    pass

                has_mask = False
                try:
                    mask_dir = os.path.join(d, "VideoMamaMaskHint")
                    if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                        has_mask = True
                    if not has_mask:
                        for f in os.listdir(d):
                            stem, _ = os.path.splitext(f)
                            if stem.lower() == "videomamamaskhint" and is_video_file(f):
                                has_mask = True
                                break
                except OSError:
                    pass

                if entry.alpha_asset:
                    ready.append(entry)
                elif has_mask:
                    masked.append(entry)
                else:
                    raw.append(entry)

            missing_alpha = masked + raw

            # Build menu renderables
            table = Table(show_lines=True)
            table.add_column("Category", style="bold")
            table.add_column("Count", justify="right")
            table.add_column("Clips")

            table.add_row(
                "[green]Ready[/green] (AlphaHint)",
                str(len(ready)),
                ", ".join(c.name for c in ready) or "—",
            )
            table.add_row(
                "[yellow]Masked[/yellow] (VideoMaMaMaskHint)",
                str(len(masked)),
                ", ".join(c.name for c in masked) or "—",
            )
            table.add_row(
                "[red]Raw[/red] (Input only)",
                str(len(raw)),
                ", ".join(c.name for c in raw) or "—",
            )

            actions: list[str] = []
            if missing_alpha:
                actions.append(f"[bold]v[/bold] — Run VideoMaMa ({len(masked)} with masks)")
                actions.append(f"[bold]g[/bold] — Run GVM (auto-matte {len(raw)} clips)")
            if ready:
                actions.append(f"[bold]i[/bold] — Run Inference ({len(ready)} ready clips)")
            actions.append("[bold]r[/bold] — Re-scan folders")
            actions.append("[bold]q[/bold] — Quit [dim](ctrl+d)[/dim]")

            actions_panel = Panel("\n".join(actions), title="Actions", style="blue")

            # Erase previous menu, then render new one
            _erase_menu()
            with console.capture() as cap:
                console.print(table)
                console.print(actions_panel)
            menu_output = cap.get()
            menu_line_count = menu_output.count("\n")
            sys.stdout.write(menu_output)
            sys.stdout.flush()

            # Prompt on a separate line with readline-safe input
            while True:
                try:
                    choice = _readline_input("Select action")
                except EOFError:
                    choice = "q"
                    break
                choice = choice.strip().lower()
                if choice in ("v", "g", "i", "r", "q"):
                    break
                # Invalid or empty — erase the prompt line and re-prompt
                if console.is_terminal:
                    sys.stdout.write("\033[A\r\033[J")
                    sys.stdout.flush()

            if choice == "v":
                _erase_lines = menu_line_count + 2
                _erase_menu()
                with console.capture() as cap_v:
                    console.print(Panel("VideoMaMa", style="magenta"))
                v_hdr = cap_v.get()
                v_hdr_lines = v_hdr.count("\n")
                sys.stdout.write(v_hdr)
                sys.stdout.flush()
                try:
                    run_videomama(missing_alpha, chunk_size=50, device=device)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/yellow]")
                try:
                    _readline_input("Press Enter to return to menu", suffix="")
                except EOFError:
                    _erase_lines = v_hdr_lines + 3
                    continue

            elif choice == "g":
                _erase_lines = menu_line_count + 2
                _erase_menu()
                with console.capture() as cap_g:
                    console.print(Panel("GVM Auto-Matte", style="magenta"))
                g_hdr = cap_g.get()
                g_lines = g_hdr.count("\n")
                sys.stdout.write(g_hdr)
                sys.stdout.flush()
                with console.capture() as cap_g2:
                    console.print(f"Will generate alphas for {len(raw)} clips without mask hints.")
                g_info = cap_g2.get()
                g_lines += g_info.count("\n")
                sys.stdout.write(g_info)
                sys.stdout.flush()
                try:
                    gvm_yes = _readline_input(
                        "Proceed with GVM?"
                        " [bold magenta]\\[[/bold magenta]"
                        "[bold magenta]y[/bold magenta][magenta]es[/magenta]"
                        "[bold magenta]/[/bold magenta]"
                        "[bold magenta]n[/bold magenta][magenta]o[/magenta]"
                        "[bold magenta]][/bold magenta]"
                        " [cyan](no)[/cyan]",
                    ).strip().lower()
                    if gvm_yes in ("y", "yes"):
                        try:
                            generate_alphas(raw, device=device)
                        except KeyboardInterrupt:
                            console.print("\n[yellow]Interrupted.[/yellow]")
                        try:
                            _readline_input("Press Enter to return to menu", suffix="")
                        except EOFError:
                            pass
                    else:
                        # Declined — erase the sub-menu
                        _erase_lines = g_lines + 3
                        continue
                except EOFError:
                    _erase_lines = g_lines + 2
                    continue

            elif choice == "i":
                # Erase the menu before showing inference settings
                _erase_lines = menu_line_count + 2
                _erase_menu()
                with console.capture() as cap_i:
                    console.print(Panel("Corridor Key Inference", style="magenta"))
                i_hdr = cap_i.get()
                i_hdr_lines = i_hdr.count("\n")
                sys.stdout.write(i_hdr)
                sys.stdout.flush()
                try:
                    settings, settings_lines = _prompt_inference_settings()
                    if settings is None:
                        _erase_lines = i_hdr_lines + settings_lines + 2
                        continue
                    with ProgressContext() as ctx_progress:
                        run_inference(
                            ready,
                            device=device,
                            settings=settings,
                            on_clip_start=ctx_progress.on_clip_start,
                            on_frame_complete=ctx_progress.on_frame_complete,
                        )
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/yellow]")
                except Exception as e:
                    console.print(f"[bold red]Inference failed:[/bold red] {e}")
                try:
                    _readline_input("Press Enter to return to menu", suffix="")
                except EOFError:
                    pass

            elif choice == "r":
                # Erase menu + prompt line, redraw with fresh scan
                _erase_lines = menu_line_count + 2
                continue

            elif choice == "q":
                # Erase menu + prompt line before goodbye
                _erase_lines = menu_line_count + 2
                _erase_menu()
                break
    except KeyboardInterrupt:
        pass  # Fall through to goodbye message

    console.print("[bold green]Wizard complete. Goodbye![/bold green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point called by the `corridorkey` console script."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
