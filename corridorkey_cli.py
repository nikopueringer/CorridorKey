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
import shutil
import sys
import warnings
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, IntPrompt, Prompt
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
) -> InferenceSettings:
    """Interactively prompt for inference settings, skipping any pre-filled values."""
    console.print(Panel("Inference Settings", style="bold cyan"))

    if default_linear is not None:
        input_is_linear = default_linear
    else:
        gamma_choice = Prompt.ask(
            "Input colorspace",
            choices=["linear", "srgb"],
            default="srgb",
        )
        input_is_linear = gamma_choice == "linear"

    if default_despill is not None:
        despill_int = max(0, min(10, default_despill))
    else:
        despill_int = IntPrompt.ask(
            "Despill strength (0–10, 10 = max despill)",
            default=5,
        )
        despill_int = max(0, min(10, despill_int))
    despill_strength = despill_int / 10.0

    if default_despeckle is not None:
        auto_despeckle = default_despeckle
    else:
        auto_despeckle = Confirm.ask(
            "Enable auto-despeckle (removes tracking dots)?",
            default=True,
        )

    despeckle_size = default_despeckle_size if default_despeckle_size is not None else 400
    if auto_despeckle and default_despeckle_size is None and default_despeckle is None:
        despeckle_size = IntPrompt.ask(
            "Despeckle size (min pixels for a spot)",
            default=400,
        )
        despeckle_size = max(0, despeckle_size)

    if default_refiner is not None:
        refiner_scale = default_refiner
    else:
        refiner_val = Prompt.ask(
            "Refiner strength multiplier [dim](experimental)[/dim]",
            default="1.0",
        )
        try:
            refiner_scale = float(refiner_val)
        except ValueError:
            refiner_scale = 1.0

    return InferenceSettings(
        input_is_linear=input_is_linear,
        despill_strength=despill_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner_scale,
    )


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
    with console.status("[bold green]Loading GVM model..."):
        generate_alphas(clips, device=ctx.obj["device"], on_clip_start=_on_clip_start_log_only)
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
        settings = _prompt_inference_settings(
            default_linear=linear,
            default_despill=despill,
            default_despeckle=despeckle,
            default_despeckle_size=despeckle_size,
            default_refiner=refiner,
        )

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
        if Confirm.ask("\nOrganize clips & create hint folders?", default=False):
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
            mask_dir = os.path.join(d, "VideoMamaMaskHint")
            if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                has_mask = True
            if not has_mask:
                for f in os.listdir(d):
                    stem, _ = os.path.splitext(f)
                    if stem.lower() == "videomamamaskhint" and is_video_file(f):
                        has_mask = True
                        break

            if entry.alpha_asset:
                ready.append(entry)
            elif has_mask:
                masked.append(entry)
            else:
                raw.append(entry)

        table = Table(title="Status Report", show_lines=True)
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
        console.print(table)

        missing_alpha = masked + raw
        actions: list[str] = []

        if missing_alpha:
            actions.append(f"[bold]v[/bold] — Run VideoMaMa ({len(masked)} with masks)")
            actions.append(f"[bold]g[/bold] — Run GVM (auto-matte {len(raw)} clips)")
        if ready:
            actions.append(f"[bold]i[/bold] — Run Inference ({len(ready)} ready clips)")
        actions.append("[bold]r[/bold] — Re-scan folders")
        actions.append("[bold]q[/bold] — Quit")

        console.print(Panel("\n".join(actions), title="Actions", style="blue"))

        choice = Prompt.ask("Select action", choices=["v", "g", "i", "r", "q"], default="q")

        if choice == "v":
            console.print(Panel("VideoMaMa", style="magenta"))
            run_videomama(missing_alpha, chunk_size=50, device=device)
            Prompt.ask("VideoMaMa batch complete. Press Enter to re-scan")

        elif choice == "g":
            console.print(Panel("GVM Auto-Matte", style="magenta"))
            console.print(f"Will generate alphas for {len(raw)} clips without mask hints.")
            if Confirm.ask("Proceed with GVM?", default=False):
                generate_alphas(raw, device=device)
                Prompt.ask("GVM batch complete. Press Enter to re-scan")

        elif choice == "i":
            console.print(Panel("Corridor Key Inference", style="magenta"))
            try:
                settings = _prompt_inference_settings()
                with ProgressContext() as ctx_progress:
                    run_inference(
                        ready,
                        device=device,
                        settings=settings,
                        on_clip_start=ctx_progress.on_clip_start,
                        on_frame_complete=ctx_progress.on_frame_complete,
                    )
            except (RuntimeError, FileNotFoundError) as e:
                console.print(f"[bold red]Inference failed:[/bold red] {e}")
            Prompt.ask("Inference batch complete. Press Enter to re-scan")

        elif choice == "r":
            console.print("Re-scanning…")

        elif choice == "q":
            break

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
