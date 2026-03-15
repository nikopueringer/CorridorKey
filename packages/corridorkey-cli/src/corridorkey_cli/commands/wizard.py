"""``corridorkey wizard`` - interactive processing loop."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from corridorkey import (
    ClipEntry,
    ClipState,
    CorridorKeyConfig,
    CorridorKeyService,
    InferenceParams,
    OutputConfig,
    load_config,
)
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from corridorkey_cli._helpers import ProgressContext, console, err_console

app = typer.Typer(help="Interactive processing wizard.")

_STATE_COLOURS: dict[str, str] = {
    "EXTRACTING": "orange3",
    "RAW": "white",
    "MASKED": "blue",
    "READY": "yellow",
    "COMPLETE": "green",
    "ERROR": "red",
}


@app.callback(invoke_without_command=True)
def wizard(
    clips_dir: Annotated[
        Path | None,
        typer.Argument(help="Directory to process. Prompted interactively if omitted."),
    ] = None,
) -> None:
    """Interactive wizard: scan → review → configure → process → re-scan."""
    console.print(Panel("[bold cyan]CorridorKey — Wizard[/bold cyan]", expand=False))

    # Resolve directory
    if clips_dir is None:
        raw = Prompt.ask("Clips directory")
        clips_dir = Path(raw.strip())

    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    config = load_config()
    service = CorridorKeyService(config)

    # Main loop
    while True:
        clips = service.scan_clips(str(clips_dir))
        _print_state_table(clips, clips_dir)

        ready = [c for c in clips if c.state == ClipState.READY]
        extracting = [c for c in clips if c.state == ClipState.EXTRACTING]
        skippable = [c for c in clips if c.state in (ClipState.RAW, ClipState.MASKED)]
        errors = [c for c in clips if c.state == ClipState.ERROR]

        if skippable:
            console.print(
                f"[yellow]{len(skippable)} clip(s) in RAW/MASKED state.[/yellow] "
                "Install an alpha generator package to process these."
            )
        if errors:
            console.print(f"[red]{len(errors)} clip(s) in ERROR state.[/red] Inspect errors above.")

        # Build action menu
        actionable = ready + extracting
        actions: list[tuple[str, str]] = []
        if actionable:
            label_parts = []
            if ready:
                label_parts.append(f"{len(ready)} READY")
            if extracting:
                label_parts.append(f"{len(extracting)} to extract")
            actions.append(("i", f"Process {', '.join(label_parts)} clip(s)"))
        actions.append(("r", "Re-scan directory"))
        actions.append(("q", "Quit"))

        _print_menu(actions)
        choices = [a[0] for a in actions]
        choice = Prompt.ask("Select action", choices=choices, default="q")

        if choice == "q":
            break

        if choice == "r":
            console.print("[dim]Re-scanning...[/dim]")
            continue

        if choice == "i":
            params, output_config = _prompt_inference_settings(config)
            _run_inference(service, actionable, params, output_config)
            Prompt.ask("\nPress Enter to re-scan")

    console.print("[bold green]Goodbye.[/bold green]")


def _print_state_table(clips: list[ClipEntry], clips_dir: Path) -> None:
    if not clips:
        console.print(f"[yellow]No clips found in {clips_dir}[/yellow]")
        return

    table = Table(title=f"Clips in {clips_dir}", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("State")
    table.add_column("Input", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Error")

    for clip in clips:
        colour = _STATE_COLOURS.get(clip.state.value, "white")
        table.add_row(
            clip.name,
            f"[{colour}]{clip.state.value}[/{colour}]",
            str(clip.input_asset.frame_count) if clip.input_asset else "-",
            str(clip.alpha_asset.frame_count) if clip.alpha_asset else "-",
            f"[red]{clip.error_message}[/red]" if clip.error_message else "",
        )

    console.print(table)


def _print_menu(actions: list[tuple[str, str]]) -> None:
    lines = [f"  [[bold]{key}[/bold]] {label}" for key, label in actions]
    console.print(Panel("\n".join(lines), title="Actions", border_style="blue"))


def _prompt_inference_settings(config: CorridorKeyConfig) -> tuple[InferenceParams, OutputConfig]:
    """Show current config defaults and let the user accept or override them."""
    defaults = InferenceParams(
        input_is_linear=config.input_is_linear,
        despill_strength=config.despill_strength,
        auto_despeckle=config.auto_despeckle,
        despeckle_size=config.despeckle_size,
        refiner_scale=config.refiner_scale,
    )

    # Show current defaults
    table = Table(title="Current Inference Settings", show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("input_is_linear", str(defaults.input_is_linear))
    table.add_row("despill_strength", str(defaults.despill_strength))
    table.add_row("auto_despeckle", str(defaults.auto_despeckle))
    table.add_row("despeckle_size", str(defaults.despeckle_size))
    table.add_row("refiner_scale", str(defaults.refiner_scale))
    console.print(table)

    params = defaults if Confirm.ask("Use these settings?", default=True) else _collect_inference_params(defaults)

    output_config = OutputConfig(
        fg_format=config.fg_format,
        matte_format=config.matte_format,
        comp_format=config.comp_format,
        processed_format=config.processed_format,
    )

    return params, output_config


def _collect_inference_params(defaults: InferenceParams) -> InferenceParams:
    """Interactively prompt for each inference parameter."""
    console.print(Panel("Inference Settings", style="bold cyan"))

    colorspace = Prompt.ask(
        "Input colorspace",
        choices=["linear", "srgb"],
        default="linear" if defaults.input_is_linear else "srgb",
    )
    input_is_linear = colorspace == "linear"

    despill_int = IntPrompt.ask(
        "Despill strength (0-10, 10 = max)",
        default=int(defaults.despill_strength * 10),
    )
    despill_strength = max(0, min(10, despill_int)) / 10.0

    auto_despeckle = Confirm.ask("Enable auto-despeckle?", default=defaults.auto_despeckle)

    despeckle_size = defaults.despeckle_size
    if auto_despeckle:
        despeckle_size = IntPrompt.ask(
            "Despeckle size (min pixel area)",
            default=defaults.despeckle_size,
        )
        despeckle_size = max(0, despeckle_size)

    refiner_scale = FloatPrompt.ask(
        "Refiner scale (1.0 = default)",
        default=defaults.refiner_scale,
    )

    return InferenceParams(
        input_is_linear=input_is_linear,
        despill_strength=despill_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner_scale,
    )


def _run_inference(
    service: CorridorKeyService,
    clips: list[ClipEntry],
    params: InferenceParams,
    output_config: OutputConfig,
) -> None:
    """Extract (if needed) and run inference on a list of clips with a progress bar."""
    import time

    from rich.progress import Progress as RichProgress
    from rich.progress import SpinnerColumn, TextColumn

    failed: list[str] = []

    # Extract any EXTRACTING clips first.
    for clip in clips:
        if clip.state != ClipState.EXTRACTING:
            continue
        total = clip.input_asset.frame_count if clip.input_asset else 0
        console.print(f"\nExtracting [cyan]{clip.name}[/cyan]  ({total} frames)")
        with ProgressContext() as prog:
            try:
                service.extract_clip(clip, on_progress=prog.on_progress)
            except Exception as e:
                console.print(f"  [red]Extraction failed:[/red] {e}")
                failed.append(clip.name)

    # Filter to clips that are now READY (extraction may have left some RAW).
    ready_clips = [c for c in clips if c.state == ClipState.READY]
    if not ready_clips:
        if failed:
            console.print(f"\n[red]Failed clips:[/red] {', '.join(failed)}")
        else:
            console.print("\n[yellow]No READY clips after extraction.[/yellow]")
        return

    # Pre-load the engine once with a spinner so the user sees activity
    # during the potentially long model load + first-frame compilation.
    if not service.is_engine_loaded():
        with RichProgress(
            SpinnerColumn(),
            TextColumn("[cyan]Loading model (first run compiles kernels, ~1 min)...[/cyan]"),
            console=console,
            transient=True,
        ) as spin:
            spin.add_task("")
            service.load_engine()

    for clip in ready_clips:
        total = clip.input_asset.frame_count if clip.input_asset else 0
        console.print(f"\nProcessing [cyan]{clip.name}[/cyan]  ({total} frames)")

        with ProgressContext() as prog:
            t0 = time.monotonic()
            try:
                results = service.run_inference(
                    clip,
                    params,
                    on_progress=prog.on_progress,
                    on_warning=prog.on_warning,
                    output_config=output_config,
                )
                elapsed = time.monotonic() - t0
                ok = sum(1 for r in results if r.success)
                fps = ok / elapsed if elapsed > 0 else 0
                console.print(
                    f"  [green]Done:[/green] {ok}/{len(results)} frames  [dim]{elapsed:.1f}s  ({fps:.2f} fps)[/dim]"
                )
            except Exception as e:
                console.print(f"  [red]Failed:[/red] {e}")
                failed.append(clip.name)

    if failed:
        console.print(f"\n[red]Failed clips:[/red] {', '.join(failed)}")
    else:
        console.print("\n[green]All clips processed successfully.[/green]")
