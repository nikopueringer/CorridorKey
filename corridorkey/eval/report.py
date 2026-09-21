"""Evaluation reporting and formatting utilities.

Generates structured JSON receipts and clean terminal tables for
model benchmark results.
"""

from __future__ import annotations

import json
from typing import Any, TextIO

from .runner import EvaluationReport


def format_json_report(report: EvaluationReport, indent: int = 2) -> str:
    """Format an EvaluationReport as a structured JSON string."""
    return json.dumps(report.to_dict(), indent=indent)


def write_json_report(report: EvaluationReport, out_stream: TextIO | str) -> None:
    """Write an EvaluationReport to an open file stream or path string."""
    content = format_json_report(report)
    if isinstance(out_stream, str):
        with open(out_stream, "w", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")
    else:
        out_stream.write(content)
        out_stream.write("\n")


def format_text_table(report: EvaluationReport) -> str:
    """Format an EvaluationReport as a clean ASCII benchmark table."""
    lines: list[str] = []
    lines.append("=" * 84)
    header_model = f" CorridorKey Evaluation Benchmark: {report.model_name or 'Default'}"
    lines.append(f"{header_model:<84}")
    lines.append("=" * 84)

    # Summary section
    lines.append(
        f"Samples: {report.sample_count:<6} | Alpha MAE: {report.alpha_mae:.4f} | "
        f"FG MAE: {report.foreground_mae:.4f} | Boundary MAE: {report.boundary_alpha_mae:.4f} | "
        f"Recomp: {report.recomposition_mae:.4f}"
    )
    lines.append("-" * 84)

    # Case breakdown
    col_case = "Test Case"
    lines.append(f"{col_case:<26} {'Alpha MAE':>12} {'FG MAE':>12} {'Boundary MAE':>14} {'Recomp MAE':>14}")
    lines.append("-" * 84)

    for name, c in sorted(report.cases.items()):
        lines.append(
            f"{name:<26} {c.alpha_mae:>12.4f} {c.foreground_mae:>12.4f} "
            f"{c.boundary_alpha_mae:>14.4f} {c.recomposition_mae:>14.4f}"
        )

    # Tag breakdown if multiple tags exist
    if report.tag_breakdown:
        lines.append("-" * 84)
        lines.append("Failure Mode / Category Breakdown:")
        lines.append(
            f"{'Category Tag':<26} {'Count':>6} {'Alpha MAE':>12} {'FG MAE':>10} {'Boundary':>12} {'Recomp':>12}"
        )
        lines.append("-" * 84)
        for tag, metrics in sorted(report.tag_breakdown.items()):
            cnt = int(metrics.get("count", 0))
            lines.append(
                f"{tag:<26} {cnt:>6} {metrics['alpha_mae']:>12.4f} {metrics['foreground_mae']:>10.4f} "
                f"{metrics['boundary_alpha_mae']:>12.4f} {metrics['recomposition_mae']:>12.4f}"
            )

    lines.append("=" * 84)
    return "\n".join(lines)


def print_report(report: EvaluationReport) -> None:
    """Print the benchmark report to stdout, utilizing rich if available."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"CorridorKey Model Quality Benchmark: [bold green]{report.model_name or 'Candidate'}[/bold green]",
            header_style="bold cyan",
        )
        table.add_column("Test Case", style="bold")
        table.add_column("Alpha MAE", justify="right")
        table.add_column("Foreground MAE", justify="right")
        table.add_column("Boundary Alpha MAE", justify="right")
        table.add_column("Recomposition MAE", justify="right")

        for name, c in sorted(report.cases.items()):
            table.add_row(
                name,
                f"{c.alpha_mae:.4f}",
                f"{c.foreground_mae:.4f}",
                f"{c.boundary_alpha_mae:.4f}",
                f"{c.recomposition_mae:.4f}",
            )

        # Global row
        table.add_section()
        table.add_row(
            "[bold yellow]GLOBAL MEAN[/bold yellow]",
            f"[bold]{report.alpha_mae:.4f}[/bold]",
            f"[bold]{report.foreground_mae:.4f}[/bold]",
            f"[bold]{report.boundary_alpha_mae:.4f}[/bold]",
            f"[bold]{report.recomposition_mae:.4f}[/bold]",
        )
        console.print(table)

    except ImportError:
        print(format_text_table(report))


def format_hint_sweep_text_table(report: Any) -> str:
    """Format a HintSweepReport as an ASCII table."""
    lines: list[str] = []
    lines.append("=" * 86)
    header = f" CorridorKey Alpha-Hint Asymmetry Sweep: {report.model_name or 'Default'}"
    lines.append(f"{header:<86}")
    lines.append("=" * 86)
    hdr = (
        f"{'Corruption Profile':<24} {'Hint IoU':>10} {'Alpha MAE':>12} "
        f"{'Boundary MAE':>14} {'FP Alpha':>12} {'FN Alpha':>12}"
    )
    lines.append(hdr)
    lines.append("-" * 86)
    for r in report.results:
        lines.append(
            f"{r.profile_name:<24} {r.hint_iou:>10.4f} {r.alpha_mae:>12.4f} "
            f"{r.boundary_alpha_mae:>14.4f} {r.fp_alpha:>12.4f} {r.fn_alpha:>12.4f}"
        )
    lines.append("=" * 86)
    return "\n".join(lines)


def print_hint_sweep_report(report: Any) -> None:
    """Print the hint sweep benchmark to stdout, using rich if available."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=(
                f"CorridorKey Alpha-Hint Corruption Sweep: [bold green]{report.model_name or 'Candidate'}[/bold green]"
            ),
            header_style="bold cyan",
        )
        table.add_column("Corruption Profile", style="bold")
        table.add_column("Hint IoU", justify="right")
        table.add_column("Alpha MAE", justify="right")
        table.add_column("Boundary Alpha MAE", justify="right")
        table.add_column("FP Alpha", justify="right")
        table.add_column("FN Alpha", justify="right")

        for r in report.results:
            table.add_row(
                r.profile_name,
                f"{r.hint_iou:.4f}",
                f"{r.alpha_mae:.4f}",
                f"{r.boundary_alpha_mae:.4f}",
                f"{r.fp_alpha:.4f}",
                f"{r.fn_alpha:.4f}",
            )
        console.print(table)
    except ImportError:
        print(format_hint_sweep_text_table(report))
