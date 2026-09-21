"""Command-line entrypoint for CorridorKey model evaluation.

Usage:
    python -m corridorkey.eval --checkpoint CorridorKey_v1.0 --dataset tests/eval/fixtures --output eval.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .cases import load_dataset
from .fixtures import create_diagnostic_fixtures
from .report import print_report, write_json_report
from .runner import evaluate_dataset


def _create_engine(checkpoint: str | None, device: str = "cpu", backend: str = "auto"):
    """Instantiate inference engine based on checkpoint parameter or mock."""
    if checkpoint in ("mock", "dummy"):
        # Deterministic mock engine for testing/CI without weight download
        def _mock_engine(composite, hint):
            # Returns composite as fg, hint as alpha
            return hint, composite

        return _mock_engine

    from CorridorKeyModule.backend import create_engine

    if checkpoint and os.path.isfile(checkpoint):
        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        return CorridorKeyEngine(checkpoint_path=checkpoint, device=device)

    # Check if checkpoint specifies blue model
    screen_color = "blue" if checkpoint and "blue" in checkpoint.lower() else "green"
    return create_engine(backend=backend, device=device, screen_color=screen_color)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="corridorkey.eval",
        description="CorridorKey Model-Quality Benchmark & Evaluation Framework",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint (.safetensors/.pth) or model name "
            "('CorridorKey_v1.0', 'CorridorKeyBlue_1.0', 'mock')."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tests/eval/fixtures",
        help="Directory containing benchmark cases or a single case directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to write JSON evaluation results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device ('cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "mlx"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--generate-fixtures",
        action="store_true",
        help="Generate canonical diagnostic fixtures if dataset directory is missing or empty.",
    )

    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists() or not any(dataset_path.iterdir()):
        if args.generate_fixtures or "fixtures" in str(dataset_path):
            print(f"Dataset path {dataset_path} empty or missing. Generating diagnostic fixtures...")
            create_diagnostic_fixtures(dataset_path)
        else:
            print(f"Error: Dataset directory {dataset_path} does not exist.", file=sys.stderr)
            return 1

    try:
        samples = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}", file=sys.stderr)
        return 1

    model_label = args.checkpoint or "CorridorKey_v1.0"
    print(f"Loaded {len(samples)} evaluation samples from {dataset_path}")
    print(f"Initializing engine ({model_label}) on {args.device}...")

    try:
        engine = _create_engine(args.checkpoint, device=args.device, backend=args.backend)
    except Exception as e:
        print(f"Failed to initialize engine for {model_label}: {e}", file=sys.stderr)
        return 1

    print("Running evaluation suite...")
    report = evaluate_dataset(engine, samples, model_name=model_label)

    # Print summary table
    print_report(report)

    # Write JSON report if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_report(report, str(out_path))
        print(f"Wrote structured evaluation receipt to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
