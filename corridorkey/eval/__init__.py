"""Model-quality evaluation and benchmark framework for CorridorKey."""

from .cases import MattingEvalSample, load_dataset, load_sample_from_directory
from .fixtures import FIXTURE_GENERATORS, create_diagnostic_fixtures
from .hint_corruption import (
    CANONICAL_HINT_PROFILES,
    HintCorruptionSpec,
    HintProfileResult,
    HintSweepReport,
    compute_asymmetric_error,
    compute_hint_iou,
    corrupt_alpha,
    run_hint_sweep,
)
from .metrics import (
    compute_alpha_mae,
    compute_boundary_alpha_mae,
    compute_boundary_mask,
    compute_foreground_mae,
    compute_recomposition_mae,
)
from .report import format_json_report, format_text_table, print_report, write_json_report
from .runner import CaseResult, EvaluationReport, evaluate_dataset, evaluate_sample

__all__ = [
    "MattingEvalSample",
    "load_dataset",
    "load_sample_from_directory",
    "compute_alpha_mae",
    "compute_foreground_mae",
    "compute_boundary_mask",
    "compute_boundary_alpha_mae",
    "compute_recomposition_mae",
    "create_diagnostic_fixtures",
    "FIXTURE_GENERATORS",
    "CaseResult",
    "EvaluationReport",
    "evaluate_sample",
    "evaluate_dataset",
    "format_json_report",
    "format_text_table",
    "write_json_report",
    "print_report",
    "HintCorruptionSpec",
    "corrupt_alpha",
    "compute_hint_iou",
    "compute_asymmetric_error",
    "CANONICAL_HINT_PROFILES",
    "HintProfileResult",
    "HintSweepReport",
    "run_hint_sweep",
]
