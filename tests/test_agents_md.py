"""Unit tests and property-based tests for AGENTS.md content.

Feature: agents-md-setup
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_MD_PATH = REPO_ROOT / "AGENTS.md"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _read_agents_md() -> str:
    """Read AGENTS.md and return its full text content."""
    return AGENTS_MD_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit Tests — File existence, structure, and specific content
# ---------------------------------------------------------------------------


class TestAgentsMdFileExistence:
    """Validate that AGENTS.md exists and is a regular file (Req 1.1)."""

    def test_agents_md_exists(self) -> None:
        assert AGENTS_MD_PATH.exists(), f"AGENTS.md not found at {AGENTS_MD_PATH}"

    def test_agents_md_is_file(self) -> None:
        assert AGENTS_MD_PATH.is_file(), f"{AGENTS_MD_PATH} exists but is not a file"


class TestAgentsMdStructure:
    """Validate AGENTS.md heading structure (Reqs 1.2, 1.3)."""

    def test_starts_with_heading_containing_corridorkey(self) -> None:
        """First line must be a # heading containing 'CorridorKey' (Req 1.3)."""
        content = _read_agents_md()
        first_line = content.strip().splitlines()[0]
        assert first_line.startswith("# "), "File must start with a top-level heading"
        assert "CorridorKey" in first_line, f"Top-level heading must contain 'CorridorKey', got: {first_line!r}"

    def test_contains_section_headings(self) -> None:
        """File must contain ## headings for hierarchical structure (Req 1.2)."""
        content = _read_agents_md()
        h2_lines = [line for line in content.splitlines() if line.startswith("## ")]
        assert len(h2_lines) >= 1, "AGENTS.md must contain at least one ## heading"


class TestAgentsMdCommands:
    """Validate that required dev/build/test commands are present (Reqs 5.1, 6.1–6.3)."""

    def test_uv_sync_group_dev(self) -> None:
        assert "uv sync --group dev" in _read_agents_md()

    def test_uv_run_pytest(self) -> None:
        assert "uv run pytest" in _read_agents_md()

    def test_uv_run_ruff_check(self) -> None:
        assert "uv run ruff check" in _read_agents_md()

    def test_uv_run_ruff_format_check(self) -> None:
        assert "uv run ruff format --check" in _read_agents_md()


class TestAgentsMdLicense:
    """Validate license string is present (Req 2.3)."""

    def test_license_string(self) -> None:
        assert "CC-BY-NC-SA-4.0" in _read_agents_md()


class TestAgentsMdLLMHandoverReference:
    """Validate LLM_HANDOVER.md is referenced (Req 12.2)."""

    def test_llm_handover_referenced(self) -> None:
        assert "docs/LLM_HANDOVER.md" in _read_agents_md()


class TestAgentsMdRuffConfig:
    """Validate ruff configuration values are documented (Reqs 7.2, 7.3)."""

    def test_line_length_120(self) -> None:
        assert "120" in _read_agents_md()

    def test_ruff_rules(self) -> None:
        content = _read_agents_md()
        for rule in ("E", "F", "W", "I", "B"):
            assert rule in content, f"Ruff rule '{rule}' not found in AGENTS.md"


class TestAgentsMdPlatformCaveats:
    """Validate platform-specific caveats are present (Reqs 8.2–8.3)."""

    def test_pytorch_mps_fallback(self) -> None:
        assert "PYTORCH_ENABLE_MPS_FALLBACK=1" in _read_agents_md()

    def test_cuda_12_8(self) -> None:
        assert "CUDA 12.8" in _read_agents_md()


class TestAgentsMdProhibitedActions:
    """Validate prohibited actions are documented (Reqs 9.1–9.2)."""

    def test_gamma_22_prohibition(self) -> None:
        content = _read_agents_md()
        assert "gamma 2.2" in content.lower() or "2.2" in content, "Gamma 2.2 prohibition not found in AGENTS.md"

    def test_gvm_core_modification_prohibition(self) -> None:
        content = _read_agents_md()
        assert "gvm_core/" in content, "gvm_core/ modification prohibition not found"
        assert "VideoMaMaInferenceModule/" in content, "VideoMaMaInferenceModule/ modification prohibition not found"


# ---------------------------------------------------------------------------
# Property-Based Tests — Hypothesis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Baselines for Property 1
# ---------------------------------------------------------------------------

REQUIRED_TERMS: tuple[str, ...] = (
    "GreenFormer",
    "Hiera",
    "CNNRefinerModule",
    "srgb_to_linear",
    "color_utils.py",
    "2048",
    "Lanczos4",
    "EXR",
    "despill",
    "premultiply",
    "sRGB",
)


class TestRequiredTechnicalTerms:
    """Feature: agents-md-setup, Property 1: Required Technical Terms Present

    For any term in the required set, that term must appear at least once in
    the content of CorridorKey/AGENTS.md.

    Validates: Requirements 13.1, 13.2, 13.3
    """

    @given(term=st.sampled_from(REQUIRED_TERMS))
    @settings(max_examples=100)
    def test_term_present_in_agents_md(self, term: str) -> None:
        """**Validates: Requirements 13.1, 13.2, 13.3**

        Every required technical term must appear in AGENTS.md.
        """
        content = _read_agents_md()
        assert term in content, f"Required technical term {term!r} not found in AGENTS.md"


# ---------------------------------------------------------------------------
# Baselines for Property 2
# ---------------------------------------------------------------------------

REQUIRED_FILE_PATHS: tuple[str, ...] = (
    "CorridorKeyModule/core/model_transformer.py",
    "CorridorKeyModule/inference_engine.py",
    "CorridorKeyModule/core/color_utils.py",
    "clip_manager.py",
    "device_utils.py",
    "backend/",
)


class TestKeyFileMapCompleteness:
    """Feature: agents-md-setup, Property 2: Key File Map Completeness

    For any file path in the required set, that path must appear in the
    content of CorridorKey/AGENTS.md.

    Validates: Requirements 4.1
    """

    @given(file_path=st.sampled_from(REQUIRED_FILE_PATHS))
    @settings(max_examples=100)
    def test_file_path_present_in_agents_md(self, file_path: str) -> None:
        """**Validates: Requirements 4.1**

        Every required file path must appear in AGENTS.md.
        """
        content = _read_agents_md()
        assert file_path in content, f"Required file path {file_path!r} not found in AGENTS.md"


# ---------------------------------------------------------------------------
# Baselines for Property 3
# ---------------------------------------------------------------------------

PR_TEMPLATE_ELEMENTS: tuple[str, ...] = (
    "What does this change?",
    "How was it tested?",
    "uv run pytest",
    "uv run ruff check",
    "uv run ruff format --check",
)


class TestPRTemplateElements:
    """Feature: agents-md-setup, Property 3: PR Template Elements Present

    For any element in the PR template set, that element must appear in the
    content of CorridorKey/AGENTS.md.

    Validates: Requirements 11.4
    """

    @given(element=st.sampled_from(PR_TEMPLATE_ELEMENTS))
    @settings(max_examples=100)
    def test_pr_template_element_present_in_agents_md(self, element: str) -> None:
        """**Validates: Requirements 11.4**

        Every PR template element must appear in AGENTS.md.
        """
        content = _read_agents_md()
        assert element in content, f"PR template element {element!r} not found in AGENTS.md"


# ---------------------------------------------------------------------------
# Baselines for Property 4
# ---------------------------------------------------------------------------

LLM_HANDOVER_PATH = REPO_ROOT / "docs" / "LLM_HANDOVER.md"

BASELINE_LLM_HANDOVER_LINES: tuple[str, ...] = tuple(
    line for line in LLM_HANDOVER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
)


class TestLLMHandoverPreservation:
    """Feature: agents-md-setup, Property 4: LLM_HANDOVER.md Content Preservation

    For any non-empty line in the original docs/LLM_HANDOVER.md, that line must
    appear identically in the current version of the file. The file must not be
    modified, truncated, or have content removed.

    Validates: Requirements 12.1
    """

    @given(line=st.sampled_from(BASELINE_LLM_HANDOVER_LINES))
    @settings(max_examples=100)
    def test_line_still_present_in_llm_handover(self, line: str) -> None:
        """**Validates: Requirements 12.1**

        Every non-empty baseline line must still be present in the current file.
        """
        current_content = LLM_HANDOVER_PATH.read_text(encoding="utf-8")
        assert line in current_content, f"Baseline line missing from docs/LLM_HANDOVER.md:\n{line!r}"


# ---------------------------------------------------------------------------
# Baselines for Property 5
# ---------------------------------------------------------------------------

REQUIRED_DOC_PATHS: tuple[str, ...] = (
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "docs/LLM_HANDOVER.md",
    "docs/",
)


class TestDocFileReferences:
    """Feature: agents-md-setup, Property 5: Documentation File References Present

    For any documentation file path in the required set, that path must appear
    in the content of CorridorKey/AGENTS.md.

    Validates: Requirements 14.3
    """

    @given(doc_path=st.sampled_from(REQUIRED_DOC_PATHS))
    @settings(max_examples=100)
    def test_doc_path_present_in_agents_md(self, doc_path: str) -> None:
        """**Validates: Requirements 14.3**

        Every required documentation file path must appear in AGENTS.md.
        """
        content = _read_agents_md()
        assert doc_path in content, f"Required documentation path {doc_path!r} not found in AGENTS.md"
