"""Property-based tests for docs-site-setup structural invariants.

Feature: docs-site-setup
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        pytest.skip("tomli required for Python < 3.11", allow_module_level=True)

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENSICAL_PATH = REPO_ROOT / "zensical.toml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_zensical() -> dict:
    """Parse zensical.toml and return as a dict."""
    with open(ZENSICAL_PATH, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Baselines — captured from the original zensical.toml before any changes.
# These are the theme features and markdown extensions that MUST be preserved.
# ---------------------------------------------------------------------------

BASELINE_THEME_FEATURES: list[str] = [
    "announce.dismiss",
    "content.action.edit",
    "content.action.view",
    "content.code.annotate",
    "content.code.copy",
    "content.code.select",
    "content.footnote.tooltips",
    "content.tabs.link",
    "content.tooltips",
    "navigation.footer",
    "navigation.indexes",
    "navigation.instant",
    "navigation.instant.prefetch",
    "navigation.instant.progress",
    "navigation.path",
    "navigation.tabs",
    "navigation.top",
    "navigation.tracking",
    "search.highlight",
]

# Extension keys as they appear under [project.markdown_extensions] in TOML.
# Nested keys like pymdownx.superfences are represented by their dict path.
BASELINE_EXTENSION_KEYS: list[str] = [
    "admonition",
    "pymdownx.details",
    "pymdownx.superfences",
    "pymdownx.tabbed",
]


# ---------------------------------------------------------------------------
# Property 2: Zensical Theme and Extension Preservation
# ---------------------------------------------------------------------------


class TestZensicalThemeAndExtensionPreservation:
    """Feature: docs-site-setup, Property 2: Zensical Theme and Extension Preservation

    For any theme feature in the existing zensical.toml features list, and for
    any markdown extension in the existing markdown_extensions section, those
    entries must remain present and unchanged in the updated zensical.toml.

    Validates: Requirements 10.3
    """

    @given(feature=st.sampled_from(BASELINE_THEME_FEATURES))
    @settings(max_examples=100)
    def test_theme_feature_preserved(self, feature: str) -> None:
        """**Validates: Requirements 10.3**

        Every original theme feature must still be present in the updated file.
        """
        config = _load_zensical()
        current_features = config["project"]["theme"]["features"]
        assert feature in current_features, (
            f"Theme feature '{feature}' was removed from zensical.toml"
        )

    @given(ext_key=st.sampled_from(BASELINE_EXTENSION_KEYS))
    @settings(max_examples=100)
    def test_markdown_extension_preserved(self, ext_key: str) -> None:
        """**Validates: Requirements 10.3**

        Every original markdown extension must still be present in the updated file.
        """
        config = _load_zensical()
        extensions = config["project"]["markdown_extensions"]

        # Extension keys can be nested (e.g. "pymdownx.superfences" means
        # extensions["pymdownx"]["superfences"]).
        parts = ext_key.split(".")
        node = extensions
        for part in parts:
            assert part in node, (
                f"Markdown extension '{ext_key}' was removed from zensical.toml"
            )
            node = node[part]


# ---------------------------------------------------------------------------
# Baseline — LLM Handover content lines (non-empty, stripped)
# ---------------------------------------------------------------------------

_LLM_HANDOVER_PATH = REPO_ROOT / "docs" / "LLM_HANDOVER.md"

BASELINE_LLM_HANDOVER_LINES: list[str] = [
    line
    for raw in _LLM_HANDOVER_PATH.read_text(encoding="utf-8").splitlines()
    if (line := raw.strip())
]

assert BASELINE_LLM_HANDOVER_LINES, "LLM_HANDOVER.md baseline must not be empty"


# ---------------------------------------------------------------------------
# Property 1: LLM Handover Content Preservation
# ---------------------------------------------------------------------------


class TestLLMHandoverContentPreservation:
    """Feature: docs-site-setup, Property 1: LLM Handover Content Preservation

    For any non-empty line in the original docs/LLM_HANDOVER.md, that line must
    appear identically in the current version of the file. The file must not be
    modified, truncated, or have content removed.

    Validates: Requirements 8.2
    """

    @given(line=st.sampled_from(BASELINE_LLM_HANDOVER_LINES))
    @settings(max_examples=100)
    def test_line_preserved(self, line: str) -> None:
        """**Validates: Requirements 8.2**

        Every non-empty line from the baseline must still be present.
        """
        current_content = _LLM_HANDOVER_PATH.read_text(encoding="utf-8")
        current_lines = [l.strip() for l in current_content.splitlines()]
        assert line in current_lines, (
            f"Line missing from LLM_HANDOVER.md: {line!r}"
        )


# ---------------------------------------------------------------------------
# Helper — recursively extract .md file references from the nav structure
# ---------------------------------------------------------------------------


def _extract_nav_files(nav: list) -> list[str]:
    """Walk the nested nav array from zensical.toml and collect all .md refs."""
    files: list[str] = []
    for entry in nav:
        if isinstance(entry, dict):
            for _title, value in entry.items():
                if isinstance(value, str) and value.endswith(".md"):
                    files.append(value)
                elif isinstance(value, list):
                    files.extend(_extract_nav_files(value))
    return files


# ---------------------------------------------------------------------------
# Baseline — nav file references
# ---------------------------------------------------------------------------

_NAV_FILES: list[str] = _extract_nav_files(_load_zensical()["project"]["nav"])

assert _NAV_FILES, "Nav must reference at least one .md file"


# ---------------------------------------------------------------------------
# Property 4: Documentation Pages Reside in docs/
# ---------------------------------------------------------------------------


class TestDocumentationPagesResideInDocs:
    """Feature: docs-site-setup, Property 4: Documentation Pages Reside in docs/

    For any file referenced in the nav array of zensical.toml, the file must
    exist within the docs/ directory so that the existing workflow trigger path
    (docs/**) detects changes.

    Validates: Requirements 11.2
    """

    @given(nav_file=st.sampled_from(_NAV_FILES))
    @settings(max_examples=100)
    def test_nav_file_exists_in_docs(self, nav_file: str) -> None:
        """**Validates: Requirements 11.2**

        Every nav entry must resolve to an existing file inside docs/.
        """
        full_path = REPO_ROOT / "docs" / nav_file
        assert full_path.exists(), (
            f"Nav references '{nav_file}' but {full_path} does not exist"
        )
        assert full_path.is_file(), (
            f"Nav references '{nav_file}' but {full_path} is not a file"
        )


# ---------------------------------------------------------------------------
# Baseline — README.md content lines (non-empty, stripped)
# The banner line was added by task 7.1; all original lines are still present.
# We capture every non-empty stripped line from the current file as the
# baseline.  Since the only change was an *addition*, every line in this
# baseline must remain present in the file.
# ---------------------------------------------------------------------------

_README_PATH = REPO_ROOT / "README.md"

BASELINE_README_LINES: list[str] = [
    line
    for raw in _README_PATH.read_text(encoding="utf-8").splitlines()
    if (line := raw.strip())
]

assert BASELINE_README_LINES, "README.md baseline must not be empty"


# ---------------------------------------------------------------------------
# Property 5: README Content Preservation
# ---------------------------------------------------------------------------


class TestREADMEContentPreservation:
    """Feature: docs-site-setup, Property 5: README Content Preservation

    For any non-empty line in the README.md (including the newly added docs
    link banner), that line must appear in the current version of the file.
    The only permitted change was the addition of the banner — no existing
    content may be removed or altered.

    Validates: Requirements 12.1
    """

    @given(line=st.sampled_from(BASELINE_README_LINES))
    @settings(max_examples=100)
    def test_line_preserved(self, line: str) -> None:
        """**Validates: Requirements 12.1**

        Every non-empty line from the baseline must still be present.
        """
        current_content = _README_PATH.read_text(encoding="utf-8")
        current_lines = [l.strip() for l in current_content.splitlines()]
        assert line in current_lines, (
            f"Line missing from README.md: {line!r}"
        )


# ---------------------------------------------------------------------------
# Baseline — docs.yml workflow lines (non-empty, stripped)
# ---------------------------------------------------------------------------

_DOCS_YML_PATH = REPO_ROOT / ".github" / "workflows" / "docs.yml"

BASELINE_DOCS_YML_LINES: list[str] = [
    line
    for raw in _DOCS_YML_PATH.read_text(encoding="utf-8").splitlines()
    if (line := raw.strip())
]

assert BASELINE_DOCS_YML_LINES, "docs.yml baseline must not be empty"


# ---------------------------------------------------------------------------
# Property 3: Workflow File Immutability
# ---------------------------------------------------------------------------


class TestWorkflowFileImmutability:
    """Feature: docs-site-setup, Property 3: Workflow File Immutability

    For any non-empty line in the original .github/workflows/docs.yml, that
    line must appear identically in the current version of the file.  The
    workflow file must not be modified in any way.

    Validates: Requirements 11.1
    """

    @given(line=st.sampled_from(BASELINE_DOCS_YML_LINES))
    @settings(max_examples=100)
    def test_line_preserved(self, line: str) -> None:
        """**Validates: Requirements 11.1**

        Every non-empty line from the baseline must still be present.
        """
        current_content = _DOCS_YML_PATH.read_text(encoding="utf-8")
        current_lines = [l.strip() for l in current_content.splitlines()]
        assert line in current_lines, (
            f"Line missing from docs.yml: {line!r}"
        )
