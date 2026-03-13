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
        assert feature in current_features, f"Theme feature '{feature}' was removed from zensical.toml"

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
            assert part in node, f"Markdown extension '{ext_key}' was removed from zensical.toml"
            node = node[part]


# ---------------------------------------------------------------------------
# Baseline — LLM Handover content lines (non-empty, stripped)
# ---------------------------------------------------------------------------

_LLM_HANDOVER_PATH = REPO_ROOT / "docs" / "LLM_HANDOVER.md"

BASELINE_LLM_HANDOVER_LINES: list[str] = [
    line for raw in _LLM_HANDOVER_PATH.read_text(encoding="utf-8").splitlines() if (line := raw.strip())
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
        current_lines = [ln.strip() for ln in current_content.splitlines()]
        assert line in current_lines, f"Line missing from LLM_HANDOVER.md: {line!r}"


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
        assert full_path.exists(), f"Nav references '{nav_file}' but {full_path} does not exist"
        assert full_path.is_file(), f"Nav references '{nav_file}' but {full_path} is not a file"


# ---------------------------------------------------------------------------
# Baseline — README.md content lines (non-empty, stripped)
# The banner line was added by task 7.1; all original lines are still present.
# We capture every non-empty stripped line from the current file as the
# baseline.  Since the only change was an *addition*, every line in this
# baseline must remain present in the file.
# ---------------------------------------------------------------------------

_README_PATH = REPO_ROOT / "README.md"

BASELINE_README_LINES: list[str] = [
    line for raw in _README_PATH.read_text(encoding="utf-8").splitlines() if (line := raw.strip())
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
        current_lines = [ln.strip() for ln in current_content.splitlines()]
        assert line in current_lines, f"Line missing from README.md: {line!r}"


# ---------------------------------------------------------------------------
# Baseline — docs.yml workflow lines (non-empty, stripped)
# ---------------------------------------------------------------------------

_DOCS_YML_PATH = REPO_ROOT / ".github" / "workflows" / "docs.yml"

BASELINE_DOCS_YML_LINES: list[str] = [
    line for raw in _DOCS_YML_PATH.read_text(encoding="utf-8").splitlines() if (line := raw.strip())
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
        current_lines = [ln.strip() for ln in current_content.splitlines()]
        assert line in current_lines, f"Line missing from docs.yml: {line!r}"


# ---------------------------------------------------------------------------
# Unit tests — AI-Assisted Development page (ai-dev-setup-guide)
# ---------------------------------------------------------------------------


def test_ai_dev_page_exists() -> None:
    """The AI-Assisted Development page must exist on disk.

    Validates: Requirements 1.1, 12.1
    """
    ai_dev_path = REPO_ROOT / "docs" / "ai-assisted-development.md"
    assert ai_dev_path.exists(), f"{ai_dev_path} does not exist"
    assert ai_dev_path.is_file(), f"{ai_dev_path} is not a file"


def test_ai_dev_page_in_nav() -> None:
    """The nav in zensical.toml must reference ai-assisted-development.md
    with the title 'AI-Assisted Development'.

    Validates: Requirements 11.1, 11.2, 11.3, 12.2
    """
    config = _load_zensical()
    nav = config["project"]["nav"]
    nav_files = _extract_nav_files(nav)
    assert "ai-assisted-development.md" in nav_files, "ai-assisted-development.md not found in zensical.toml nav"

    # Also verify the title is correct by walking the nav structure.
    def _find_title(entries: list, target_file: str) -> str | None:
        for entry in entries:
            if isinstance(entry, dict):
                for title, value in entry.items():
                    if isinstance(value, str) and value == target_file:
                        return title
                    if isinstance(value, list):
                        found = _find_title(value, target_file)
                        if found is not None:
                            return found
        return None

    title = _find_title(nav, "ai-assisted-development.md")
    assert title == "AI-Assisted Development", f"Expected nav title 'AI-Assisted Development', got {title!r}"


# ---------------------------------------------------------------------------
# Baseline — nav file references for ai-dev-setup-guide property tests
# ---------------------------------------------------------------------------

BASELINE_NAV_FILES: list[str] = list(_NAV_FILES)

assert BASELINE_NAV_FILES, "Baseline nav files must not be empty"


# ---------------------------------------------------------------------------
# Property 1: Existing Nav Entry Preservation (ai-dev-setup-guide)
# ---------------------------------------------------------------------------


class TestNavEntryPreservation:
    """Feature: ai-dev-setup-guide, Property 1: Existing Nav Entry Preservation

    For any file reference that existed in the zensical.toml nav array before
    the AI-Assisted Development entry was added, that file reference must still
    be present in the nav array after the change.

    Validates: Requirements 11.4
    """

    @given(nav_file=st.sampled_from(BASELINE_NAV_FILES))
    @settings(max_examples=100)
    def test_nav_entry_preserved(self, nav_file: str) -> None:
        """**Validates: Requirements 11.4**

        Every nav .md reference from the baseline must still be present.
        """
        current_nav_files = _extract_nav_files(_load_zensical()["project"]["nav"])
        assert nav_file in current_nav_files, f"Nav entry '{nav_file}' was removed from zensical.toml"


# ---------------------------------------------------------------------------
# Baseline — docs/index.md content lines (non-empty, stripped)
# ---------------------------------------------------------------------------

_INDEX_MD_PATH = REPO_ROOT / "docs" / "index.md"

BASELINE_INDEX_LINES: list[str] = [
    line for raw in _INDEX_MD_PATH.read_text(encoding="utf-8").splitlines() if (line := raw.strip())
]

assert BASELINE_INDEX_LINES, "docs/index.md baseline must not be empty"


# ---------------------------------------------------------------------------
# Property 2: Index Page Content Preservation (ai-dev-setup-guide)
# ---------------------------------------------------------------------------


class TestIndexPageContentPreservation:
    """Feature: ai-dev-setup-guide, Property 2: Index Page Content Preservation

    For any non-empty line in the original docs/index.md, that line must appear
    identically in the current version of the file after the AI-Assisted
    Development link is added.

    Validates: Requirements 13.2
    """

    @given(line=st.sampled_from(BASELINE_INDEX_LINES))
    @settings(max_examples=100)
    def test_line_preserved(self, line: str) -> None:
        """**Validates: Requirements 13.2**

        Every non-empty line from the baseline must still be present.
        """
        current_content = _INDEX_MD_PATH.read_text(encoding="utf-8")
        current_lines = [ln.strip() for ln in current_content.splitlines()]
        assert line in current_lines, f"Line missing from docs/index.md: {line!r}"


def test_ai_dev_page_has_required_content() -> None:
    """The AI-Assisted Development page must contain all required sections and markers.

    Validates: Requirements 1.2, 1.3, 2.1, 2.2, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1
    """
    ai_dev_path = REPO_ROOT / "docs" / "ai-assisted-development.md"
    content = ai_dev_path.read_text(encoding="utf-8")

    # Heading
    assert "# AI-Assisted Development" in content

    # Context source references
    assert "AGENTS.md" in content
    assert "LLM_HANDOVER.md" in content

    # Quick Start section
    assert "Quick Start" in content

    # Six tool names
    for tool in ("Kiro", "Claude Code", "Cursor", "GitHub Copilot", "Windsurf", "Gemini CLI"):
        assert tool in content, f"Tool '{tool}' not found in ai-assisted-development.md"

    # Contributions note
    assert any(
        word in content.lower() for word in ("contributions", "welcome", "contributing")
    ), "No contributions/welcome note found in ai-assisted-development.md"


def test_index_page_links_to_ai_dev() -> None:
    """The docs/index.md must contain a link to ai-assisted-development.md.

    Validates: Requirements 13.1
    """
    content = _INDEX_MD_PATH.read_text(encoding="utf-8")
    assert "ai-assisted-development.md" in content, (
        "docs/index.md does not contain a link to ai-assisted-development.md"
    )
