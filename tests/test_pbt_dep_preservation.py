"""Property-based test for non-torch dependency preservation.

# Feature: uv-lock-drift-fix, Property 2: Non-torch dependency preservation

Validates: Requirements 6.1, 6.2, 6.3
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

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Known original non-torch, non-torchvision base dependencies from the design doc.
# These are the dependencies that MUST be preserved after the restructuring.
KNOWN_ORIGINAL_NON_TORCH_DEPS: list[str] = [
    "timm==1.0.24",
    "numpy",
    "opencv-python",
    "tqdm",
    "setuptools",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "av",
    "Pillow",
    "PIMS",
    "easydict",
    "imageio",
    "matplotlib",
    "einops",
    "huggingface-hub",
    "typer>=0.12",
    "rich>=13",
]


def _parse_base_dependencies() -> list[str]:
    """Parse the current pyproject.toml and return the base dependencies list."""
    with open(PYPROJECT_PATH, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["dependencies"]


def _normalize_dep(dep: str) -> str:
    """Normalize a dependency string for comparison.

    Lowercases the package name portion (before any version specifier or marker),
    strips whitespace, and normalizes underscores to hyphens in the package name.
    """
    return dep.strip().lower().replace("_", "-")


def _dep_present(dep: str, dep_list: list[str]) -> bool:
    """Check if a dependency string is present in a list, using normalized comparison."""
    normalized = _normalize_dep(dep)
    return any(_normalize_dep(d) == normalized for d in dep_list)


# --- Property-based test ---


@settings(max_examples=200)
@given(dep=st.sampled_from(KNOWN_ORIGINAL_NON_TORCH_DEPS))
def test_non_torch_dependency_preserved(dep: str) -> None:
    """For any non-torch, non-torchvision dependency from the original base
    dependencies list, that dependency (with identical name and version
    constraint) shall appear in the updated pyproject.toml base dependencies.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """
    current_deps = _parse_base_dependencies()
    assert _dep_present(dep, current_deps), (
        f"Dependency {dep!r} from the original pyproject.toml is missing "
        f"or has a different version constraint in the current base dependencies.\n"
        f"Current dependencies: {current_deps}"
    )
