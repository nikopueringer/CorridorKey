"""Structural validation tests for pyproject.toml extras configuration.

Validates that the pyproject.toml correctly defines CUDA/MLX extras,
scoped index sources, and conflict groups to eliminate lockfile drift.
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

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    """Parse and return the pyproject.toml as a dict."""
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Requirement 2.1, 2.2: CUDA optional extra
# ---------------------------------------------------------------------------


class TestCudaExtra:
    """Validates: Requirements 1.2, 1.3, 2.1, 2.2, 2.3"""

    def test_cuda_extra_contains_torch(self, pyproject: dict) -> None:
        cuda_deps = pyproject["project"]["optional-dependencies"]["cuda"]
        assert "torch==2.8.0" in cuda_deps

    def test_cuda_extra_contains_torchvision(self, pyproject: dict) -> None:
        cuda_deps = pyproject["project"]["optional-dependencies"]["cuda"]
        assert "torchvision==0.23.0" in cuda_deps


# ---------------------------------------------------------------------------
# Requirement 3.1, 3.2, 3.3: MLX optional extra
# ---------------------------------------------------------------------------


class TestMlxExtra:
    """Validates: Requirements 3.1, 3.2, 3.3"""

    def test_mlx_extra_contains_corridorkey_mlx(self, pyproject: dict) -> None:
        mlx_deps = pyproject["project"]["optional-dependencies"]["mlx"]
        mlx_dep_names = [d.split(";")[0].strip() for d in mlx_deps]
        assert "corridorkey-mlx" in mlx_dep_names


# ---------------------------------------------------------------------------
# Requirement 2.3: pytorch index scoped to cuda extra
# ---------------------------------------------------------------------------


class TestPytorchIndex:
    """Validates: Requirements 2.3"""

    def test_pytorch_index_has_cuda_extra(self, pyproject: dict) -> None:
        indexes = pyproject["tool"]["uv"]["index"]
        pytorch_entries = [idx for idx in indexes if idx.get("name") == "pytorch"]
        assert len(pytorch_entries) == 1, "Expected exactly one pytorch index entry"
        assert pytorch_entries[0].get("extra") == "cuda"


# ---------------------------------------------------------------------------
# Requirement 2.1, 2.3: torch/torchvision source overrides scoped to cuda
# ---------------------------------------------------------------------------


class TestUvSources:
    """Validates: Requirements 1.3, 2.1, 2.3"""

    def test_torch_source_has_cuda_extra(self, pyproject: dict) -> None:
        sources = pyproject["tool"]["uv"]["sources"]
        torch_src = sources["torch"]
        cuda_entry = next(s for s in torch_src if s.get("extra") == "cuda")
        assert cuda_entry["index"] == "pytorch"

    def test_torchvision_source_has_cuda_extra(self, pyproject: dict) -> None:
        sources = pyproject["tool"]["uv"]["sources"]
        tv_src = sources["torchvision"]
        cuda_entry = next(s for s in tv_src if s.get("extra") == "cuda")
        assert cuda_entry["index"] == "pytorch"

    def test_torch_source_has_rocm_extra(self, pyproject: dict) -> None:
        sources = pyproject["tool"]["uv"]["sources"]
        torch_src = sources["torch"]
        rocm_entry = next(s for s in torch_src if s.get("extra") == "rocm")
        assert rocm_entry["index"] == "pytorch-rocm"

    def test_torchvision_source_has_rocm_extra(self, pyproject: dict) -> None:
        sources = pyproject["tool"]["uv"]["sources"]
        tv_src = sources["torchvision"]
        rocm_entry = next(s for s in tv_src if s.get("extra") == "rocm")
        assert rocm_entry["index"] == "pytorch-rocm"


# ---------------------------------------------------------------------------
# Requirement 4.1: Conflict group between cuda and mlx
# ---------------------------------------------------------------------------


class TestConflicts:
    """Validates: Requirements 4.1"""

    def test_cuda_mlx_conflict_declared(self, pyproject: dict) -> None:
        conflicts = pyproject["tool"]["uv"]["conflicts"]
        # conflicts is a list of conflict groups; each group is a list of dicts
        extras_in_groups = [
            {entry["extra"] for entry in group} for group in conflicts if all("extra" in entry for entry in group)
        ]
        assert {"cuda", "mlx", "rocm"} in extras_in_groups, (
            "Expected a conflict group containing 'cuda', 'mlx', and 'rocm' extras"
        )


# ---------------------------------------------------------------------------
# Requirement 6.2: timm git source override preserved
# ---------------------------------------------------------------------------


class TestTimmSourcePreserved:
    """Validates: Requirements 6.2"""

    def test_timm_source_is_git(self, pyproject: dict) -> None:
        timm_src = pyproject["tool"]["uv"]["sources"]["timm"]
        assert "git" in timm_src, "timm source should be a git override"

    def test_timm_git_url(self, pyproject: dict) -> None:
        timm_src = pyproject["tool"]["uv"]["sources"]["timm"]
        assert timm_src["git"] == "https://github.com/Raiden129/pytorch-image-models-fix"

    def test_timm_git_branch(self, pyproject: dict) -> None:
        timm_src = pyproject["tool"]["uv"]["sources"]["timm"]
        assert timm_src["branch"] == "fix/hiera-flash-attention-global-4d"


# ---------------------------------------------------------------------------
# Requirement 6.3: triton-windows platform-conditional dependency preserved
# ---------------------------------------------------------------------------


class TestTritonWindowsPreserved:
    """Validates: Requirements 6.3"""

    def test_triton_windows_in_base_deps(self, pyproject: dict) -> None:
        deps = pyproject["project"]["dependencies"]
        triton_entries = [d for d in deps if "triton-windows" in d]
        assert len(triton_entries) == 1, "Expected exactly one triton-windows dependency"

    def test_triton_windows_has_win32_marker(self, pyproject: dict) -> None:
        deps = pyproject["project"]["dependencies"]
        triton_entries = [d for d in deps if "triton-windows" in d]
        assert "sys_platform == 'win32'" in triton_entries[0]


# ---------------------------------------------------------------------------
# Requirement 7.1: dev dependency group preserved
# ---------------------------------------------------------------------------


class TestDevDependencyGroup:
    """Validates: Requirements 7.1"""

    def test_dev_group_contains_pytest(self, pyproject: dict) -> None:
        dev = pyproject["dependency-groups"]["dev"]
        assert "pytest" in dev

    def test_dev_group_contains_pytest_cov(self, pyproject: dict) -> None:
        dev = pyproject["dependency-groups"]["dev"]
        assert "pytest-cov" in dev

    def test_dev_group_contains_ruff(self, pyproject: dict) -> None:
        dev = pyproject["dependency-groups"]["dev"]
        assert "ruff" in dev


# ---------------------------------------------------------------------------
# Requirement 7.2: docs dependency group preserved
# ---------------------------------------------------------------------------


class TestDocsDependencyGroup:
    """Validates: Requirements 7.2"""

    def test_docs_group_contains_zensical(self, pyproject: dict) -> None:
        docs = pyproject["dependency-groups"]["docs"]
        zensical_entries = [d for d in docs if "zensical" in d]
        assert len(zensical_entries) == 1
        assert "zensical>=0.0.24" in zensical_entries[0]
