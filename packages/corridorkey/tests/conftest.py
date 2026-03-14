"""Pytest configuration - marker auto-skip logic and CLI flags."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption("--run-gpu", action="store_true", default=False, help="Run GPU tests")
    parser.addoption("--run-mlx", action="store_true", default=False, help="Run MLX tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-slow to run"))
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-gpu to run"))
        if "mlx" in item.keywords and not config.getoption("--run-mlx"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-mlx to run"))
