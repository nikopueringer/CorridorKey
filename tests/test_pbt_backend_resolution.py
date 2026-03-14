"""Property-based test for backend resolution priority chain.

# Feature: uv-lock-drift-fix, Property 1: Backend resolution priority chain

Validates: Requirements 5.1
"""

from __future__ import annotations

import os
from unittest import mock

from hypothesis import given, settings
from hypothesis import strategies as st

from CorridorKeyModule.backend import BACKEND_ENV_VAR, resolve_backend

# --- Strategies ---

# CLI --backend values: None means not provided, "auto" means explicit auto
cli_backend = st.sampled_from([None, "auto", "torch", "mlx"])

# Environment variable: None means unset, otherwise a string value
env_backend = st.sampled_from([None, "auto", "torch", "mlx"])

# Auto-detection result (what _auto_detect_backend would return)
auto_detect_result = st.sampled_from(["torch", "mlx"])


def _expected_backend(cli: str | None, env: str | None, auto: str) -> str:
    """Reference implementation of the priority chain.

    1. CLI value if explicit and not "auto"
    2. Env var if set and not "auto"
    3. Auto-detection result
    """
    if cli is not None and cli != "auto":
        return cli
    if env is not None and env != "auto":
        return env
    return auto


@settings(max_examples=200)
@given(cli=cli_backend, env=env_backend, auto=auto_detect_result)
def test_backend_resolution_priority_chain(cli: str | None, env: str | None, auto: str) -> None:
    """For any combination of CLI flag, env var, and auto-detection result,
    resolve_backend() returns the CLI value if explicit and not "auto",
    else the env var if set and not "auto", else the auto-detection result.

    **Validates: Requirements 5.1**
    """
    expected = _expected_backend(cli, env, auto)

    # Build the environment: set or unset CORRIDORKEY_BACKEND
    env_dict = {BACKEND_ENV_VAR: env} if env is not None else {}

    # We need to mock:
    # 1. The environment variable
    # 2. _auto_detect_backend() to return our generated auto value
    # 3. _validate_mlx_available() to avoid platform checks when mlx is selected
    with (
        mock.patch.dict(os.environ, env_dict, clear=False),
        mock.patch("CorridorKeyModule.backend._auto_detect_backend", return_value=auto),
        mock.patch("CorridorKeyModule.backend._validate_mlx_available"),
    ):
        # Ensure env var is unset when env is None
        if env is None:
            os.environ.pop(BACKEND_ENV_VAR, None)

        result = resolve_backend(cli)

    assert result == expected, f"cli={cli!r}, env={env!r}, auto={auto!r} → expected {expected!r}, got {result!r}"
