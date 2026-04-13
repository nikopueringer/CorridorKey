"""Unit tests for device_utils — cross-platform device selection.

Tests cover all code paths in detect_best_device(), resolve_device(),
clear_device_cache(), and enumerate_gpus() using monkeypatch to mock
hardware availability. No GPU required.
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
import torch

from device_utils import (
    DEVICE_ENV_VAR,
    GPUInfo,
    _enumerate_amd,
    _enumerate_nvidia,
    clear_device_cache,
    detect_best_device,
    enumerate_gpus,
    resolve_device,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_gpu(monkeypatch, *, cuda=False, mps=False):
    """Mock CUDA and MPS availability flags."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    # MPS lives behind torch.backends.mps; ensure the attr path exists
    mps_backend = MagicMock()
    mps_backend.is_available = MagicMock(return_value=mps)
    monkeypatch.setattr(torch.backends, "mps", mps_backend)


# ---------------------------------------------------------------------------
# detect_best_device
# ---------------------------------------------------------------------------


class TestDetectBestDevice:
    """Priority chain: CUDA > MPS > CPU."""

    def test_returns_cuda_when_available(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=True, mps=True)
        assert detect_best_device() == "cuda"

    def test_returns_mps_when_no_cuda(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=False, mps=True)
        assert detect_best_device() == "mps"

    def test_returns_cpu_when_nothing(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=False, mps=False)
        assert detect_best_device() == "cpu"


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    """Priority chain: CLI arg > env var > auto-detect."""

    # --- auto-detect path ---

    def test_none_triggers_auto_detect(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=False, mps=False)
        monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
        assert resolve_device(None) == "cpu"

    def test_auto_string_triggers_auto_detect(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=True)
        monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
        assert resolve_device("auto") == "cuda"

    # --- env var fallback ---

    def test_env_var_used_when_no_cli_arg(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=True, mps=True)
        monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
        assert resolve_device(None) == "cpu"

    def test_env_var_auto_triggers_detect(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=False, mps=True)
        monkeypatch.setenv(DEVICE_ENV_VAR, "auto")
        assert resolve_device(None) == "mps"

    # --- CLI arg overrides env var ---

    def test_cli_arg_overrides_env_var(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=True, mps=True)
        monkeypatch.setenv(DEVICE_ENV_VAR, "mps")
        assert resolve_device("cuda") == "cuda"

    # --- explicit valid devices ---

    def test_explicit_cuda(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=True)
        assert resolve_device("cuda") == "cuda"

    def test_explicit_mps(self, monkeypatch):
        _patch_gpu(monkeypatch, mps=True)
        assert resolve_device("mps") == "mps"

    def test_explicit_cpu(self, monkeypatch):
        assert resolve_device("cpu") == "cpu"

    def test_case_insensitive(self, monkeypatch):
        assert resolve_device("CPU") == "cpu"

    # --- unavailable backend errors ---

    def test_cuda_unavailable_raises(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=False)
        with pytest.raises(RuntimeError, match="CUDA requested"):
            resolve_device("cuda")

    def test_mps_no_backend_raises(self, monkeypatch):
        # Simulate PyTorch build without MPS module in torch.backends
        _patch_gpu(monkeypatch, cuda=False, mps=False)
        # Replace torch.backends with an object that lacks "mps" entirely
        fake_backends = type("Backends", (), {})()
        monkeypatch.setattr(torch, "backends", fake_backends)
        with pytest.raises(RuntimeError, match="no MPS support"):
            resolve_device("mps")

    def test_mps_unavailable_raises(self, monkeypatch):
        _patch_gpu(monkeypatch, cuda=False, mps=False)
        with pytest.raises(RuntimeError, match="not available on this machine"):
            resolve_device("mps")

    # --- invalid device string ---

    def test_invalid_device_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="Unknown device"):
            resolve_device("tpu")


# ---------------------------------------------------------------------------
# clear_device_cache
# ---------------------------------------------------------------------------


class TestClearDeviceCache:
    """Dispatches to correct backend cache clear."""

    def test_cuda_clears_cache(self, monkeypatch):
        mock_empty = MagicMock()
        monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty)
        clear_device_cache("cuda")
        mock_empty.assert_called_once()

    def test_mps_clears_cache(self, monkeypatch):
        mock_empty = MagicMock()
        monkeypatch.setattr(torch.mps, "empty_cache", mock_empty)
        clear_device_cache("mps")
        mock_empty.assert_called_once()

    def test_cpu_is_noop(self):
        # Should not raise
        clear_device_cache("cpu")

    def test_accepts_torch_device_object(self, monkeypatch):
        mock_empty = MagicMock()
        monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty)
        clear_device_cache(torch.device("cuda"))
        mock_empty.assert_called_once()

    def test_accepts_mps_device_object(self, monkeypatch):
        mock_empty = MagicMock()
        monkeypatch.setattr(torch.mps, "empty_cache", mock_empty)
        clear_device_cache(torch.device("mps"))
        mock_empty.assert_called_once()


# ---------------------------------------------------------------------------
# enumerate_gpus helpers — subprocess mocks shared across NVIDIA/AMD tests
# ---------------------------------------------------------------------------


def _fake_completed(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Build a minimal stand-in for subprocess.CompletedProcess."""
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


class _SubprocessRouter:
    """subprocess.run stub that dispatches per-binary.

    ``responses`` maps the binary name (cmd[0]) to either a
    ``(returncode, stdout)`` tuple or an ``Exception`` to raise.
    Missing binaries raise ``FileNotFoundError`` to mimic the OS.
    """

    def __init__(self, responses: dict):
        self._responses = responses

    def __call__(self, cmd, *args, **kwargs):
        response = self._responses.get(cmd[0])
        if response is None:
            raise FileNotFoundError(cmd[0])
        if isinstance(response, BaseException):
            raise response
        returncode, stdout = response
        return _fake_completed(stdout=stdout, returncode=returncode)


# ---------------------------------------------------------------------------
# _enumerate_nvidia
# ---------------------------------------------------------------------------


class TestEnumerateNvidia:
    """nvidia-smi CSV parser — returns a list on success, None when unavailable."""

    def test_parses_single_gpu(self, monkeypatch):
        csv = "0, NVIDIA RTX 5090, 32768, 30720"
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            lambda *a, **kw: _fake_completed(stdout=csv),
        )

        gpus = _enumerate_nvidia()

        assert gpus is not None
        assert len(gpus) == 1
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA RTX 5090"
        # nvidia-smi reports MiB; helper divides by 1024 to reach GiB.
        assert gpus[0].vram_total_gb == pytest.approx(32.0)
        assert gpus[0].vram_free_gb == pytest.approx(30.0)

    def test_parses_multi_gpu(self, monkeypatch):
        csv = "0, NVIDIA RTX 5090, 32768, 30720\n1, NVIDIA RTX 4090, 24576, 20480"
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            lambda *a, **kw: _fake_completed(stdout=csv),
        )

        gpus = _enumerate_nvidia()

        assert gpus is not None
        assert [g.index for g in gpus] == [0, 1]
        assert gpus[1].name == "NVIDIA RTX 4090"
        assert gpus[1].vram_total_gb == pytest.approx(24.0)

    def test_nonzero_returncode_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            lambda *a, **kw: _fake_completed(stdout="", returncode=9),
        )
        assert _enumerate_nvidia() is None

    def test_binary_missing_returns_none(self, monkeypatch):
        def raise_fnf(*a, **kw):
            raise FileNotFoundError

        monkeypatch.setattr("device_utils.subprocess.run", raise_fnf)
        assert _enumerate_nvidia() is None

    def test_timeout_returns_none(self, monkeypatch):
        def raise_timeout(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        monkeypatch.setattr("device_utils.subprocess.run", raise_timeout)
        assert _enumerate_nvidia() is None

    def test_malformed_rows_are_skipped(self, monkeypatch):
        # First row is valid, second has fewer than 4 fields and must be ignored.
        csv = "0, NVIDIA RTX 5090, 32768, 30720\nbogus, row"
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            lambda *a, **kw: _fake_completed(stdout=csv),
        )

        gpus = _enumerate_nvidia()

        assert gpus is not None
        assert len(gpus) == 1


# ---------------------------------------------------------------------------
# _enumerate_amd
# ---------------------------------------------------------------------------


class TestEnumerateAmd:
    """amd-smi JSON → rocm-smi CSV → registry fallback chain."""

    def test_amdsmi_success(self, monkeypatch):
        payload = json.dumps(
            [
                {
                    "asic": {"market_name": "Radeon RX 7800 XT"},
                    "vram": {"size": {"value": 16384}},
                }
            ]
        )
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            _SubprocessRouter({"amd-smi": (0, payload)}),
        )

        gpus = _enumerate_amd()

        assert gpus is not None
        assert len(gpus) == 1
        assert gpus[0].name == "Radeon RX 7800 XT"
        assert gpus[0].vram_total_gb == pytest.approx(16.0)
        # amd-smi static output has no free-memory field; helper mirrors total.
        assert gpus[0].vram_free_gb == gpus[0].vram_total_gb

    def test_amdsmi_malformed_entry_skipped(self, monkeypatch):
        # Second entry's VRAM size is non-numeric; the inner try/except must
        # skip it (float("garbage") raises ValueError) rather than crashing
        # the whole enumeration. This is the hardening PR #211 added.
        payload = json.dumps(
            [
                {
                    "asic": {"market_name": "Radeon RX 7800 XT"},
                    "vram": {"size": {"value": 16384}},
                },
                {
                    "asic": {"market_name": "Radeon RX 9070"},
                    "vram": {"size": {"value": "garbage"}},
                },
            ]
        )
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            _SubprocessRouter({"amd-smi": (0, payload)}),
        )

        gpus = _enumerate_amd()

        assert gpus is not None
        assert len(gpus) == 1
        assert gpus[0].name == "Radeon RX 7800 XT"

    def test_amdsmi_invalid_json_falls_through_to_rocmsmi(self, monkeypatch):
        # amd-smi returns garbage → JSONDecodeError caught → rocm-smi runs.
        # rocm-smi reports bytes (16 GiB total, 1 GiB used).
        rocm_csv = "device,Total,Used\n0,17179869184,1073741824"
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            _SubprocessRouter(
                {
                    "amd-smi": (0, "not-json"),
                    "rocm-smi": (0, rocm_csv),
                }
            ),
        )

        gpus = _enumerate_amd()

        assert gpus is not None
        assert len(gpus) == 1
        assert gpus[0].index == 0
        assert gpus[0].vram_total_gb == pytest.approx(16.0)
        assert gpus[0].vram_free_gb == pytest.approx(15.0)

    def test_amdsmi_missing_falls_through_to_rocmsmi(self, monkeypatch):
        rocm_csv = "device,Total,Used\n0,17179869184,0"
        monkeypatch.setattr(
            "device_utils.subprocess.run",
            _SubprocessRouter({"rocm-smi": (0, rocm_csv)}),
        )

        gpus = _enumerate_amd()

        assert gpus is not None
        assert len(gpus) == 1
        assert gpus[0].vram_total_gb == pytest.approx(16.0)

    def test_no_tools_on_non_windows_returns_none(self, monkeypatch):
        # Force the Windows registry branch off so the assertion is deterministic
        # on any CI runner. On Windows with no AMD card the branch returns None
        # anyway, but patching here makes the outcome platform-independent.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("device_utils.subprocess.run", _SubprocessRouter({}))

        assert _enumerate_amd() is None


# ---------------------------------------------------------------------------
# enumerate_gpus
# ---------------------------------------------------------------------------


def _amd_must_not_run():
    raise AssertionError("_enumerate_amd should not be called")


class TestEnumerateGpus:
    """Dispatch chain: NVIDIA → AMD → torch.cuda → []."""

    def test_prefers_nvidia(self, monkeypatch):
        nvidia = [GPUInfo(index=0, name="NVIDIA RTX 5090", vram_total_gb=32.0, vram_free_gb=30.0)]
        monkeypatch.setattr("device_utils._enumerate_nvidia", lambda: nvidia)
        # AMD helper must not even be consulted when NVIDIA reports GPUs.
        monkeypatch.setattr("device_utils._enumerate_amd", _amd_must_not_run)

        assert enumerate_gpus() == nvidia

    def test_falls_back_to_amd_when_nvidia_unavailable(self, monkeypatch):
        amd = [GPUInfo(index=0, name="Radeon RX 7800 XT", vram_total_gb=16.0, vram_free_gb=16.0)]
        monkeypatch.setattr("device_utils._enumerate_nvidia", lambda: None)
        monkeypatch.setattr("device_utils._enumerate_amd", lambda: amd)

        assert enumerate_gpus() == amd

    def test_empty_nvidia_result_is_returned_as_is(self, monkeypatch):
        # nvidia-smi ran successfully but reported zero GPUs: we trust that
        # and must NOT fall through to AMD (None is the unavailable sentinel).
        monkeypatch.setattr("device_utils._enumerate_nvidia", lambda: [])
        monkeypatch.setattr("device_utils._enumerate_amd", _amd_must_not_run)

        assert enumerate_gpus() == []

    def test_torch_fallback_when_smi_tools_absent(self, monkeypatch):
        monkeypatch.setattr("device_utils._enumerate_nvidia", lambda: None)
        monkeypatch.setattr("device_utils._enumerate_amd", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

        fake_a = MagicMock(total_memory=24 * (1024**3))
        fake_a.name = "NVIDIA RTX 3090"
        fake_b = MagicMock(total_memory=10 * (1024**3))
        fake_b.name = "NVIDIA RTX 3080"
        props = {0: fake_a, 1: fake_b}
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props[i])

        gpus = enumerate_gpus()

        assert [g.name for g in gpus] == ["NVIDIA RTX 3090", "NVIDIA RTX 3080"]
        assert gpus[0].vram_total_gb == pytest.approx(24.0)
        assert gpus[1].vram_total_gb == pytest.approx(10.0)
        # torch.cuda exposes no free-memory field; helper mirrors total.
        assert gpus[0].vram_free_gb == gpus[0].vram_total_gb

    def test_returns_empty_when_nothing_available(self, monkeypatch):
        monkeypatch.setattr("device_utils._enumerate_nvidia", lambda: None)
        monkeypatch.setattr("device_utils._enumerate_amd", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert enumerate_gpus() == []
