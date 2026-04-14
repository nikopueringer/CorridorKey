"""Feature: auto-model-download — Property-based tests for auto checkpoint download.

Properties tested:
  1: Missing checkpoint triggers download and returns valid path.
  2: Existing checkpoint skips download.
  3: Auto-download is Torch-only.
  4: Network errors produce actionable messages.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from CorridorKeyModule.backend import (
    HF_CHECKPOINT_FILENAME_SAFETENSORS,
    HF_REPO_ID,
    TORCH_EXT,
    _discover_checkpoint,
    _ensure_torch_checkpoint,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# File extensions that are NOT recognised as Torch checkpoints — used to
# populate "non-empty but no usable checkpoint" dirs. ``.safetensors`` is
# deliberately excluded because the Torch backend now treats it as a valid
# checkpoint (preferred over ``.pth``), so its presence would legitimately
# satisfy discovery and skip the auto-download path that Property 1 exercises.
_non_pth_extensions = st.sampled_from(
    [
        ".txt",
        ".json",
        ".bin",
        ".onnx",
        ".csv",
        ".log",
        ".yaml",
    ]
)

# Strategy: list of non-.pth filenames to place in the checkpoint dir
_junk_filenames = st.lists(
    st.tuples(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
            min_size=1,
            max_size=12,
        ),
        _non_pth_extensions,
    ).map(lambda t: f"{t[0]}{t[1]}"),
    min_size=0,
    max_size=5,
)


# ---------------------------------------------------------------------------
# Property 1: Missing checkpoint triggers download and returns valid path
# ---------------------------------------------------------------------------


class TestMissingCheckpointTriggersDownload:
    """Property 1: For any empty checkpoint directory (no .pth files),
    calling _discover_checkpoint(TORCH_EXT) invokes hf_hub_download with
    the correct repo ID and filename, copies the result to
    CHECKPOINT_DIR/CorridorKey.pth, and returns a Path that exists on disk.

    Feature: auto-model-download, Property 1: Missing checkpoint triggers download and returns valid path

    **Validates: Requirements 1.1, 1.2, 4.1, 4.2**
    """

    @settings(max_examples=100)
    @given(junk_files=_junk_filenames)
    def test_missing_pth_triggers_download_and_returns_valid_path(
        self,
        junk_files: list[str],
    ) -> None:
        """Feature: auto-model-download, Property 1: Missing checkpoint triggers download and returns valid path

        **Validates: Requirements 1.1, 1.2, 4.1, 4.2**
        """
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            ckpt_dir.mkdir()

            # Populate with non-.pth junk files (may be empty list)
            for fname in junk_files:
                (ckpt_dir / fname).touch()

            # Prepare a fake cached file that hf_hub_download would return
            cache_dir = Path(tmp) / "hf_cache"
            cache_dir.mkdir()
            cached_file = cache_dir / HF_CHECKPOINT_FILENAME_SAFETENSORS
            cached_file.write_bytes(b"fake-checkpoint-bytes")

            with (
                mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(ckpt_dir)),
                mock.patch(
                    "huggingface_hub.hf_hub_download",
                    return_value=str(cached_file),
                ) as mock_dl,
            ):
                result = _discover_checkpoint(TORCH_EXT)

                # Primary path lands the .safetensors in the checkpoint dir
                expected = ckpt_dir / HF_CHECKPOINT_FILENAME_SAFETENSORS
                assert result == expected, f"Expected {expected}, got {result}"

                # The file must actually exist on disk
                assert result.exists(), f"Returned path does not exist: {result}"

                # hf_hub_download must have been called with the safetensors filename
                mock_dl.assert_called_once_with(
                    repo_id=HF_REPO_ID,
                    filename=HF_CHECKPOINT_FILENAME_SAFETENSORS,
                )


# ---------------------------------------------------------------------------
# Strategies for Property 2
# ---------------------------------------------------------------------------

# Strategy: valid .pth filenames (alphanumeric + underscore/dash, non-empty)
_pth_basenames = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
).map(lambda s: f"{s}.pth")


# ---------------------------------------------------------------------------
# Property 2: Existing checkpoint skips download
# ---------------------------------------------------------------------------


class TestExistingCheckpointSkipsDownload:
    """Property 2: For any checkpoint directory that already contains a .pth
    file, calling _discover_checkpoint(TORCH_EXT) returns the existing file's
    path without invoking hf_hub_download.

    Feature: auto-model-download, Property 2: Existing checkpoint skips download

    **Validates: Requirements 1.3**
    """

    @settings(max_examples=100)
    @given(pth_name=_pth_basenames)
    def test_existing_pth_skips_download(self, pth_name: str) -> None:
        """Feature: auto-model-download, Property 2: Existing checkpoint skips download

        **Validates: Requirements 1.3**
        """
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            ckpt_dir.mkdir()

            # Place exactly one .pth file in the checkpoint directory
            existing_file = ckpt_dir / pth_name
            existing_file.write_bytes(b"fake-checkpoint-data")

            with (
                mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(ckpt_dir)),
                mock.patch(
                    "huggingface_hub.hf_hub_download",
                ) as mock_dl,
            ):
                result = _discover_checkpoint(TORCH_EXT)

                # hf_hub_download must NOT have been called
                mock_dl.assert_not_called()

                # The returned path must match the existing file
                assert result == existing_file, f"Expected {existing_file}, got {result}"


# ---------------------------------------------------------------------------
# Property 3: Auto-download is Torch-only
# ---------------------------------------------------------------------------


class TestAutoDownloadIsTorchOnly:
    """Property 3: For any extension that is not TORCH_EXT, calling
    _discover_checkpoint(ext) with zero matches raises FileNotFoundError
    without invoking hf_hub_download.

    Feature: auto-model-download, Property 3: Auto-download is Torch-only

    **Validates: Requirements 1.4, 4.3**
    """

    @settings(max_examples=100)
    @given(ext=_non_pth_extensions.map(lambda e: e if e.startswith(".") else f".{e}"))
    def test_non_pth_extension_raises_without_download(self, ext: str) -> None:
        """Feature: auto-model-download, Property 3: Auto-download is Torch-only

        **Validates: Requirements 1.4, 4.3**
        """
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            ckpt_dir.mkdir()

            with (
                mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(ckpt_dir)),
                mock.patch(
                    "huggingface_hub.hf_hub_download",
                ) as mock_dl,
            ):
                with pytest.raises(FileNotFoundError):
                    _discover_checkpoint(ext)

                # hf_hub_download must NOT have been called
                mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# Strategies for Property 4
# ---------------------------------------------------------------------------


def _make_hf_hub_http_error(message: str) -> Exception:
    """Create an HfHubHTTPError with a mock response object."""
    import requests
    from huggingface_hub.utils import HfHubHTTPError

    response = requests.Response()
    response.status_code = 503
    return HfHubHTTPError(message, response=response)


# Network-related exception factories: each takes a message and returns an exception
_network_exception_factories = [
    lambda msg: ConnectionError(msg),
    lambda msg: TimeoutError(msg),
    lambda msg: _make_hf_hub_http_error(msg),
]

_network_exception_strategy = st.tuples(
    st.sampled_from(_network_exception_factories),
    st.text(min_size=1, max_size=50),
).map(lambda t: t[0](t[1]))


# ---------------------------------------------------------------------------
# Property 4: Network errors produce actionable messages
# ---------------------------------------------------------------------------


class TestNetworkErrorsProduceActionableMessages:
    """Property 4: For any network-related exception raised by
    hf_hub_download, _ensure_torch_checkpoint() raises a RuntimeError
    whose message contains both the HuggingFace repository URL and a
    connectivity troubleshooting hint.

    Feature: auto-model-download, Property 4: Network errors produce actionable messages

    **Validates: Requirements 3.1**
    """

    @settings(max_examples=100)
    @given(exc=_network_exception_strategy)
    def test_network_errors_produce_actionable_messages(
        self,
        exc: Exception,
    ) -> None:
        """Feature: auto-model-download, Property 4: Network errors produce actionable messages

        **Validates: Requirements 3.1**
        """
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            ckpt_dir.mkdir()

            with (
                mock.patch("CorridorKeyModule.backend.CHECKPOINT_DIR", str(ckpt_dir)),
                mock.patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=exc,
                ),
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    _ensure_torch_checkpoint()

                error_msg = str(exc_info.value)

                # Must contain the HuggingFace repo URL
                expected_url = f"https://huggingface.co/{HF_REPO_ID}"
                assert expected_url in error_msg, (
                    f"Error message missing HF repo URL.\nExpected URL: {expected_url}\nGot message: {error_msg}"
                )

                # Must contain the connectivity hint
                expected_hint = "Check your network connection and try again"
                assert expected_hint in error_msg, (
                    f"Error message missing connectivity hint.\n"
                    f"Expected hint: {expected_hint}\n"
                    f"Got message: {error_msg}"
                )
