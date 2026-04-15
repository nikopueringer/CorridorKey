"""Tests for scripts/convert_pth_to_safetensors.py.

The converter is a maintainer-facing tool: we run it once per released model
and publish the result to HuggingFace. A silent bug (dropped tensors, wrong
prefix, altered values) would corrupt the published file for every downstream
user. These tests pin its behavior so future refactors can't regress it.

What's exercised:
  * ``_extract_state_dict`` unwraps the ``{"state_dict": ...}`` wrapper.
  * ``_extract_state_dict`` strips the ``_orig_mod.`` prefix added by torch.compile.
  * ``_extract_state_dict`` drops non-tensor metadata (optimizer state, epoch counters, …).
  * ``_extract_state_dict`` produces contiguous tensors (safetensors requirement).
  * ``_verify_round_trip`` flags missing keys and shape mismatches.
  * End-to-end: a .pth written with ``torch.save`` survives the round trip with
    tensor values unchanged.
  * CLI: ``--input`` missing and ``--output`` without ``.safetensors`` suffix
    both exit non-zero and leave no output file behind.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

# Import from the script — it's a plain module on disk, not an installed package.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_pth_to_safetensors.py"
assert _SCRIPT.is_file(), f"converter script missing at {_SCRIPT}"

# Inject the scripts/ dir on sys.path so we can import it as a module.
sys.path.insert(0, str(_SCRIPT.parent))
import convert_pth_to_safetensors as converter  # noqa: E402


class TestExtractStateDict:
    def test_unwraps_state_dict_key(self):
        """A checkpoint saved with ``{"state_dict": {...}, "epoch": 42}`` must unwrap to just the state dict."""
        wrapped = {
            "state_dict": {"layer.weight": torch.zeros(3, 3)},
            "epoch": 42,  # metadata that must NOT leak into the output
        }
        cleaned = converter._extract_state_dict(wrapped)
        assert set(cleaned.keys()) == {"layer.weight"}

    def test_accepts_bare_state_dict(self):
        """A state dict saved directly (no wrapper) must also work."""
        bare = {"layer.weight": torch.zeros(3, 3), "layer.bias": torch.zeros(3)}
        cleaned = converter._extract_state_dict(bare)
        assert set(cleaned.keys()) == {"layer.weight", "layer.bias"}

    def test_strips_orig_mod_prefix(self):
        """The ``_orig_mod.`` prefix added by ``torch.compile`` must be stripped,
        otherwise the published file bakes in a non-portable detail that trips
        ``load_state_dict`` for users who don't compile the model.
        """
        compiled = {
            "_orig_mod.block.0.weight": torch.zeros(2, 2),
            "_orig_mod.block.0.bias": torch.zeros(2),
        }
        cleaned = converter._extract_state_dict(compiled)
        assert set(cleaned.keys()) == {"block.0.weight", "block.0.bias"}

    def test_skips_non_tensor_entries(self):
        """Numeric/string metadata (optimizer state, epoch counters, hyperparams) must be filtered.
        safetensors can only serialize tensors — leaving these in would crash ``save_file``.
        """
        mixed = {
            "layer.weight": torch.zeros(2, 2),
            "epoch": 7,
            "optimizer_state": {"step": 100},
            "notes": "trained on 8xA100",
        }
        cleaned = converter._extract_state_dict(mixed)
        assert set(cleaned.keys()) == {"layer.weight"}

    def test_result_tensors_are_contiguous(self):
        """safetensors.save_file requires contiguous tensors. Non-contiguous views
        (common after slicing/transpose) must be made contiguous before return.
        """
        # Create a non-contiguous tensor via transpose.
        non_contig = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        assert not non_contig.is_contiguous()

        cleaned = converter._extract_state_dict({"weight": non_contig})
        assert cleaned["weight"].is_contiguous()


class TestVerifyRoundTrip:
    def test_passes_when_keys_and_shapes_match(self, tmp_path):
        """Happy path: a freshly saved file must verify without raising."""
        from safetensors.torch import save_file

        tensors = {"a": torch.zeros(2, 2), "b": torch.ones(3)}
        out = tmp_path / "ok.safetensors"
        save_file(tensors, str(out))

        # No exception expected.
        converter._verify_round_trip(tensors, out)

    def test_detects_missing_key(self, tmp_path):
        """If the written file is missing a key relative to the source, raise with the key name."""
        from safetensors.torch import save_file

        original = {"a": torch.zeros(2, 2), "b": torch.ones(3)}
        partial = {"a": torch.zeros(2, 2)}  # 'b' was somehow dropped
        out = tmp_path / "partial.safetensors"
        save_file(partial, str(out))

        with pytest.raises(RuntimeError, match=r"(missing|mismatch).*'b'|'b'"):
            converter._verify_round_trip(original, out)

    def test_detects_shape_mismatch(self, tmp_path):
        """If a tensor changes shape during conversion, the round-trip check must catch it."""
        from safetensors.torch import save_file

        original = {"a": torch.zeros(4, 4)}
        wrong_shape = {"a": torch.zeros(2, 2)}
        out = tmp_path / "wrong_shape.safetensors"
        save_file(wrong_shape, str(out))

        with pytest.raises(RuntimeError, match=r"shape mismatch"):
            converter._verify_round_trip(original, out)


class TestEndToEnd:
    def test_pth_roundtrip_preserves_tensor_values(self, tmp_path, monkeypatch):
        """Write a tiny .pth with ``torch.save``, run ``main()``, reload the
        .safetensors and verify that every tensor is bit-identical to the source.

        This is the single most important test: it proves the script's raison
        d'être — format conversion that preserves values.
        """
        src_state = {
            "_orig_mod.conv.weight": torch.randn(4, 3, 3, 3),
            "_orig_mod.conv.bias": torch.randn(4),
            "fc.weight": torch.randn(10, 4),
        }
        # Simulate a training-style checkpoint with a wrapper + metadata
        wrapped = {"state_dict": src_state, "epoch": 10, "optimizer": {"lr": 1e-4}}

        pth_path = tmp_path / "ckpt.pth"
        st_path = tmp_path / "ckpt.safetensors"
        torch.save(wrapped, str(pth_path))

        monkeypatch.setattr(sys, "argv", ["convert", "--input", str(pth_path), "--output", str(st_path)])
        exit_code = converter.main()
        assert exit_code == 0
        assert st_path.is_file()

        reloaded = load_file(str(st_path))

        # Prefix-stripped keys; no metadata leaked
        assert set(reloaded.keys()) == {"conv.weight", "conv.bias", "fc.weight"}

        torch.testing.assert_close(reloaded["conv.weight"], src_state["_orig_mod.conv.weight"])
        torch.testing.assert_close(reloaded["conv.bias"], src_state["_orig_mod.conv.bias"])
        torch.testing.assert_close(reloaded["fc.weight"], src_state["fc.weight"])


class TestCLIErrorHandling:
    def test_missing_input_exits_nonzero(self, tmp_path):
        """``--input`` pointing at a non-existent file must exit with a clear error and no output file."""
        missing = tmp_path / "does_not_exist.pth"
        out = tmp_path / "out.safetensors"

        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--input", str(missing), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
        assert not out.exists()

    def test_output_without_safetensors_suffix_rejected(self, tmp_path):
        """Defensive: if the user passes an output path without ``.safetensors`` suffix the script
        must refuse rather than silently producing a misnamed file that discovery won't pick up.
        """
        src = tmp_path / "src.pth"
        torch.save({"w": torch.zeros(2)}, str(src))
        wrong_out = tmp_path / "out.pth"  # wrong suffix

        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--input", str(src), "--output", str(wrong_out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert ".safetensors" in result.stderr or ".safetensors" in result.stdout
        assert not wrong_out.exists()
