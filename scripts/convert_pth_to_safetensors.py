"""Convert a CorridorKey ``.pth`` checkpoint into the ``.safetensors`` format.

Intended for maintainers publishing the official HuggingFace model, and for
contributors converting custom fine-tunes. Does not change tensor values — only
the serialization format.

Usage:
    uv run python scripts/convert_pth_to_safetensors.py \\
        --input  CorridorKeyModule/checkpoints/CorridorKey.pth \\
        --output CorridorKeyModule/checkpoints/CorridorKey.safetensors

Post-conversion, the script reloads the ``.safetensors`` and diffs the key set
and tensor shapes against the original; mismatches exit non-zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _extract_state_dict(raw: dict) -> dict[str, torch.Tensor]:
    """Mirror the engine's loader: unwrap the ``"state_dict"`` key if present
    and strip the ``_orig_mod.`` prefix left behind by ``torch.compile``.
    """
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    cleaned: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            # Skip metadata entries (optimizer state, epoch counters, …) — the
            # engine ignores them too. They don't belong in a published file.
            continue
        new_key = key.removeprefix("_orig_mod.")
        # safetensors requires contiguous tensors.
        cleaned[new_key] = tensor.detach().contiguous()
    return cleaned


def _verify_round_trip(original: dict[str, torch.Tensor], output_path: Path) -> None:
    """Reload the written file and assert key set + shapes match the source."""
    reloaded = load_file(str(output_path))

    missing = set(original) - set(reloaded)
    extra = set(reloaded) - set(original)
    if missing or extra:
        raise RuntimeError(f"Round-trip key mismatch.\n  missing={sorted(missing)}\n  extra={sorted(extra)}")

    for key, tensor in original.items():
        if reloaded[key].shape != tensor.shape:
            raise RuntimeError(
                f"Round-trip shape mismatch for {key}: {tuple(tensor.shape)} -> {tuple(reloaded[key].shape)}"
            )

    print(f"  round-trip OK: {len(reloaded)} tensors verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Source .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Destination .safetensors path")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1
    if args.output.suffix != ".safetensors":
        print(f"error: --output must end in .safetensors (got {args.output})", file=sys.stderr)
        return 1

    print(f"Loading {args.input} ...")
    raw = torch.load(args.input, map_location="cpu", weights_only=True)
    cleaned = _extract_state_dict(raw)
    print(f"  extracted {len(cleaned)} tensors")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.output} ...")
    save_file(cleaned, str(args.output))

    print("Verifying round-trip ...")
    _verify_round_trip(cleaned, args.output)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
