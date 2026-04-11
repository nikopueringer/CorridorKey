#!/usr/bin/env python3
"""Convert .pth / .pt checkpoints to the safetensors format.

Usage
-----
# Convert all checkpoints in the default directory:
python convert_checkpoint.py

# Convert specific files or directories:
python convert_checkpoint.py path/to/model.pth path/to/lora_dir/

# Convert and remove the originals after success:
python convert_checkpoint.py --delete-original
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CHECKPOINT_DIR = Path("CorridorKeyModule/checkpoints")


def _convert(src: Path) -> Path:
    """Load *src* (.pth / .pt) and save as a .safetensors file next to it.

    Handles both plain state-dicts and checkpoints that wrap the state-dict
    under a ``"state_dict"`` key (e.g. checkpoints saved by PyTorch Lightning
    or the CorridorKey training harness).

    Returns the path of the newly created .safetensors file.
    """
    import torch
    from safetensors.torch import save_file

    print(f"  Loading  {src} …")
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    non_tensor = [k for k, v in state_dict.items() if not hasattr(v, "contiguous")]
    if non_tensor:
        print(f"  Warning: skipping non-tensor keys: {non_tensor}")
        state_dict = {k: v for k, v in state_dict.items() if hasattr(v, "contiguous")}

    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    dst = src.with_suffix(".safetensors")
    print(f"  Saving   {dst} ({len(state_dict)} tensors) …")
    save_file(state_dict, dst)
    return dst


def convert_file(src: Path, *, delete_original: bool = False) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Not found: {src}")
    if src.suffix not in (".pth", ".pt"):
        raise ValueError(f"Expected .pth or .pt, got: {src}")

    dst = _convert(src)

    if delete_original:
        src.unlink()
        print(f"  Deleted  {src}")

    print(f"  Done: {dst}\n")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .pth/.pt checkpoints to safetensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to convert. Defaults to CorridorKeyModule/checkpoints/.",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Remove the original .pth/.pt file after a successful conversion.",
    )
    args = parser.parse_args()

    targets: list[Path] = []
    search_roots = [Path(p) for p in args.paths] if args.paths else [DEFAULT_CHECKPOINT_DIR]

    for root in search_roots:
        if root.is_file():
            targets.append(root)
        elif root.is_dir():
            found = list(root.glob("*.pth")) + list(root.glob("*.pt"))
            if not found:
                print(f"No .pth/.pt files found in {root}")
            targets.extend(found)
        else:
            print(f"Warning: path not found — {root}", file=sys.stderr)

    if not targets:
        print("Nothing to convert.")
        sys.exit(0)

    errors: list[tuple[Path, Exception]] = []
    for src in targets:
        print(f"Converting {src.name}:")
        try:
            convert_file(src, delete_original=args.delete_original)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}\n", file=sys.stderr)
            errors.append((src, exc))

    if errors:
        print(f"\n{len(errors)} file(s) failed to convert.", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {len(targets)} file(s).")


if __name__ == "__main__":
    main()
