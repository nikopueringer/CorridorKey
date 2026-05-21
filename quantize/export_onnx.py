#!/usr/bin/env python3
"""
Stage 3: Export GreenFormer to ONNX (FP32).

Usage:
    python quantize/export_onnx.py \
        --checkpoint models/CorridorKey.safetensors \
        --output models/corridorkey_fp32.onnx \
        --img-size 512
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from CorridorKeyModule.core.model_transformer import GreenFormer


class _OnnxWrapper(nn.Module):
    """Returns (alpha, fg) tuple instead of dict — required for ONNX export."""

    def __init__(self, model: GreenFormer) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgba_input: torch.Tensor):
        out = self.model(rgba_input)
        return out["alpha"], out["fg"]


def _load_state_dict(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            return load_file(checkpoint_path)
        except ImportError:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return state.get("state_dict", state)


def _strip_compiled_prefix(state_dict: dict) -> dict:
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}


def _fix_pos_embeds(state_dict: dict, model: nn.Module) -> dict:
    """Bicubic-interpolate positional embeddings when checkpoint and model resolutions differ.

    The CorridorKey checkpoint is trained at 2048×2048 (pos_embed has 262144 tokens).
    When exporting at 512×512 the model expects 16384 tokens — we resize here, exactly
    as CorridorKeyEngine._load_model() does.
    """
    model_state = model.state_dict()
    fixed = {}
    for k, v in state_dict.items():
        if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
            n_src = v.shape[1]
            n_dst = model_state[k].shape[1]
            c     = v.shape[2]
            g_src = int(math.sqrt(n_src))
            g_dst = int(math.sqrt(n_dst))
            v_img     = v.permute(0, 2, 1).view(1, c, g_src, g_src)
            v_resized = F.interpolate(v_img, size=(g_dst, g_dst), mode="bicubic", align_corners=False)
            v = v_resized.flatten(2).transpose(1, 2)
            print(f"  resized pos_embed {k}: {n_src} -> {n_dst} tokens")
        fixed[k] = v
    return fixed


def _disable_flash_attention(model: nn.Module) -> None:
    """Force standard SDPA path so ONNX tracing doesn't hit fused kernel ops."""
    for module in model.modules():
        for attr in ("fused_attn", "use_fused_attn"):
            if hasattr(module, attr):
                setattr(module, attr, False)


def export(checkpoint: str, output: str, img_size: int, opset: int) -> None:
    print(f"Loading checkpoint: {checkpoint}  (img_size={img_size})")

    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=img_size,
        use_refiner=True,
    )

    state_dict = _strip_compiled_prefix(_load_state_dict(checkpoint))
    state_dict = _fix_pos_embeds(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model.eval()
    _disable_flash_attention(model)

    # Force math SDPA backend (CPU-safe, ONNX-traceable)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    wrapper = _OnnxWrapper(model)
    dummy = torch.zeros(1, 4, img_size, img_size, dtype=torch.float32)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX opset {opset} -> {output}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["rgba_input"],
            output_names=["alpha", "fg"],
            dynamic_axes={
                "rgba_input": {0: "batch"},
                "alpha": {0: "batch"},
                "fg": {0: "batch"},
            },
            do_constant_folding=True,
        )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved FP32 ONNX model: {output}  ({size_mb:.1f} MB)")

    # Optional graph simplification
    try:
        import onnx
        import onnxsim

        print("Simplifying ONNX graph...")
        model_onnx = onnx.load(str(output_path))
        simplified, ok = onnxsim.simplify(model_onnx)
        if ok:
            simplified_path = output_path.with_name(output_path.stem + "_simplified.onnx")
            onnx.save(simplified, str(simplified_path))
            size_mb = simplified_path.stat().st_size / 1024 / 1024
            print(f"Saved simplified ONNX: {simplified_path}  ({size_mb:.1f} MB)")
        else:
            print("Simplification returned no changes.")
    except ImportError:
        print("Skipping simplification (install onnx-simplifier: pip install onnxsim onnx)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GreenFormer to ONNX (FP32)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth or .safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument("--img-size", type=int, default=512, help="Input resolution (default: 512 for edge)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.img_size, args.opset)


if __name__ == "__main__":
    main()
