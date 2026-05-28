#!/usr/bin/env python3
"""
Creates a minimal dummy ONNX model with GreenFormer's exact input/output spec.
Used for testing the camera pipeline without the real 300 MB checkpoint.

Input:  rgba_input  [B, 4, H, W]  — normalised RGB + green hint mask
Output: alpha       [B, 1, H, W]  — transparency matte  (0=screen, 1=fg)
        fg          [B, 3, H, W]  — foreground colour

The dummy model applies a few depthwise convolutions so the output isn't a
trivial pass-through, giving the pipeline something realistic to exercise.

Usage:
    python quantize/create_dummy_model.py --output models/dummy_512.onnx --size 512
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _conv_weights(out_ch: int, in_ch: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out_ch, in_ch, k, k)).astype(np.float32)
    # Kaiming-ish normalisation so activations stay reasonable
    w *= np.sqrt(2.0 / (in_ch * k * k))
    return w


def build_dummy_model(img_size: int) -> onnx.ModelProto:
    """
    Architecture (pure ONNX, no PyTorch needed):

        rgba_input [B,4,H,W]
            │
            ├─ Slice channels 0:3 ──► Conv(3→8) ──► ReLU ──► Conv(8→3) ──► Sigmoid ──► fg
            │
            └─ Slice channel  3:4 ──► Conv(1→8) ──► ReLU ──► Conv(8→1) ──► Sigmoid ──► alpha
    """
    pad = 1

    # ── weights (stored as ONNX initialisers) ──────────────────────────────
    w_rgb1 = _conv_weights(8, 3, 3, seed=0)
    b_rgb1 = np.zeros(8, dtype=np.float32)
    w_rgb2 = _conv_weights(3, 8, 3, seed=1)
    b_rgb2 = np.zeros(3, dtype=np.float32)

    w_msk1 = _conv_weights(8, 1, 3, seed=2)
    b_msk1 = np.zeros(8, dtype=np.float32)
    w_msk2 = _conv_weights(1, 8, 3, seed=3)
    b_msk2 = np.zeros(1, dtype=np.float32)

    inits = [
        numpy_helper.from_array(w_rgb1, name="w_rgb1"),
        numpy_helper.from_array(b_rgb1, name="b_rgb1"),
        numpy_helper.from_array(w_rgb2, name="w_rgb2"),
        numpy_helper.from_array(b_rgb2, name="b_rgb2"),
        numpy_helper.from_array(w_msk1, name="w_msk1"),
        numpy_helper.from_array(b_msk1, name="b_msk1"),
        numpy_helper.from_array(w_msk2, name="w_msk2"),
        numpy_helper.from_array(b_msk2, name="b_msk2"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="s_rgb_start"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), name="s_rgb_end"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), name="s_msk_start"),
        numpy_helper.from_array(np.array([4], dtype=np.int64), name="s_msk_end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="s_axis"),
    ]

    # ── nodes ──────────────────────────────────────────────────────────────
    nodes = [
        # Split channels
        helper.make_node("Slice", ["rgba_input", "s_rgb_start", "s_rgb_end", "s_axis"], ["rgb_slice"]),
        helper.make_node("Slice", ["rgba_input", "s_msk_start", "s_msk_end", "s_axis"], ["msk_slice"]),
        # RGB branch → fg
        helper.make_node("Conv", ["rgb_slice", "w_rgb1", "b_rgb1"], ["rgb_h1"], pads=[pad] * 4),
        helper.make_node("Relu", ["rgb_h1"], ["rgb_r1"]),
        helper.make_node("Conv", ["rgb_r1", "w_rgb2", "b_rgb2"], ["rgb_h2"], pads=[pad] * 4),
        helper.make_node("Sigmoid", ["rgb_h2"], ["fg"]),
        # Mask branch → alpha
        helper.make_node("Conv", ["msk_slice", "w_msk1", "b_msk1"], ["msk_h1"], pads=[pad] * 4),
        helper.make_node("Relu", ["msk_h1"], ["msk_r1"]),
        helper.make_node("Conv", ["msk_r1", "w_msk2", "b_msk2"], ["msk_h2"], pads=[pad] * 4),
        helper.make_node("Sigmoid", ["msk_h2"], ["alpha"]),
    ]

    # ── graph ──────────────────────────────────────────────────────────────
    graph = helper.make_graph(
        nodes,
        "GreenFormerDummy",
        inputs=[
            helper.make_tensor_value_info("rgba_input", TensorProto.FLOAT, [1, 4, img_size, img_size]),
        ],
        outputs=[
            helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [1, 1, img_size, img_size]),
            helper.make_tensor_value_info("fg", TensorProto.FLOAT, [1, 3, img_size, img_size]),
        ],
        initializer=inits,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dummy GreenFormer ONNX model for pipeline testing")
    parser.add_argument("--output", default="models/dummy_512.onnx", help="Output path")
    parser.add_argument("--size", type=int, default=512, help="Input resolution")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building dummy model (size={args.size})...")
    model = build_dummy_model(args.size)
    onnx.save(model, str(out))

    size_kb = out.stat().st_size / 1024
    print(f"Saved: {out}  ({size_kb:.1f} KB)")
    print("\nTest it with:")
    print(f"  python camera/infer_pi.py --model {out} --image <your_image.jpg> --output result.png")


if __name__ == "__main__":
    main()
