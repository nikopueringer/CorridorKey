#!/usr/bin/env python3
"""
Stage 5: Raspberry Pi inference test.

Runs a single image through the INT8 ONNX model and saves the keyed result.

Usage:
    python camera/infer_pi.py \
        --model models/corridorkey_int8.onnx \
        --image test.jpg \
        --output result.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_GREEN_HSV_LOWER = np.array([35, 40, 40], dtype=np.uint8)
_GREEN_HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def load_model(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


def generate_green_hint_mask(rgb: np.ndarray) -> np.ndarray:
    """HSV chroma-key hint mask.  1 = foreground, 0 = green screen."""
    bgr = rgb[:, :, ::-1].astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    screen = cv2.inRange(hsv, _GREEN_HSV_LOWER, _GREEN_HSV_UPPER).astype(np.float32) / 255.0
    return 1.0 - screen


def preprocess(image_pil: PILImage, target_size: int) -> np.ndarray:
    """Return [1, 4, H, W] float32: ImageNet-normalised RGB + green hint mask."""
    resized = image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    rgb = np.array(resized, dtype=np.float32)

    chw = (rgb / 255.0).transpose(2, 0, 1)  # [3, H, W]
    chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD

    mask = generate_green_hint_mask(rgb.astype(np.uint8))  # [H, W]
    mask = mask[np.newaxis]  # [1, H, W]

    rgba = np.concatenate([chw, mask], axis=0)  # [4, H, W]
    return rgba[np.newaxis].astype(np.float32)  # [1, 4, H, W]


def infer(session: ort.InferenceSession, image_pil: PILImage, target_size: int = 512):
    """
    Run inference. Returns (alpha, fg):
        alpha: [H, W]  float32, 0=transparent 1=opaque
        fg:    [H, W, 3] float32, foreground RGB in [0, 1]
    """
    input_tensor = preprocess(image_pil, target_size)
    alpha_raw, fg_raw = session.run(["alpha", "fg"], {"rgba_input": input_tensor})
    # alpha_raw: [1, 1, H, W]  fg_raw: [1, 3, H, W]
    alpha = alpha_raw[0, 0]  # [H, W]
    fg = fg_raw[0].transpose(1, 2, 0)  # [H, W, 3]
    return alpha, fg


def composite_over_black(alpha: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Premultiply foreground over a black background → [H, W, 3] uint8."""
    composited = fg * alpha[:, :, np.newaxis]
    return (composited * 255).clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="CorridorKey Raspberry Pi inference test")
    parser.add_argument("--model", default="models/corridorkey_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="output.png", help="Path to save keyed output")
    parser.add_argument("--size", type=int, default=512, help="Model input resolution")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    session = load_model(args.model)

    image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = image.size

    print("Running inference...")
    alpha, fg = infer(session, image, target_size=args.size)

    # Resize outputs back to original resolution
    alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    fg_pil = Image.fromarray((fg * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    alpha_up = np.array(alpha_pil) / 255.0
    fg_up = np.array(fg_pil) / 255.0

    result = composite_over_black(alpha_up, fg_up)
    Image.fromarray(result).save(args.output)
    print(f"Saved keyed output: {args.output}")


if __name__ == "__main__":
    main()
