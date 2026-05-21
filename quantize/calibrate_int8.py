#!/usr/bin/env python3
"""
Stage 4: INT8 Post-Training Quantization via ONNX Runtime.

Reads green screen frames from calibration_frames/ to calibrate INT8 thresholds,
then writes a quantized model ready for Raspberry Pi / edge deployment.

Usage:
    python quantize/calibrate_int8.py \
        --fp32-model models/corridorkey_fp32_simplified.onnx \
        --int8-model models/corridorkey_int8.onnx \
        --frames-dir calibration_frames/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_GREEN_HSV_LOWER = np.array([35, 40, 40], dtype=np.uint8)
_GREEN_HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def generate_green_hint_mask(rgb: np.ndarray) -> np.ndarray:
    """Fast HSV chroma-key hint mask.  1 = foreground, 0 = green screen."""
    bgr = rgb[:, :, ::-1].astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    screen = cv2.inRange(hsv, _GREEN_HSV_LOWER, _GREEN_HSV_UPPER).astype(np.float32) / 255.0
    return 1.0 - screen  # invert


def preprocess(rgb: np.ndarray, img_size: int) -> np.ndarray:
    """Return [1, 4, H, W] float32 tensor (ImageNet-normalised RGB + hint mask)."""
    pil = Image.fromarray(rgb).resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0          # [H, W, 3]
    chw = arr.transpose(2, 0, 1)                            # [3, H, W]
    chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD

    mask = generate_green_hint_mask(np.array(pil))         # [H, W]
    mask = mask[np.newaxis]                                 # [1, H, W]

    rgba = np.concatenate([chw, mask], axis=0)             # [4, H, W]
    return rgba[np.newaxis].astype(np.float32)             # [1, 4, H, W]


class GreenScreenCalibrationReader(CalibrationDataReader):
    def __init__(self, frames_dir: str, img_size: int = 512, max_frames: int = 200) -> None:
        frames_path = Path(frames_dir)
        self.frames = sorted(frames_path.glob("*.png")) + sorted(frames_path.glob("*.jpg"))
        self.frames = self.frames[:max_frames]
        self.img_size = img_size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.frames):
            return None
        path = self.frames[self.idx]
        self.idx += 1
        rgb = np.array(Image.open(path).convert("RGB"))
        return {"rgba_input": preprocess(rgb, self.img_size)}

    def rewind(self) -> None:
        self.idx = 0


def calibrate(fp32_model: str, int8_model: str, frames_dir: str, img_size: int) -> None:
    frames_path = Path(frames_dir)
    all_frames = list(frames_path.glob("*.png")) + list(frames_path.glob("*.jpg"))

    if not all_frames:
        print(
            f"No frames found in {frames_dir}.\n"
            "Add 100-200 green screen PNG/JPG frames, then re-run.\n"
            "Frames are gitignored — see EDGE_DEPLOY.md for details."
        )
        return

    print(f"Found {len(all_frames)} calibration frames in {frames_dir}")

    output_path = Path(int8_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ONNX Runtime recommends shape inference + model pre-processing before
    # static quantization to improve INT8 accuracy.
    preprocessed_path = output_path.with_name(output_path.stem + "_preproc.onnx")
    print("Pre-processing model for quantization...")
    quant_pre_process(fp32_model, str(preprocessed_path), skip_symbolic_shape=True)

    reader = GreenScreenCalibrationReader(frames_dir, img_size=img_size)

    print("Running static INT8 quantization (this may take a few minutes)...")
    quantize_static(
        str(preprocessed_path),
        str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        reduce_range=False,
    )

    preprocessed_path.unlink(missing_ok=True)  # remove intermediate file

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved INT8 model: {int8_model}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 static calibration for edge deployment")
    parser.add_argument("--fp32-model", required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--int8-model", required=True, help="Output INT8 ONNX model path")
    parser.add_argument("--frames-dir", default="calibration_frames", help="Directory with calibration frames")
    parser.add_argument("--img-size", type=int, default=512, help="Model input resolution (must match export)")
    args = parser.parse_args()
    calibrate(args.fp32_model, args.int8_model, args.frames_dir, args.img_size)


if __name__ == "__main__":
    main()
