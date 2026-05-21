#!/usr/bin/env python3
"""
Stage 6: Full camera capture pipeline with real-time green screen removal.

Usage:
    python camera/camera_capture.py \
        --model models/corridorkey_int8.onnx \
        --camera 0 \
        --output-dir output/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Allow running from both the project root and the camera/ directory
sys.path.insert(0, str(Path(__file__).parent))

from infer_pi import composite_over_black, infer, load_model

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def capture_and_infer(
    session: ort.InferenceSession,
    cap: cv2.VideoCapture,
    model_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one frame, run inference. Returns (original_bgr, alpha, fg)."""
    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    alpha, fg = infer(session, pil_image, target_size=model_size)
    return frame_bgr, alpha, fg


def main() -> None:
    parser = argparse.ArgumentParser(description="CorridorKey real-time green screen removal")
    parser.add_argument("--model", default="models/corridorkey_int8.onnx", help="Path to INT8 ONNX model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--output-dir", default="output", help="Directory to save processed frames")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS")
    parser.add_argument("--size", type=int, default=512, help="Model input resolution")
    parser.add_argument("--save-frames", action="store_true", help="Save each processed frame to --output-dir")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    if args.save_frames:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    session = load_model(args.model)

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print("Running — press 'q' to quit, 's' to toggle frame saving")
    frame_count = 0
    start_time = time.time()
    save_frames = args.save_frames

    try:
        while True:
            try:
                frame_bgr, alpha, fg = capture_and_infer(session, cap, args.size)
            except RuntimeError as exc:
                print(exc)
                break

            orig_h, orig_w = frame_bgr.shape[:2]

            # Resize model outputs back to camera resolution
            alpha_up = (
                np.array(
                    Image.fromarray((alpha * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                )
                / 255.0
            )
            fg_up = (
                np.array(
                    Image.fromarray((fg * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                )
                / 255.0
            )

            result_rgb = composite_over_black(alpha_up, fg_up)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            if save_frames:
                out_path = Path(args.output_dir) / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(out_path), result_bgr)

            cv2.imshow("CorridorKey Edge", result_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_frames = not save_frames
                print(f"Frame saving {'enabled' if save_frames else 'disabled'}")

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"FPS: {frame_count / elapsed:.1f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        if frame_count:
            print(f"Processed {frame_count} frames in {elapsed:.1f}s  ({frame_count / elapsed:.1f} FPS avg)")


if __name__ == "__main__":
    main()
