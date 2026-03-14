import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

torch.set_float32_matmul_precision(["high", "highest"][0])


class ImagePreprocessor:
    def __init__(self, resolution: Tuple[int, int] = (1024, 1024)) -> None:
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def proc(self, image: Image.Image) -> torch.Tensor:
        image = self.transform_image(image)
        return image


usage_to_weights_file = {
    "General": "BiRefNet",
    "General-dynamic": "BiRefNet_dynamic",
    "General-HR": "BiRefNet_HR",
    "General-Lite": "BiRefNet_lite",
    "General-Lite-2K": "BiRefNet_lite-2K",
    "General-reso_512": "BiRefNet_512x512",
    "Matting": "BiRefNet-matting",
    "Matting-dynamic": "BiRefNet_dynamic-matting",
    "Matting-HR": "BiRefNet_HR-Matting",
    "Matting-Lite": "BiRefNet_lite-matting",
    "Portrait": "BiRefNet-portrait",
    "DIS": "BiRefNet-DIS5K",
    "HRSOD": "BiRefNet-HRSOD",
    "COD": "BiRefNet-COD",
    "DIS-TR_TEs": "BiRefNet-DIS5K-TR_TEs",
    "General-legacy": "BiRefNet-legacy",
}

half_precision = True

base_folder = os.path.join(os.path.dirname(__file__), "checkpoints")


class BiRefNetHandler:
    def __init__(self, device="cpu", usage="General"):
        self.device = device

        # Set resolution
        if usage in ["General-Lite-2K"]:
            self.resolution = (2560, 1440)
        elif usage in ["General-reso_512"]:
            self.resolution = (512, 512)
        elif usage in ["General-HR", "Matting-HR"]:
            self.resolution = (2048, 2048)
        else:
            if "-dynamic" in usage:
                self.resolution = None
            else:
                self.resolution = (1024, 1024)

        repo_name = usage_to_weights_file[usage]
        repo_id = f"ZhengPeng7/{repo_name}"
        model_local_dir = os.path.join(base_folder, repo_name)

        snapshot_download(
            repo_id=repo_id,
            local_dir=model_local_dir,
            local_dir_use_symlinks=False,  # Ensures actual files are downloaded, not just symlinks to the cache
        )

        self.birefnet = AutoModelForImageSegmentation.from_pretrained(model_local_dir, trust_remote_code=True)

        self.birefnet.to(device)
        self.birefnet.eval()
        if half_precision:
            self.birefnet.half()

    def cleanup(self):
        """Explicitly clear model and release GPU memory."""
        # Delete the model reference
        if hasattr(self, "birefnet"):
            del self.birefnet

        # Clear Python garbage
        import gc

        gc.collect()

        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def process(self, input_path, alpha_output_dir=None, dilate_radius=0, on_frame_complete=None):
        """
        Process a single video or directory of images.
        """
        input_path = Path(input_path)
        file_name = input_path.stem
        is_video = input_path.suffix.lower() in [".mp4", ".mkv", ".gif", ".mov", ".avi"]

        def get_frames():
            """Yields tuples of (image_numpy_array, output_file_name)"""
            if is_video:
                cap = cv2.VideoCapture(str(input_path))
                count = 0
                while True:
                    success, img = cap.read()
                    if not success:
                        break
                    yield img, f"{file_name}_alpha_{count:05d}.png"
                    count += 1
                cap.release()
            else:
                image_files = sorted(
                    [
                        f
                        for f in input_path.iterdir()
                        if f.is_file() and f.suffix.lower() in [".jpg", ".png", ".jpeg", ".exr"]
                    ]
                )
                if not image_files:
                    logging.warning(f"No images found in {input_path}")
                    return

                # Setup EXR support once if needed
                if "OPENCV_IO_ENABLE_OPENEXR" not in os.environ:
                    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

                for img_path in image_files:
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    # Keep original filename for image sequences
                    yield img, f"alphaSeq_{img_path.stem}.png"

        count = 0
        for image, out_name in get_frames():
            # Ensure correct conversion to RGB regardless of input format (EXR/PNG/JPG)
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # EXR images load as float32. PIL expects uint8. Normalize if necessary.
            if image_rgb.dtype != np.uint8:
                image_rgb = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            pil_image = Image.fromarray(image_rgb)

            # Preprocess
            if self.resolution is None:  # Account for dynamic models
                resolution_div_by_32 = [int(int(reso) // 32 * 32) for reso in pil_image.size]
                if resolution_div_by_32 != self.resolution:
                    self.resolution = resolution_div_by_32
            image_preprocessor = ImagePreprocessor(resolution=tuple(self.resolution))
            image_proc = image_preprocessor.proc(pil_image).unsqueeze(0).to(self.device)
            if half_precision:
                image_proc = image_proc.half()

            # Inference
            with torch.no_grad():
                preds = self.birefnet(image_proc)[-1].sigmoid().cpu()

            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred.float())

            # Post-Process
            target_size = (image.shape[1], image.shape[0])
            mask = pred_pil.resize(target_size)
            mask_np = np.array(mask)

            # Dilate
            if dilate_radius != 0:
                abs_radius = abs(dilate_radius)
                k_size = abs_radius * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                if dilate_radius > 0:
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)  # Expansion
                else:
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)  # Contraction

            # Strict Binary Threshold
            _, mask_np = cv2.threshold(mask_np, 10, 255, cv2.THRESH_BINARY)

            # Save
            if alpha_output_dir:
                save_path = os.path.join(alpha_output_dir, out_name)
                cv2.imwrite(save_path, mask_np)

            if on_frame_complete:
                on_frame_complete(count, 0)
