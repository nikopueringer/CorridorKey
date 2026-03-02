"""
Robust Video Matting (RVM) based alpha hint generator for CorridorKey.

Uses PeterL1n's RVM (MobileNetV3 backbone) to generate foreground alpha mattes
as hints. Fully automatic — no trimap, no interaction, no green-screen assumption.
Understands "this is a person" semantically.

Temporal consistency via recurrent ConvGRU states passed between frames.

Model: rvm_mobilenetv3 (3.7M params, 14.5MB, ~30ms/frame on MPS)
Default hint method on Apple Silicon.
"""
import os
import cv2
import numpy as np
import torch
from device_utils import resolve_device, clear_device_cache

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class RVMHintGenerator:
    def __init__(self, device=None, variant="mobilenetv3"):
        self.device = torch.device(resolve_device(device))
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting", variant, trust_repo=True
        )
        self.model = self.model.eval().to(self.device)
        self._rec = [None] * 4

    def _pick_downsample_ratio(self, h, w):
        """Choose downsample_ratio based on resolution.

        The downsampled resolution should land between 256-512px for best
        quality/speed tradeoff.
        """
        max_dim = max(h, w)
        if max_dim > 2000:
            return 0.125  # 4K
        elif max_dim > 1200:
            return 0.25   # FHD
        else:
            return 0.375  # HD or smaller

    def reset(self):
        """Reset recurrent states. Call between unrelated clips."""
        self._rec = [None] * 4

    def generate_hint(self, frame_bgr):
        """Generate an alpha hint mask for a single BGR frame.

        Maintains recurrent state between calls for temporal consistency.
        Call reset() between unrelated clips.

        Returns: numpy uint8 array [H, W] with values 0 (background) to 255 (foreground).
        """
        from torchvision.transforms.functional import to_tensor

        h, w = frame_bgr.shape[:2]
        ds_ratio = self._pick_downsample_ratio(h, w)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src = to_tensor(rgb).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        with torch.no_grad():
            fgr, pha, *self._rec = self.model(
                src, *self._rec, downsample_ratio=ds_ratio
            )

        alpha = pha[0, 0].cpu().numpy()
        mask = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        return mask

    def generate_hints_for_video(self, video_path, output_dir, max_frames=None,
                                 start_frame=0):
        """Generate alpha hints for all frames in a video file.

        Args:
            video_path: Path to input video.
            output_dir: Directory to write hint PNGs.
            max_frames: Optional limit on number of frames to process.
            start_frame: Frame index to start from (skip earlier frames).

        Returns:
            Tuple of (frame_count, fps).
        """
        os.makedirs(output_dir, exist_ok=True)
        self.reset()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if max_frames:
            total = min(total - start_frame, max_frames)

        count = 0
        while count < total:
            ret, frame = cap.read()
            if not ret:
                break
            mask = self.generate_hint(frame)
            cv2.imwrite(os.path.join(output_dir, f"{count:05d}.png"), mask)
            count += 1
            if (count % 10 == 0) or count == 1:
                print(f"  RVM hints: {count}/{total}")

        cap.release()
        return count, fps
