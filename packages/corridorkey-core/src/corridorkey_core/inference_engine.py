"""Runtime inference engine for the CorridorKey chroma keying model.

Wraps GreenFormer with checkpoint loading, optional torch.compile, and a
full per-frame processing pipeline including preprocessing, inference,
despilling, matte cleanup, and compositing.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional

from corridorkey_core.compositing import (
    clean_matte,
    composite_premul,
    composite_straight,
    create_checkerboard,
    despill,
    linear_to_srgb,
    premultiply,
    srgb_to_linear,
)
from corridorkey_core.model_transformer import GreenFormer

logger = logging.getLogger(__name__)


class CorridorKeyEngine:  # pragma: no cover
    """Inference engine for the CorridorKey chroma keying model.

    Loads a GreenFormer checkpoint, optionally compiles it with torch.compile,
    and exposes process_frame for per-frame alpha matte prediction.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        mixed_precision: bool = True,
        model_precision: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the engine and load the model from a checkpoint.

        Args:
            checkpoint_path: Path to the .pt or .pth checkpoint file.
            device: Torch device string, e.g. "cpu", "cuda", "cuda:0".
            img_size: Square resolution the model runs at internally.
            use_refiner: Whether to enable the CNN refiner module.
            mixed_precision: Whether to run inference in fp16 autocast.
            model_precision: Weight dtype for the model (float32 or float16).
        """
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = Path(checkpoint_path)
        self.use_refiner = use_refiner

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if mixed_precision or model_precision != torch.float32:
            # Faster matmul at the cost of slightly reduced float32 precision,
            # negligible compared to fp16 quantization error.
            torch.set_float32_matmul_precision("high")

        self.mixed_precision = mixed_precision
        if mixed_precision and model_precision == torch.float16:
            # Autocast to fp16 on top of fp16 weights adds overhead with no benefit.
            self.mixed_precision = False

        self.model_precision = model_precision

        model = self._load_model().to(model_precision)

        # torch.compile is only tested on Linux and Windows; skip on other platforms.
        if sys.platform == "linux" or sys.platform == "win32":
            try:
                # Point the inductor cache at a stable directory so compiled
                # kernels survive across runs — first run compiles once, all
                # subsequent runs load from cache and skip the ~60s wait.
                cache_dir = Path.home() / ".cache" / "corridorkey" / "torch_compile"
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
                self.model = torch.compile(model)
            except Exception as e:
                logger.warning("Model compilation failed (%s). Falling back to eager mode.", e)
                torch.cuda.empty_cache()
                self.model = model

    def _load_model(self) -> GreenFormer:
        """Load and return a GreenFormer model from the configured checkpoint path.

        Handles the _orig_mod. prefix left by torch.compile and resizes positional
        embeddings when the checkpoint resolution differs from img_size.
        """
        logger.info("Loading CorridorKey from %s", self.checkpoint_path)
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k", img_size=self.img_size, use_refiner=self.use_refiner
        )
        model = model.to(self.device)
        model.eval()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip the _orig_mod. prefix that torch.compile adds to state dict keys.
        # Also resize positional embeddings when the checkpoint was trained at a different resolution.
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]

            if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
                logger.debug("Resizing %s from %s to %s", k, v.shape, model_state[k].shape)
                # Treat the sequence dimension as a square spatial grid and bicubic-interpolate.
                seq_len_src = v.shape[1]
                seq_len_dst = model_state[k].shape[1]
                embed_dim = v.shape[2]

                grid_size_src = int(math.sqrt(seq_len_src))
                grid_size_dst = int(math.sqrt(seq_len_dst))

                pos_embed_spatial = v.permute(0, 2, 1).view(1, embed_dim, grid_size_src, grid_size_src)
                pos_embed_resized = functional.interpolate(
                    pos_embed_spatial, size=(grid_size_dst, grid_size_dst), mode="bicubic", align_corners=False
                )
                v = pos_embed_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            logger.warning("Missing keys in checkpoint: %s", missing)
        if len(unexpected) > 0:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

        return model

    @torch.inference_mode()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """Run the full keying pipeline on a single frame.

        Args:
            image: RGB float array [H, W, 3] in range 0.0-1.0 or uint8 0-255.
                Assumed sRGB unless input_is_linear is True.
            mask_linear: Grayscale float array [H, W] or [H, W, 1] in range 0.0-1.0.
                Always assumed to be linear.
            refiner_scale: Multiplier applied to the CNN refiner output deltas.
                Values above 1.0 strengthen refinement, below 1.0 weaken it.
            input_is_linear: If True, the image is treated as linear light and
                converted to sRGB before being passed to the model.
            fg_is_straight: If True, the foreground output is treated as straight
                (unpremultiplied) during compositing.
            despill_strength: Blend factor for the despill effect (0.0 to 1.0).
            auto_despeckle: If True, removes small disconnected foreground islands
                from the predicted alpha matte.
            despeckle_size: Minimum pixel area for a foreground island to be kept
                when auto_despeckle is enabled.

        Returns:
            A dict with four keys:
                "alpha": Raw predicted alpha [H, W, 1], linear float.
                "fg": Raw predicted foreground [H, W, 3], sRGB straight float.
                "comp": Preview composite over checkerboard [H, W, 3], sRGB float.
                "processed": Final RGBA [H, W, 4], linear premultiplied float.
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # Resize to the model's internal resolution.
        # When the input is linear, resize before converting to sRGB to preserve
        # energy in highlights during downsampling.
        if input_is_linear:
            image_resized_linear = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            image_resized_srgb = linear_to_srgb(image_resized_linear)
        else:
            image_resized_srgb = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # Normalize with ImageNet mean and std, which the Hiera encoder was pretrained with.
        image_normalized = (image_resized_srgb - self.mean) / self.std

        # Stack image and mask into a single [B, 4, H, W] tensor.
        model_input_np = np.concatenate([image_normalized, mask_resized], axis=-1)
        model_input = (
            torch.from_numpy(model_input_np.transpose((2, 0, 1))).unsqueeze(0).to(self.model_precision).to(self.device)
        )

        # Optionally scale the refiner's delta output via a forward hook.
        hook_handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:  # ty:ignore[unresolved-attribute]

            def scale_hook(module, input, output):
                return output * refiner_scale

            hook_handle = self.model.refiner.register_forward_hook(scale_hook)  # ty:ignore[unresolved-attribute]

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
            model_output = self.model(model_input)

        if hook_handle:
            hook_handle.remove()

        raw_alpha = model_output["alpha"]
        raw_fg = model_output["fg"]

        # Resize predictions back to the original frame resolution.
        # Lanczos4 minimises blur when upscaling back to 4K.
        alpha_pred = raw_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        fg_pred = raw_fg[0].permute(1, 2, 0).float().cpu().numpy()
        alpha_pred = cv2.resize(alpha_pred, (w, h), interpolation=cv2.INTER_LANCZOS4)
        fg_pred = cv2.resize(fg_pred, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if alpha_pred.ndim == 2:
            alpha_pred = alpha_pred[:, :, np.newaxis]

        # Remove small foreground islands (tracking markers, noise) from the matte.
        if auto_despeckle:
            alpha_despeckled = clean_matte(alpha_pred, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            alpha_despeckled = alpha_pred

        # Remove green spill from the foreground (still in sRGB at this point).
        fg_despilled = despill(fg_pred, green_limit_mode="average", strength=despill_strength)

        # Convert to linear and premultiply for EXR output.
        # EXR files must store linear, premultiplied color.
        fg_despilled_linear = srgb_to_linear(fg_despilled)
        fg_premultiplied = premultiply(fg_despilled_linear, alpha_despeckled)

        # Pack the final linear premultiplied RGBA.
        output_rgba = np.concatenate([fg_premultiplied, alpha_despeckled], axis=-1)

        # Build a checkerboard preview composite in linear light, then convert to sRGB for display.
        checkerboard_srgb = create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        checkerboard_linear = srgb_to_linear(checkerboard_srgb)

        if fg_is_straight:
            composite_linear = composite_straight(fg_despilled_linear, checkerboard_linear, alpha_despeckled)
        else:
            composite_linear = composite_premul(fg_despilled_linear, checkerboard_linear, alpha_despeckled)

        composite_srgb = linear_to_srgb(composite_linear)

        return {
            "alpha": alpha_pred,  # linear float, raw prediction
            "fg": fg_pred,  # sRGB float, straight (unpremultiplied)
            "comp": composite_srgb,  # sRGB float, preview composite over checkerboard
            "processed": output_rgba,  # linear float, premultiplied RGBA
        }  # ty:ignore[invalid-return-type]
