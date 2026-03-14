from __future__ import annotations

import logging
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import color_utils as cu
from .core.model_transformer import GreenFormer

logger = logging.getLogger(__name__)


class CorridorKeyEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        mixed_precision: bool = True,
        model_precision: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner

        # ImageNet normalization constants — the Hiera backbone was pretrained
        # with these values, so inference must use the same transform
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if mixed_precision or model_precision != torch.float32:
            # Use TF32 tensor cores for matrix multiplications on Ampere+ GPUs.
            # ~2x faster than IEEE FP32 with negligible precision loss
            # (mantissa truncated from 23 to 10 bits — well within fp16 noise).
            torch.set_float32_matmul_precision("high")

        self.mixed_precision = mixed_precision
        if mixed_precision and model_precision == torch.float16:
            # autocast fp16→fp16 adds overhead for zero benefit — the model
            # is already in half precision, so skip the autocast wrapper
            self.mixed_precision = False

        self.model_precision = model_precision

        model = self._load_model().to(model_precision)

        # We only tested compilation on windows and linux. For other platforms compilation is disabled as a precaution.
        if sys.platform == "linux" or sys.platform == "win32":
            # Try compiling the model. Fallback to eager mode if it fails.
            try:
                self.model = torch.compile(model)
                # Trigger compilation with a dummy input
                dummy_input = torch.zeros(1, 4, img_size, img_size, dtype=model_precision, device=self.device)
                with torch.inference_mode():
                    self.model(dummy_input)
            except Exception as e:
                logger.info(f"Model compilation failed with error: {e}")
                logger.warning("Model compilation failed. Falling back to eager mode.")
                torch.cuda.empty_cache()
                self.model = model

    def _load_model(self) -> GreenFormer:
        logger.info("Loading CorridorKey from %s", self.checkpoint_path)
        # Initialize Model (Hiera Backbone)
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k", img_size=self.img_size, use_refiner=self.use_refiner
        )
        model = model.to(self.device)
        model.eval()

        # Load Weights
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Fix Compiled Model Prefix & Handle PosEmbed Mismatch
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]

            # Check for PosEmbed Mismatch
            if "pos_embed" in k and k in model_state:
                if v.shape != model_state[k].shape:
                    print(f"Resizing {k} from {v.shape} to {model_state[k].shape}")
                    # v: [1, N_src, C]
                    # target: [1, N_dst, C]
                    # We assume square grid
                    N_src = v.shape[1]
                    N_dst = model_state[k].shape[1]
                    C = v.shape[2]

                    grid_src = int(math.sqrt(N_src))
                    grid_dst = int(math.sqrt(N_dst))

                    # Reshape to [1, C, H, W]
                    v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)

                    # Interpolate
                    v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)

                    # Reshape back
                    v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"[Warning] Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"[Warning] Unexpected keys: {unexpected}")

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
        """
        Process a single frame.
        Args:
            image: Numpy array [H, W, 3] (0.0-1.0 or 0-255).
                   - If input_is_linear=False (Default): Assumed sRGB.
                   - If input_is_linear=True: Assumed Linear.
            mask_linear: Numpy array [H, W] or [H, W, 1] (0.0-1.0). Assumed Linear.
            refiner_scale: Multiplier for Refiner Deltas (default 1.0).
            input_is_linear: bool. If True, resizes in Linear then transforms to sRGB.
                             If False, resizes in sRGB (standard).
            fg_is_straight: bool. If True, assumes FG output is Straight (unpremultiplied).
                            If False, assumes FG output is Premultiplied.
            despill_strength: float. 0.0 to 1.0 multiplier for the despill effect.
            auto_despeckle: bool. If True, cleans up small disconnected components from the predicted alpha matte.
            despeckle_size: int. Minimum number of consecutive pixels required to keep an island.
        Returns:
             dict: {'alpha': np, 'fg': np (sRGB), 'comp': np (sRGB on Gray)}
        """
        # 1. Inputs Check & Normalization
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        # Ensure Mask Shape
        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # 2. Resize to Model Size
        # If input is linear, we resize in linear to preserve energy/highlights,
        # THEN convert to sRGB for the model.
        if input_is_linear:
            # Resize in Linear
            img_resized_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # Convert to sRGB for Model
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            # Standard sRGB Resize
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # 3. ImageNet normalization — required because the Hiera backbone
        # was pretrained with this transform; skipping it shifts all activations
        img_normalized = (img_resized - self.imagenet_mean) / self.imagenet_std

        # 4. Prepare input tensor: concatenate normalized RGB + mask as 4-channel input
        # then convert from HWC to BCHW for PyTorch conv layers
        model_input_np = np.concatenate([img_normalized, mask_resized], axis=-1)  # [H, W, 4]
        model_input = (
            torch.from_numpy(model_input_np.transpose((2, 0, 1))).unsqueeze(0).to(self.model_precision).to(self.device)
        )

        # 5. Inference
        # Dynamic refiner scaling: temporarily hook the refiner's output to
        # multiply by refiner_scale. This lets artists dial refinement up/down
        # without retraining. The hook is removed immediately after inference.
        refiner_hook_handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            refiner_hook_handle = self.model.refiner.register_forward_hook(scale_hook)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
            model_output = self.model(model_input)

        if refiner_hook_handle:
            refiner_hook_handle.remove()

        # Free cached CUDA allocations between stages to reduce peak VRAM.
        # Without this, the allocator holds onto blocks from inference that
        # aren't needed during post-processing, inflating peak VRAM by ~500MB.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        predicted_alpha = model_output["alpha"]
        predicted_fg = model_output["fg"]  # sRGB space (sigmoid-activated)

        # 6. Resize predictions back to original resolution.
        # Lanczos4 preserves sharp edges better than bilinear — important for
        # alpha mattes where soft edges would create fringing artifacts.
        alpha_fullres = predicted_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        fg_fullres = predicted_fg[0].permute(1, 2, 0).float().cpu().numpy()
        alpha_fullres = cv2.resize(alpha_fullres, (w, h), interpolation=cv2.INTER_LANCZOS4)
        fg_fullres = cv2.resize(fg_fullres, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if alpha_fullres.ndim == 2:
            alpha_fullres = alpha_fullres[:, :, np.newaxis]

        # --- Post-processing pipeline ---

        # A. Auto-despeckle: remove small disconnected alpha islands
        # (tracking markers, noise) while preserving the main subject
        if auto_despeckle:
            cleaned_alpha = cu.clean_matte(alpha_fullres, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            cleaned_alpha = alpha_fullres

        # B. Green spill removal on the foreground (still in sRGB)
        fg_despilled = cu.despill(fg_fullres, green_limit_mode="average", strength=despill_strength)

        # C. Premultiply for EXR output.
        # EXR convention: linear color premultiplied by linear alpha.
        # We must convert sRGB→linear BEFORE premultiplying to avoid
        # gamma-curved color values bleeding into the alpha multiply.
        fg_despilled_linear = cu.srgb_to_linear(fg_despilled)
        fg_premultiplied_linear = cu.premultiply(fg_despilled_linear, cleaned_alpha)

        # D. Pack RGBA — all channels strictly linear float for compositing software
        processed_rgba = np.concatenate([fg_premultiplied_linear, cleaned_alpha], axis=-1)

        # 7. Preview composite: FG over checkerboard for visual QC.
        # All compositing done in linear space, then converted to sRGB for display.
        checkerboard_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        checkerboard_linear = cu.srgb_to_linear(checkerboard_srgb)

        if fg_is_straight:
            composite_linear = cu.composite_straight(fg_despilled_linear, checkerboard_linear, cleaned_alpha)
        else:
            composite_linear = cu.composite_premul(fg_despilled_linear, checkerboard_linear, cleaned_alpha)

        composite_srgb = cu.linear_to_srgb(composite_linear)

        return {  # type: ignore[return-value]  # cu.* returns ndarray|Tensor but inputs are always ndarray here
            "alpha": alpha_fullres,  # Linear, raw prediction (before despeckle)
            "fg": fg_fullres,  # sRGB, raw prediction (straight/unpremultiplied)
            "comp": composite_srgb,  # sRGB, preview composite over checkerboard
            "processed": processed_rgba,  # Linear premultiplied RGBA (despeckled + despilled)
        }
