from __future__ import annotations

import contextlib
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import color_utils as cu
from .core.model_transformer import GreenFormer


class CorridorKeyEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        precision: str = "auto",
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self.precision = self._resolve_precision(precision)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.model = self._load_model()

    def _resolve_precision(self, precision: str) -> str:
        precision = precision.lower()
        if precision == "auto":
            return "fp16" if self.device.type == "cuda" else "fp32"
        if precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be one of: auto, fp16, fp32")
        if precision == "fp16" and self.device.type != "cuda":
            return "fp32"
        return precision

    def _load_model(self) -> GreenFormer:
        print(f"Loading CorridorKey from {self.checkpoint_path}...")
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
        if self.precision == "fp16":
            model = model.half()
        if len(missing) > 0:
            print(f"[Warning] Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"[Warning] Unexpected keys: {unexpected}")

        return model

    @torch.no_grad()
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

        # 3. Normalize (ImageNet)
        # Model expects sRGB input normalized
        img_norm = (img_resized - self.mean) / self.std

        # 4. Prepare Tensor
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1)  # [H, W, 4]
        input_dtype = torch.float16 if self.precision == "fp16" else torch.float32
        inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).to(dtype=input_dtype).unsqueeze(0).to(self.device)

        # 5. Inference
        # Hook for Refiner Scaling
        handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            handle = self.model.refiner.register_forward_hook(scale_hook)

        amp_context = contextlib.nullcontext()
        if self.device.type == "cuda" and self.precision == "fp16":
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)

        with torch.inference_mode(), amp_context:
            out = self.model(inp_t)

        if handle:
            handle.remove()

        pred_alpha = out["alpha"]
        pred_fg = out["fg"]  # Output is sRGB (Sigmoid)

        # 6. Post-Process (Resize Back to Original Resolution)
        # We use Lanczos4 for high-quality resampling to minimize blur when going back to 4K/Original.
        res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
        res_alpha = cv2.resize(res_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg = cv2.resize(res_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        # --- ADVANCED COMPOSITING ---

        # A. Clean Matte (Auto-Despeckle)
        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = res_alpha

        # B. Despill FG
        # res_fg is sRGB.
        fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)

        # C. Premultiply (for EXR Output)
        # CONVERT TO LINEAR FIRST! EXRs must house linear color premultiplied by linear alpha.
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)

        # D. Pack RGBA
        # [H, W, 4] - All channels are now strictly Linear Float
        processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)

        # ----------------------------

        # 7. Composite (on Checkerboard) for checking
        # Generate Dark/Light Gray Checkerboard (in sRGB, convert to Linear)
        bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        bg_lin = cu.srgb_to_linear(bg_srgb)

        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            # If premultiplied model, we shouldn't multiply again (though our pipeline forces straight)
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

        comp_srgb = cu.linear_to_srgb(comp_lin)

        return {
            "alpha": res_alpha,  # Linear, Raw Prediction
            "fg": res_fg,  # sRGB, Raw Prediction (Straight)
            "comp": comp_srgb,  # sRGB, Composite
            "processed": processed_rgba,  # Linear/Premul, RGBA, Garbage Matted & Despilled
        }
