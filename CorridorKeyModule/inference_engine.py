from __future__ import annotations

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
        backbone_size: int | None = None,
        use_refiner: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.backbone_size = backbone_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self._checker_cache_srgb: torch.Tensor | None = None
        self._checker_cache_lin: torch.Tensor | None = None
        self._checker_cache_key: tuple[int, int] = (0, 0)

        self.model = self._load_model()

    def _load_model(self) -> GreenFormer:
        print(f"Loading CorridorKey from {self.checkpoint_path}...")
        # Initialize Model (Hiera Backbone)
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            backbone_size=self.backbone_size,
            use_refiner=self.use_refiner,
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

        # Cast weights to FP16 — autocast already handles FP16 activations,
        # this halves static VRAM footprint (~400MB savings)
        model = model.half()

        return model

    def _get_checkerboard(self, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (width, height)
        if self._checker_cache_srgb is None or self._checker_cache_key != key:
            bg_np = cu.create_checkerboard(width, height, checker_size=128, color1=0.15, color2=0.55)
            self._checker_cache_srgb = torch.from_numpy(bg_np).to(self.device)
            self._checker_cache_lin = cu.srgb_to_linear(self._checker_cache_srgb)
            self._checker_cache_key = key
        return self._checker_cache_srgb, self._checker_cache_lin

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
        inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)

        # 5. Inference
        # Hook for Refiner Scaling
        handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            handle = self.model.refiner.register_forward_hook(scale_hook)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = self.model(inp_t)

        if handle:
            handle.remove()

        pred_alpha = out["alpha"]  # [1, 1, model_size, model_size]
        pred_fg = out["fg"]  # [1, 3, model_size, model_size] sRGB (Sigmoid)

        # 6. Post-Process (Resize Back to Original Resolution)
        # Bicubic on GPU replaces Lanczos4 on CPU — avoids PCIe transfer bottleneck
        res_alpha = F.interpolate(pred_alpha.float(), size=(h, w), mode="bicubic", align_corners=False)
        res_fg = F.interpolate(pred_fg.float(), size=(h, w), mode="bicubic", align_corners=False)

        # Clamp after bicubic (can overshoot [0,1])
        res_alpha = res_alpha.clamp(0.0, 1.0)
        res_fg = res_fg.clamp(0.0, 1.0)

        # Permute to HWC for color_utils compatibility
        res_alpha = res_alpha[0].permute(1, 2, 0)  # [H, W, 1]
        res_fg = res_fg[0].permute(1, 2, 0)  # [H, W, 3]

        # --- ADVANCED COMPOSITING (all on GPU) ---

        # A. Clean Matte (Auto-Despeckle) — CPU-only (cv2.connectedComponents)
        if auto_despeckle:
            alpha_cpu = res_alpha.cpu().numpy()
            processed_alpha_np = cu.clean_matte(alpha_cpu, area_threshold=despeckle_size, dilation=25, blur_size=5)
            processed_alpha = torch.from_numpy(processed_alpha_np).to(self.device)
        else:
            processed_alpha = res_alpha

        # B. Despill FG (GPU tensor)
        fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)

        # C. Premultiply (for EXR Output)
        # CONVERT TO LINEAR FIRST! EXRs must house linear color premultiplied by linear alpha.
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)

        # D. Pack RGBA
        # [H, W, 4] - All channels are now strictly Linear Float
        processed_rgba = torch.cat([fg_premul_lin, processed_alpha], dim=-1)

        # ----------------------------

        # 7. Composite (on Checkerboard) — cached GPU tensor
        _bg_srgb, bg_lin = self._get_checkerboard(w, h)

        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

        comp_srgb = cu.linear_to_srgb(comp_lin)

        # 8. Transfer to CPU (single batch transfer at the end)
        return {
            "alpha": res_alpha.cpu().numpy(),  # Linear, Raw Prediction
            "fg": res_fg.cpu().numpy(),  # sRGB, Raw Prediction (Straight)
            "comp": comp_srgb.cpu().numpy(),  # sRGB, Composite
            "processed": processed_rgba.cpu().numpy(),  # Linear/Premul, RGBA, Garbage Matted & Despilled
        }
