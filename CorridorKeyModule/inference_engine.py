from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import color_utils as cu
from .core.model_transformer import GreenFormer


def _make_blend_weight(tile_size: int, overlap: int) -> np.ndarray:
    """Create a raised-cosine blending weight for tiled inference.

    Returns an array of shape [tile_size, tile_size] with values in [0, 1].
    The edges within the overlap zone taper to 0 using a cosine ramp,
    giving seamless blending when tiles overlap.
    """
    weight = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap <= 0:
        return weight

    # Build 1-D cosine ramp: 0 → 1 over `overlap` pixels
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(ramp * np.pi)  # raised-cosine

    # Apply ramp to all four edges
    for i in range(overlap):
        weight[i, :] *= ramp[i]  # top
        weight[-(i + 1), :] *= ramp[i]  # bottom
        weight[:, i] *= ramp[i]  # left
        weight[:, -(i + 1)] *= ramp[i]  # right

    return weight


class CorridorKeyEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        low_vram: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self.low_vram = low_vram

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.model = self._load_model()

    def _load_model(self) -> GreenFormer:
        print(f"Loading CorridorKey from {self.checkpoint_path}...")
        # Initialize Model (Hiera Backbone)
        # In low-VRAM mode, run the refiner at half resolution to save ~1.5 GB
        half_res_refiner = self.low_vram
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            use_refiner=self.use_refiner,
            half_res_refiner=half_res_refiner,
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

        # --- OPTIMIZATION: Convert model weights to fp16 on GPU ---
        # Saves ~0.3-0.5 GB VRAM. The model already runs under autocast(fp16),
        # so keeping weights in fp16 avoids the fp32→fp16 cast overhead too.
        if self.device.type in ("cuda", "mps"):
            model = model.half()
            print("Model weights converted to fp16 for VRAM savings.")

        # --- OPTIMIZATION: Enable cuDNN benchmark for fixed input sizes ---
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        return model

    @torch.inference_mode()
    def _run_model(self, inp_t: torch.Tensor, refiner_scale: float = 1.0) -> dict[str, torch.Tensor]:
        """Run model forward pass with autocast. Separated for tiling reuse."""
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

        return out

    def _prepare_input_tensor(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        target_size: int,
        input_is_linear: bool = False,
    ) -> torch.Tensor:
        """Resize, normalize, and create input tensor for the model."""
        if input_is_linear:
            img_resized_lin = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            img_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        img_norm = (img_resized - self.mean) / self.std
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1)  # [H, W, 4]
        inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        return inp_t

    def _postprocess(
        self,
        pred_alpha: torch.Tensor,
        pred_fg: torch.Tensor,
        h: int,
        w: int,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """Post-process model outputs to final composited results.

        Uses in-place deletion to minimize peak CPU RAM usage.
        """
        # 6. Post-Process (Resize Back to Original Resolution)
        res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
        res_alpha = cv2.resize(res_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg = cv2.resize(res_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        # A. Clean Matte (Auto-Despeckle)
        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = res_alpha

        # B. Despill FG (res_fg is sRGB) — keep a reference before overwriting
        fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)

        # C. Premultiply (for EXR Output — Linear space)
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)

        # D. Pack RGBA — all channels Linear Float
        processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)
        del fg_premul_lin  # Free immediately

        # 7. Composite (on Checkerboard)
        bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        bg_lin = cu.srgb_to_linear(bg_srgb)
        del bg_srgb

        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)
        del fg_despilled_lin, bg_lin

        comp_srgb = cu.linear_to_srgb(comp_lin)
        del comp_lin

        return {
            "alpha": res_alpha,  # Linear, Raw Prediction
            "fg": res_fg,  # sRGB, Raw Prediction (Straight)
            "comp": comp_srgb,  # sRGB, Composite
            "processed": processed_rgba,  # Linear/Premul, RGBA, Garbage Matted & Despilled
        }

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
             dict: {'alpha': np, 'fg': np (sRGB), 'comp': np (sRGB on Gray), 'processed': np (RGBA)}
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

        # 2-5. Prepare tensor and run inference
        inp_t = self._prepare_input_tensor(image, mask_linear, self.img_size, input_is_linear)
        out = self._run_model(inp_t, refiner_scale)
        del inp_t  # Free GPU tensor immediately

        # 6-7. Post-process
        return self._postprocess(
            out["alpha"],
            out["fg"],
            h,
            w,
            fg_is_straight=fg_is_straight,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
        )

    @torch.inference_mode()
    def process_frame_tiled(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 128,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """Process a frame using tiled inference to drastically reduce VRAM.

        Instead of processing at full img_size (e.g. 2048×2048) in one pass,
        the image is split into overlapping tiles of ``tile_size`` and blended
        using raised-cosine weights. This cuts activation memory by ~4× when
        going from 2048 to 1024 tiles.

        Falls back to single-pass ``process_frame`` if the image fits in one tile.

        Args:
            tile_size: Size each tile is resized to for model inference.
            overlap: Pixel overlap between adjacent tiles in the *input* image.
            (all other args: same as process_frame)
        Returns:
            dict: same contract as process_frame
        """
        # 1. Inputs Check & Normalization
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]
        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # If image is small enough, just run single-pass
        if h <= tile_size and w <= tile_size:
            return self.process_frame(
                image,
                mask_linear,
                refiner_scale=refiner_scale,
                input_is_linear=input_is_linear,
                fg_is_straight=fg_is_straight,
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size,
            )

        # Calculate tile grid over the *input* image
        stride = tile_size - overlap
        tiles_y = max(1, math.ceil((h - overlap) / stride))
        tiles_x = max(1, math.ceil((w - overlap) / stride))

        # Accumulation buffers
        alpha_acc = np.zeros((h, w, 1), dtype=np.float64)
        fg_acc = np.zeros((h, w, 3), dtype=np.float64)
        weight_acc = np.zeros((h, w, 1), dtype=np.float64)

        # Blending weight for a tile_size × tile_size region
        blend_base = _make_blend_weight(tile_size, overlap)

        print(f"Tiled inference: {tiles_y}×{tiles_x} tiles ({tile_size}px, {overlap}px overlap) over {w}×{h} image")

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Compute tile coordinates (clamp to image bounds)
                y0 = min(ty * stride, max(0, h - tile_size))
                x0 = min(tx * stride, max(0, w - tile_size))
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)

                tile_h = y1 - y0
                tile_w = x1 - x0

                tile_img = image[y0:y1, x0:x1]
                tile_mask = mask_linear[y0:y1, x0:x1]

                # Prepare tensor at tile_size for the model
                inp_t = self._prepare_input_tensor(tile_img, tile_mask, tile_size, input_is_linear)
                out = self._run_model(inp_t, refiner_scale)
                del inp_t

                # Extract raw predictions and resize to tile's actual size
                tile_alpha = out["alpha"][0].permute(1, 2, 0).float().cpu().numpy()
                tile_fg = out["fg"][0].permute(1, 2, 0).float().cpu().numpy()
                del out

                tile_alpha = cv2.resize(tile_alpha, (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4)
                tile_fg = cv2.resize(tile_fg, (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4)

                if tile_alpha.ndim == 2:
                    tile_alpha = tile_alpha[:, :, np.newaxis]

                # Blend weight for this tile (may be smaller than full tile at edges)
                bw = blend_base[:tile_h, :tile_w, np.newaxis]

                alpha_acc[y0:y1, x0:x1] += tile_alpha * bw
                fg_acc[y0:y1, x0:x1] += tile_fg * bw
                weight_acc[y0:y1, x0:x1] += bw

                # Clear GPU cache between tiles in low-VRAM mode
                if self.low_vram and self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Normalize by accumulated weights
        weight_acc = np.maximum(weight_acc, 1e-8)
        res_alpha = (alpha_acc / weight_acc).astype(np.float32)
        res_fg = (fg_acc / weight_acc).astype(np.float32)
        del alpha_acc, fg_acc, weight_acc

        # --- Post-processing (same as process_frame) ---
        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = res_alpha

        fg_despilled = cu.despill(res_fg, green_limit_mode="average", strength=despill_strength)
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
        processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)
        del fg_premul_lin

        bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
        bg_lin = cu.srgb_to_linear(bg_srgb)
        del bg_srgb

        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)
        del fg_despilled_lin, bg_lin

        comp_srgb = cu.linear_to_srgb(comp_lin)
        del comp_lin

        return {
            "alpha": res_alpha,
            "fg": res_fg,
            "comp": comp_srgb,
            "processed": processed_rgba,
        }
