from __future__ import annotations

import logging
import math
import os
import sys
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

from .core import color_utils as cu
from .core.model_transformer import GreenFormer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_checkerboard_linear_torch(w: int, h: int, device: torch.device) -> torch.Tensor:
    """Return a cached checkerboard tensor [3, H, W] on device in linear space."""
    checker_size = 128
    y_coords = torch.arange(h, device=device) // checker_size
    x_coords = torch.arange(w, device=device) // checker_size
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    checker = ((x_grid + y_grid) % 2).float()
    # Map 0 -> 0.15, 1 -> 0.55 (sRGB), then convert to linear before caching
    bg_srgb = checker * 0.4 + 0.15  # [H, W]
    bg_srgb_3 = bg_srgb.unsqueeze(0).expand(3, -1, -1)
    return cu.srgb_to_linear(bg_srgb_3)


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

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=model_precision, device=self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=model_precision, device=self.device)

        if mixed_precision or model_precision != torch.float32:
            # Use faster matrix multiplication implementation
            # This reduces the floating point precision a little bit,
            # but it should be negligible compared to fp16 precision
            torch.set_float32_matmul_precision("high")

        self.mixed_precision = mixed_precision
        if mixed_precision and model_precision == torch.float16:
            # using mixed precision, when the precision is already fp16, is slower
            self.mixed_precision = False

        self.model_precision = model_precision

        self.model = self._load_model().to(model_precision)

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

        model = model.to(self.model_precision)

        # We only tested compilation on Windows and Linux. For other platforms compilation is disabled as a precaution.
        if sys.platform == "linux" or sys.platform == "win32":
            # Try compiling the model. Fallback to eager mode if it fails.
            try:
                compiled_model = torch.compile(model, mode="max-autotune")
                # Trigger compilation with a dummy input
                dummy_input = torch.zeros(
                    1, 4, self.img_size, self.img_size, dtype=self.model_precision, device=self.device
                ).to(memory_format=torch.channels_last)
                with torch.inference_mode():
                    compiled_model(dummy_input)
                model = compiled_model

                self._preprocess_input = torch.compile(self._preprocess_input, mode="max-autotune")
                self._despill_gpu = torch.compile(self._despill_gpu, mode="max-autotune")
                self._clean_matte_gpu = torch.compile(self._clean_matte_gpu, mode="max-autotune")

            except Exception as e:
                print(f"Model compilation failed with error: {e}")
                logger.warning("Model compilation failed. Falling back to eager mode.")
                torch.cuda.empty_cache()

        return model

    def _preprocess_input(
        self, image_batch: torch.Tensor, mask_batch_linear: torch.Tensor, input_is_linear: bool
    ) -> torch.Tensor:
        # 2. Resize to Model Size
        # If input is linear, we resize in linear to preserve energy/highlights,
        # THEN convert to sRGB for the model.
        image_batch = TF.resize(
            image_batch,
            [self.img_size, self.img_size],
            interpolation=T.InterpolationMode.BILINEAR,
        )
        if input_is_linear:
            image_batch = cu.linear_to_srgb(image_batch)

        mask_batch_linear = TF.resize(
            mask_batch_linear,
            [self.img_size, self.img_size],
            interpolation=T.InterpolationMode.BILINEAR,
        )

        # 3. Normalize (ImageNet)
        # Model expects sRGB input normalized
        image_batch = TF.normalize(image_batch, self.mean, self.std)

        # 4. Prepare Tensor
        inp_concat = torch.concat((image_batch, mask_batch_linear), -3)  # [4, H, W]

        return inp_concat

    def _postprocess_opencv(
        self,
        pred_alpha: torch.Tensor,
        pred_fg: torch.Tensor,
        w: int,
        h: int,
        fg_is_straight: bool,
        despill_strength: float,
        auto_despeckle: bool,
        despeckle_size: int,
        generate_comp: bool,
    ) -> dict[str, np.ndarray]:
        # 6. Post-Process (Resize Back to Original Resolution)
        # We use Lanczos4 for high-quality resampling to minimize blur when going back to 4K/Original.
        res_alpha = pred_alpha.permute(1, 2, 0).cpu().numpy()
        res_fg = pred_fg.permute(1, 2, 0).cpu().numpy()
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
        if generate_comp:
            bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
            bg_lin = cu.srgb_to_linear(bg_srgb)

            if fg_is_straight:
                comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
            else:
                # If premultiplied model, we shouldn't multiply again (though our pipeline forces straight)
                comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

            comp_srgb = cu.linear_to_srgb(comp_lin)
        else:
            comp_srgb = None

        return {  # type: ignore[return-value]  # cu.* returns ndarray|Tensor but inputs are always ndarray here
            "alpha": res_alpha,  # Linear, Raw Prediction
            "fg": res_fg,  # sRGB, Raw Prediction (Straight)
            "comp": comp_srgb,  # sRGB, Composite
            "processed": processed_rgba,  # Linear/Premul, RGBA, Garbage Matted & Despilled
        }

    def _postprocess_torch(
        self,
        pred_alpha: torch.Tensor,
        pred_fg: torch.Tensor,
        w: int,
        h: int,
        fg_is_straight: bool,
        despill_strength: float,
        auto_despeckle: bool,
        despeckle_size: int,
        generate_comp: bool,
    ) -> list[dict[str, np.ndarray]]:
        """Post-process on GPU, transfer final results to CPU.

        When ``sync=True`` (default), blocks until transfer completes and
        returns numpy arrays.  When ``sync=False``, starts the DMA
        non-blocking and returns a :class:`PendingTransfer` — call
        ``.resolve()`` to get the numpy dict later.
        """
        # Resize on GPU using F.interpolate (much faster than cv2 at 4K)
        alpha = TF.resize(
            pred_alpha.float(),
            [h, w],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        fg = TF.resize(
            pred_fg.float(),
            [h, w],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )

        del pred_fg, pred_alpha
        torch.cuda.empty_cache()

        # A. Clean matte
        if auto_despeckle:
            processed_alpha = self._clean_matte_gpu(alpha, despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = alpha

        # B. Despill on GPU
        processed_fg = self._despill_gpu(fg, despill_strength)

        # C. sRGB → linear on GPU
        processed_fg_lin = cu.srgb_to_linear(processed_fg)

        # D. Premultiply on GPU
        processed_fg = cu.premultiply(processed_fg_lin, processed_alpha)

        # E. Pack RGBA on GPU
        packed_processed = torch.cat([processed_fg, processed_alpha], dim=1)

        # F. Composite
        if generate_comp:
            bg_lin = _get_checkerboard_linear_torch(w, h, processed_fg.device)
            if fg_is_straight:
                comp = cu.composite_straight(processed_fg_lin, bg_lin, processed_alpha)
            else:
                comp = cu.composite_premul(processed_fg_lin, bg_lin, processed_alpha)
            comp = cu.linear_to_srgb(comp)  # [H, W, 3] opaque
        else:
            del processed_fg, processed_alpha
            comp = [None] * alpha.shape[0]  # placeholder

        alpha, fg, comp, packed_processed = (
            alpha.cpu().permute(0, 2, 3, 1).numpy(),
            fg.cpu().permute(0, 2, 3, 1).numpy(),
            comp.cpu().permute(0, 2, 3, 1).numpy() if generate_comp else comp,
            packed_processed.cpu().permute(0, 2, 3, 1).numpy(),
        )

        out = []
        for i in range(alpha.shape[0]):
            result = {
                "alpha": alpha[i],
                "fg": fg[i],
                "comp": comp[i],
                "processed": packed_processed[i],
            }
            out.append(result)
        return out

    @staticmethod
    def _clean_matte_gpu(alpha: torch.Tensor, area_threshold: int, dilation: int, blur_size: int) -> torch.Tensor:
        """
        Fully GPU matte cleanup
        """
        _device = alpha.device
        mask = alpha > 0.5  # [B, 1, H, W]

        # Find the largest connected components in the mask
        # only a limited amount of iterations is needed to find components above the area threshold
        components = cu.connected_components(mask, max_iterations=area_threshold // 8, min_component_width=2)
        sizes = torch.bincount(components.flatten())
        big_sizes = torch.nonzero(sizes >= area_threshold)

        mask = torch.zeros_like(mask).float()
        mask[torch.isin(components, big_sizes)] = 1.0

        # Dilate back to restore edges of large regions
        if dilation > 0:
            # How many applications with kernel size 5 are needed to achieve the desired dilation radius
            repeats = dilation // 2
            for _ in range(repeats):
                mask = F.max_pool2d(mask, 5, stride=1, padding=2)

        # Blur for soft edges
        if blur_size > 0:
            k = int(blur_size * 2 + 1)
            mask = TF.gaussian_blur(mask, [k, k])

        return alpha * mask

    @staticmethod
    def _despill_gpu(image: torch.Tensor, strength: float) -> torch.Tensor:
        """GPU despill — keeps data on device."""
        if strength <= 0.0:
            return image
        r, g, b = image[:, 0], image[:, 1], image[:, 2]
        limit = (r + b) / 2.0
        spill = torch.clamp(g - limit, min=0.0)
        g_new = g - spill
        r_new = r + spill * 0.5
        b_new = b + spill * 0.5
        despilled = torch.stack([r_new, g_new, b_new], dim=1)
        if strength < 1.0:
            return image * (1.0 - strength) + despilled * strength
        return despilled

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
        generate_comp: bool = True,
        post_process_on_gpu: bool = True,
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
        image_was_uint8 = image.dtype == np.uint8
        mask_was_uint8 = mask_linear.dtype == np.uint8

        # immediately casting to float is fine since fp16 can represent all uint8 values exactly
        image = torch.from_numpy(image).to(self.model_precision).to(self.device)
        mask_linear = torch.from_numpy(mask_linear).to(self.model_precision).to(self.device)
        # 1. Inputs Check & Normalization
        if image_was_uint8:
            image = image / 255.0

        if mask_was_uint8:
            mask_linear = mask_linear / 255.0

        h, w = image.shape[:2]

        # Ensure Mask Shape
        if mask_linear.ndim == 2:
            mask_linear = mask_linear.unsqueeze(-1)

        image = image.permute(2, 0, 1)  # [C, H, W]
        mask_linear = mask_linear.permute(2, 0, 1)  # [C, H, W]

        image = image.unsqueeze(0)
        mask_linear = mask_linear.unsqueeze(0)

        inp_t = self._preprocess_input(image, mask_linear, input_is_linear)

        # 5. Inference
        # Hook for Refiner Scaling
        handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            handle = self.model.refiner.register_forward_hook(scale_hook)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
            prediction = self.model(inp_t)

        if handle:
            handle.remove()

        if post_process_on_gpu:
            out = self._postprocess_torch(
                prediction["alpha"].float(),
                prediction["fg"].float(),
                w,
                h,
                fg_is_straight,
                despill_strength,
                auto_despeckle,
                despeckle_size,
                generate_comp,
            )[0]  # batch of 1, take first element
        else:
            out = self._postprocess_opencv(
                prediction["alpha"][0].float(),
                prediction["fg"][0].float(),
                w,
                h,
                fg_is_straight,
                despill_strength,
                auto_despeckle,
                despeckle_size,
                generate_comp,
            )
        return out

    @torch.inference_mode()
    def batch_process_frames(
        self,
        images: np.ndarray,
        masks_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        generate_comp: bool = True,
        post_process_on_gpu: bool = True,
    ) -> list[dict[str, np.ndarray]]:
        """
        Process a single frame.
        Args:
            images: Numpy array [B, H, W, 3] (0.0-1.0 or 0-255).
                   - If input_is_linear=False (Default): Assumed sRGB.
                   - If input_is_linear=True: Assumed Linear.
            masks_linear: Numpy array [B, H, W] or [B, H, W, 1] (0.0-1.0). Assumed Linear.
            refiner_scale: Multiplier for Refiner Deltas (default 1.0).
            input_is_linear: bool. If True, resizes in Linear then transforms to sRGB.
                             If False, resizes in sRGB (standard).
            fg_is_straight: bool. If True, assumes FG output is Straight (unpremultiplied).
                            If False, assumes FG output is Premultiplied.
            despill_strength: float. 0.0 to 1.0 multiplier for the despill effect.
            auto_despeckle: bool. If True, cleans up small disconnected components from the predicted alpha matte.
            despeckle_size: int. Minimum number of consecutive pixels required to keep an island.
            generate_comp: bool. If True, also generates a composite on checkerboard for quick checking.
            post_process_on_gpu: bool. If True, performs post-processing on GPU using PyTorch instead of OpenCV.
        Returns:
             list[dict: {'alpha': np, 'fg': np (sRGB), 'comp': np (sRGB on Gray)}]
        """
        bs, h, w = images.shape[:3]

        # 1. Inputs Check & Normalization
        images = TF.to_dtype(
            torch.from_numpy(images).permute((0, 3, 1, 2)),
            self.model_precision,
            scale=True,
        ).to(self.device, non_blocking=True)
        masks_linear = TF.to_dtype(
            torch.from_numpy(masks_linear.reshape((bs, h, w, 1))).permute((0, 3, 1, 2)),
            self.model_precision,
            scale=True,
        ).to(self.device, non_blocking=True)

        inp_t = self._preprocess_input(images, masks_linear, input_is_linear)

        # Free up unused VRAM in order to keep peak usage down and avoid OOM errors
        torch.cuda.empty_cache()

        # 5. Inference
        # Hook for Refiner Scaling
        handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:

            def scale_hook(module, input, output):
                return output * refiner_scale

            handle = self.model.refiner.register_forward_hook(scale_hook)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
            prediction = self.model(inp_t)

        # Free up unused VRAM in order to keep peak usage down and avoid OOM errors
        del inp_t

        if handle:
            handle.remove()

        if post_process_on_gpu:
            out = self._postprocess_torch(
                prediction["alpha"],
                prediction["fg"],
                w,
                h,
                fg_is_straight,
                despill_strength,
                auto_despeckle,
                despeckle_size,
                generate_comp,
            )
        else:
            # Move prediction to CPU before post-processing
            pred_alpha = prediction["alpha"].cpu().float()
            pred_fg = prediction["fg"].cpu().float()

            out = []
            for i in range(bs):
                result = self._postprocess_opencv(
                    pred_alpha[i],
                    pred_fg[i],
                    w,
                    h,
                    fg_is_straight,
                    despill_strength,
                    auto_despeckle,
                    despeckle_size,
                    generate_comp,
                )
                out.append(result)

        return out
