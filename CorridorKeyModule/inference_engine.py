from __future__ import annotations

import logging
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

from .core import color_utils as cu
from .core.model_transformer import GreenFormer

# Persist torch.compile autotune cache across runs (default is /tmp which
# gets wiped on reboot — saves 10-20 min re-autotuning on ROCm, ~30s on CUDA)
_inductor_cache = os.path.join(os.path.expanduser("~"), ".cache", "corridorkey", "inductor")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _inductor_cache)

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

        self._is_rocm = hasattr(torch.version, "hip") and torch.version.hip
        self.model = self._load_model()

        # torch.compile is tested on CUDA (Windows + Linux) and ROCm (Linux).
        # ROCm on Windows hangs during Triton kernel compilation — skip it.
        # CORRIDORKEY_SKIP_COMPILE=1 forces eager mode (useful for testing).
        skip_compile = (self._is_rocm and sys.platform == "win32") or os.environ.get("CORRIDORKEY_SKIP_COMPILE") == "1"
        if skip_compile:
            logger.info("Skipping torch.compile (eager mode)")
        elif sys.platform == "linux" or sys.platform == "win32":
            self._compile()

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

        return model

    def _compile(self):
        if self._is_rocm:
            # "default" avoids the heavy autotuning that OOM-kills 16GB cards
            # at 2048x2048. Still compiles Triton kernels, just skips the
            # exhaustive benchmarking. HIP graphs are also avoided (segfault
            # on large graphs — pytorch/pytorch#155720).
            compile_mode = "default"
        else:
            compile_mode = "max-autotune"

        try:
            if self._is_rocm:
                logger.info(
                    "Compiling model (mode=%s) — this may take 10-20 minutes on first run (ROCm). "
                    "Compiled kernels are cached for future runs.",
                    compile_mode,
                )
            else:
                logger.info("Compiling model (mode=%s)...", compile_mode)
            compiled_model = torch.compile(self.model, mode=compile_mode)
            # Trigger compilation with a dummy input (the actual compile
            # happens here, not in the torch.compile() call above)
            dummy_input = torch.zeros(
                1, 4, self.img_size, self.img_size, dtype=self.model_precision, device=self.device
            )
            with torch.inference_mode():
                compiled_model(dummy_input)
            del dummy_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = compiled_model
            logger.info("Model compiled successfully (mode=%s)", compile_mode)

        except Exception as e:
            logger.info(f"Compilation error: {e}")
            logger.warning("Model compilation failed. Falling back to eager mode.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            processed_alpha = cu.clean_matte_opencv(res_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = res_alpha

        # B. Despill FG
        # res_fg is sRGB.
        fg_despilled = cu.despill_opencv(res_fg, green_limit_mode="average", strength=despill_strength)

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
        # Resize on GPU using torchvision (much faster than cv2 at 4K)
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
            processed_alpha = cu.clean_matte_torch(alpha, despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = alpha

        # B. Despill on GPU
        processed_fg = cu.despill_torch(fg, despill_strength)

        # C. sRGB → linear on GPU
        processed_fg_lin = cu.srgb_to_linear(processed_fg)

        # D. Premultiply on GPU
        processed_fg = cu.premultiply(processed_fg_lin, processed_alpha)

        # E. Pack RGBA on GPU
        packed_processed = torch.cat([processed_fg, processed_alpha], dim=1)

        # F. Composite
        if generate_comp:
            bg_lin = cu.get_checkerboard_linear_torch(w, h, processed_fg.device)
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
    ) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
        """
        Process a single frame.
        Args:
            image: Numpy array [H, W, 3] or [B, H, W, 3] (0.0-1.0 or 0-255).
                   - If input_is_linear=False (Default): Assumed sRGB.
                   - If input_is_linear=True: Assumed Linear.
            mask_linear: Numpy array [H, W] or [B, H, W] or [H, W, 1] or [B, H, W, 1] (0.0-1.0). Assumed Linear.
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
             dict: {'alpha': np, 'fg': np (sRGB), 'comp': np (sRGB on Gray)}
        """
        torch.compiler.cudagraph_mark_step_begin()

        # If input is a single image, add batch dimension
        if image.ndim == 3:
            image = image[np.newaxis, :]
            mask_linear = mask_linear[np.newaxis, :]

        bs, h, w = image.shape[:3]

        # 1. Inputs Check & Normalization
        image = TF.to_dtype(
            torch.from_numpy(image).permute((0, 3, 1, 2)),
            self.model_precision,
            scale=True,
        ).to(self.device, non_blocking=True)
        mask_linear = TF.to_dtype(
            torch.from_numpy(mask_linear.reshape((bs, h, w, 1))).permute((0, 3, 1, 2)),
            self.model_precision,
            scale=True,
        ).to(self.device, non_blocking=True)

        inp_t = self._preprocess_input(image, mask_linear, input_is_linear)

        # Free up unused VRAM in order to keep peak usage down and avoid OOM errors
        del image, mask_linear

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

        if bs == 1:
            return out[0]

        return out
