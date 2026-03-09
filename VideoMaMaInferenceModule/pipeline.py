# pipeline_svd_masked.py

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

# Import necessary helpers from the original SVD pipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _append_dims,
    retrieve_timesteps,
    _resize_with_antialiasing,
)
import torch.nn.functional as F
from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pipeline_svd_masked import StableVideoDiffusionPipelineWithMask
        >>> from diffusers.utils import load_image, export_to_video

        >>> # Load your fine-tuned UNet, VAE, etc.
        >>> pipe = StableVideoDiffusionPipelineWithMask.from_pretrained(
        ...     "path/to/your/finetuned_model", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.to("cuda")

        >>> # Load the conditioning image and the mask
        >>> image = load_image("path/to/your/conditioning_image.png").resize((1024, 576))
        >>> mask = load_image("path/to/your/mask_image.png").resize((1024, 576))

        >>> # Generate frames
        >>> frames = pipe(
        ...     image=image,
        ...     mask_image=mask,
        ...     num_frames=25,
        ...     decode_chunk_size=8
        ... ).frames[0]

        >>> export_to_video(frames, "generated_video.mp4", fps=7)
        ```
"""


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for the custom Stable Video Diffusion pipeline.
    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape
            `(batch_size, num_frames, height, width, num_channels)`.
    """
    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class StableVideoDiffusionPipelineWithMask(DiffusionPipeline):
    r"""
    A custom pipeline based on Stable Video Diffusion that accepts an additional mask for conditioning.
    This pipeline is designed to work with a UNet fine-tuned to accept 12 input channels
    (4 for noise, 4 for VAE-encoded condition image, 4 for VAE-encoded mask).
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
            self,
            vae: AutoencoderKLTemporalDecoder,
            image_encoder: CLIPVisionModelWithProjection,
            unet: UNetSpatioTemporalConditionModel,
            scheduler: EulerDiscreteScheduler,
            feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
            self,
            image: PipelineImageInput,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
    ) -> torch.Tensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)

        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = torch.zeros_like(image_embeddings)

        return image_embeddings

    def _encode_vae_image(
            self,
            image: torch.Tensor,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
    ):
        image = image.to(device=device, dtype=torch.float16)
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
        return image_latents

    def _get_add_time_ids(
            self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        return add_time_ids

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14):
        latents = latents.flatten(0, 1).to(dtype=torch.float16)
        latents = 1 / self.vae.config.scaling_factor * latents
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            frame = self.vae.decode(latents[i: i + decode_chunk_size], num_frames=num_frames_in).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
            self,
            batch_size: int,
            num_frames: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.Tensor] = None,
            initial_latents: Optional[torch.Tensor] = None,
            denoising_strength: float = 1.0,
            timestep: Optional[torch.Tensor] = None,
    ):
        num_channels_latents = self.unet.config.out_channels
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if initial_latents is not None:
            # Noise is added to the initial latents
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # Get the initial latents at the given timestep
            latents = self.scheduler.add_noise(initial_latents, noise, timestep)
        else:
            # Standard pure noise generation
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device)
            # Scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    def _encode_video_vae(
            self,
            video_frames: torch.Tensor,  # Expects (B, F, C, H, W)
            device: Union[str, torch.device],
    ):
        video_frames = video_frames.to(device=device, dtype=self.vae.dtype)
        batch_size, num_frames = video_frames.shape[:2]

        # Reshape for VAE encoding
        video_frames_reshaped = video_frames.reshape(batch_size * num_frames, *video_frames.shape[2:])  # (B*F, C, H, W)
        latents = self.vae.encode(video_frames_reshaped).latent_dist.sample()  # (B*F, C_latent, H_latent, W_latent)

        # Reshape back to video format
        latents = latents.reshape(batch_size, num_frames, *latents.shape[1:])  # (B, F, C_latent, H_latent, W_latent)

        return latents

    @torch.no_grad()
    def __call__(
            self,
            image: Union[List[PIL.Image.Image], torch.Tensor],
            mask_image: Union[List[PIL.Image.Image], torch.Tensor],
            alpha_matte_image: Optional[Union[List[PIL.Image.Image], torch.Tensor]] = None,
            denoising_strength: float = 0.7,
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 30,
            sigmas: Optional[List[float]] = None,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.02,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            mask_noise_strength: float = 0.0,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if num_frames is None:
            if isinstance(image, list):
                num_frames = len(image)
            else:
                num_frames = self.unet.config.num_frames

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(image, height, width)
        self.check_inputs(mask_image, height, width)
        if alpha_matte_image:
            self.check_inputs(alpha_matte_image, height, width)

        batch_size = 1
        device = self._execution_device
        dtype = self.unet.dtype

        image_for_clip = image[0] if isinstance(image, list) else image[0]
        image_embeddings = self._encode_image(image_for_clip, device, num_videos_per_prompt)

        fps = fps - 1

        image_tensor = self.video_processor.preprocess(image, height=height, width=width).to(device).unsqueeze(0)
        mask_tensor = self.video_processor.preprocess(mask_image, height=height, width=width).to(device).unsqueeze(0)

        noise = randn_tensor(image_tensor.shape, generator=generator, device=device, dtype=dtype)
        image_tensor = image_tensor + noise_aug_strength * noise

        conditional_latents = self._encode_video_vae(image_tensor, device)
        conditional_latents = conditional_latents / self.vae.config.scaling_factor

        if self.unet.config.in_channels == 12:
            mask_latents = self._encode_video_vae(mask_tensor, device)
            mask_latents = mask_latents / self.vae.config.scaling_factor
        elif self.unet.config.in_channels == 9:
            mask_tensor_gray = mask_tensor.mean(dim=2, keepdim=True)
            binarized_mask = (mask_tensor_gray > 0.0).to(dtype)
            b, f, c, h, w = binarized_mask.shape
            binarized_mask_reshaped = binarized_mask.reshape(b * f, c, h, w)
            target_size = (height // self.vae_scale_factor, width // self.vae_scale_factor)
            interpolated_mask = F.interpolate(
                binarized_mask_reshaped,
                size=target_size,
                mode='nearest',
            )
            mask_latents = interpolated_mask.reshape(b, f, *interpolated_mask.shape[1:])
        else:
            raise ValueError(f"Unsupported number of UNet input channels: {self.unet.config.in_channels}.")

        if mask_noise_strength > 0.0:
            mask_noise = randn_tensor(mask_latents.shape, generator=generator, device=device, dtype=dtype)
            mask_latents = mask_latents + mask_noise_strength * mask_noise

        added_time_ids = self._get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, image_embeddings.dtype, batch_size, num_videos_per_prompt
        )
        added_time_ids = added_time_ids.to(device)

        # --- MODIFIED FOR ALPHA MATTE REFINEMENT ---
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        initial_latents = None

        if alpha_matte_image is not None:
            alpha_matte_tensor = self.video_processor.preprocess(alpha_matte_image, height=height, width=width).to(
                device).unsqueeze(0)
            initial_latents = self._encode_video_vae(alpha_matte_tensor, device)
            initial_latents = initial_latents / self.vae.config.scaling_factor

            # Adjust the number of steps and the timesteps to start from
            t_start = max(num_inference_steps - int(num_inference_steps * denoising_strength), 0)
            timesteps = timesteps[t_start:]
            # We need the first timestep to add the correct amount of noise
            start_timestep = timesteps[0]
        else:
            start_timestep = timesteps[0]  # Not used, but for clarity

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
            initial_latents=initial_latents,
            denoising_strength=denoising_strength,
            timestep=start_timestep if initial_latents is not None else None,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat([latent_model_input, conditional_latents, mask_latents], dim=2)

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=image_embeddings, added_time_ids=added_time_ids,
                    return_dict=False
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames
        return StableVideoDiffusionPipelineOutput(frames=frames)


class StableVideoDiffusionPipelineOnestepWithMask(DiffusionPipeline):
    r"""
    A custom pipeline based on Stable Video Diffusion that accepts an additional mask for conditioning.
    This pipeline is designed to work with a UNet fine-tuned to accept 12 input channels
    (4 for noise, 4 for VAE-encoded condition image, 4 for VAE-encoded mask).
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
            self,
            vae: AutoencoderKLTemporalDecoder,
            image_encoder: CLIPVisionModelWithProjection,
            unet: UNetSpatioTemporalConditionModel,
            scheduler: EulerDiscreteScheduler,
            feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
            self,
            image: PipelineImageInput,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
    ) -> torch.Tensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)

        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = torch.zeros_like(image_embeddings)

        return image_embeddings

    def _encode_vae_image(
            self,
            image: torch.Tensor,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
    ):
        image = image.to(device=device, dtype=torch.float16)
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
        return image_latents

    def _get_add_time_ids(
            self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        return add_time_ids

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14):
        latents = latents.flatten(0, 1).to(dtype=torch.float16)
        latents = 1 / self.vae.config.scaling_factor * latents
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            frame = self.vae.decode(latents[i: i + decode_chunk_size], num_frames=num_frames_in).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
            self,
            batch_size: int,
            num_frames: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.Tensor] = None,
    ):
        # The number of channels for the initial noise is based on the UNet's out_channels
        num_channels_latents = self.unet.config.out_channels
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"batch size {batch_size} must match the length of the generators {len(generator)}.")

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_video_vae(
            self,
            video_frames: torch.Tensor,  # Expects (B, F, C, H, W)
            device: Union[str, torch.device],
    ):
        video_frames = video_frames.to(device=device, dtype=self.vae.dtype)
        batch_size, num_frames = video_frames.shape[:2]

        # Reshape for VAE encoding
        video_frames_reshaped = video_frames.reshape(batch_size * num_frames, *video_frames.shape[2:])  # (B*F, C, H, W)
        latents = self.vae.encode(video_frames_reshaped).latent_dist.sample()  # (B*F, C_latent, H_latent, W_latent)

        # Reshape back to video format
        latents = latents.reshape(batch_size, num_frames, *latents.shape[1:])  # (B, F, C_latent, H_latent, W_latent)

        return latents

    @torch.no_grad()
    def __call__(
            self,
            image: Union[List[PIL.Image.Image], torch.Tensor],
            mask_image: Union[List[PIL.Image.Image], torch.Tensor],
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.0,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            mask_noise_strength: float = 0.0,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if num_frames is None:
            if isinstance(image, list):
                num_frames = len(image)
            else:
                num_frames = self.unet.config.num_frames

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(image, height, width)
        self.check_inputs(mask_image, height, width)
        if isinstance(image, list) and isinstance(mask_image, list):
            if len(image) != len(mask_image):
                raise ValueError("`image` and `mask_image` must have the same number of frames.")
            if num_frames != len(image):
                logger.warning(
                    f"Mismatch between `num_frames` ({num_frames}) and number of input images ({len(image)}). Using {len(image)}.")
                num_frames = len(image)

        batch_size = 1
        device = self._execution_device
        dtype = self.unet.dtype

        image_for_clip = image[0] if isinstance(image, list) else image[0]
        image_embeddings = self._encode_image(image_for_clip, device, num_videos_per_prompt)

        fps = fps - 1

        image_tensor = self.video_processor.preprocess(image, height=height, width=width).to(device).unsqueeze(0)
        mask_tensor = self.video_processor.preprocess(mask_image, height=height, width=width).to(
            device).unsqueeze(0)

        noise = randn_tensor(image_tensor.shape, generator=generator, device=device, dtype=dtype)
        image_tensor = image_tensor + noise_aug_strength * noise

        conditional_latents = self._encode_video_vae(image_tensor, device)
        conditional_latents = conditional_latents / self.vae.config.scaling_factor

        if self.unet.config.in_channels == 12:
            mask_latents = self._encode_video_vae(mask_tensor, device)
            mask_latents = mask_latents / self.vae.config.scaling_factor
        elif self.unet.config.in_channels == 9:
            mask_tensor_gray = mask_tensor.mean(dim=2, keepdim=True)
            binarized_mask = (mask_tensor_gray > 0.0).to(dtype)
            b, f, c, h, w = binarized_mask.shape
            binarized_mask_reshaped = binarized_mask.reshape(b * f, c, h, w)
            target_size = (height // self.vae_scale_factor, width // self.vae_scale_factor)
            interpolated_mask = F.interpolate(
                binarized_mask_reshaped,
                size=target_size,
                mode='nearest',
            )
            mask_latents = interpolated_mask.reshape(b, f, *interpolated_mask.shape[1:])
        else:
            raise ValueError(
                f"Unsupported number of UNet input channels: {self.unet.config.in_channels}. "
                "This pipeline only supports 9 (for interpolated mask) or 12 (for VAE mask)."
            )

        if mask_noise_strength > 0.0:
            mask_noise = randn_tensor(mask_latents.shape, generator=generator, device=device, dtype=dtype)
            mask_latents = mask_latents + mask_noise_strength * mask_noise

        added_time_ids = self._get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, image_embeddings.dtype, batch_size, num_videos_per_prompt
        )
        added_time_ids = added_time_ids.to(device)

        # **MODIFIED FOR SINGLE-STEP**: Prepare initial noise
        num_channels_latents = self.unet.config.out_channels
        shape = (
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # **MODIFIED FOR SINGLE-STEP**: Set a fixed high timestep
        timestep = torch.tensor([1.0], dtype=dtype, device=device)  # Use a high sigma value

        # **MODIFIED FOR SINGLE-STEP**: Single forward pass
        latent_model_input = torch.cat([latents, conditional_latents, mask_latents], dim=2)

        noise_pred = self.unet(
            latent_model_input, timestep, encoder_hidden_states=image_embeddings, added_time_ids=added_time_ids,
            return_dict=False
        )[0]

        # The model's prediction is the final denoised latent
        denoised_latents = noise_pred

        frames = self.decode_latents(denoised_latents, num_frames, decode_chunk_size)
        frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames
        return StableVideoDiffusionPipelineOutput(frames=frames)


class StableVideoDiffusionPipelineWithCrossAtnnMask(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
            self,
            vae: AutoencoderKLTemporalDecoder,
            unet: UNetSpatioTemporalConditionModel,
            scheduler: EulerDiscreteScheduler,
            mask_projector: torch.nn.Module,
            # CLIP models are not strictly needed for inference if embeddings are not used
            image_encoder: CLIPVisionModelWithProjection = None,
            feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            mask_projector=mask_projector,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)

    def _encode_image_vae(self, image: torch.Tensor, device: Union[str, torch.device]):
        image = image.to(device=device, dtype=self.vae.dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        return latent

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int):
        latents = latents.flatten(0, 1).to(dtype=torch.float16)
        latents = 1 / self.vae.config.scaling_factor * latents
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            frame = self.vae.decode(latents[i: i + decode_chunk_size], num_frames=decode_chunk_size).sample
            frames.append(frame)

        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()
        return frames

    def _encode_video_vae(
            self,
            video_frames: torch.Tensor,  # Expects (B, F, C, H, W)
            device: Union[str, torch.device],
    ):
        video_frames = video_frames.to(device=device, dtype=self.vae.dtype)
        batch_size, num_frames = video_frames.shape[:2]

        # Reshape for VAE encoding
        video_frames_reshaped = video_frames.reshape(batch_size * num_frames, *video_frames.shape[2:])  # (B*F, C, H, W)
        latents = self.vae.encode(video_frames_reshaped).latent_dist.sample()  # (B*F, C_latent, H_latent, W_latent)

        # Reshape back to video format
        latents = latents.reshape(batch_size, num_frames, *latents.shape[1:])  # (B, F, C_latent, H_latent, W_latent)

        return latents

    @torch.no_grad()
    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],  # Static image for appearance
            mask_image: List[PIL.Image.Image],  # Video mask for motion
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 25,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.0,  # Noise is added to latents now
            decode_chunk_size: Optional[int] = 8,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ):
        device = self._execution_device
        dtype = self.unet.dtype
        num_frames = num_frames if num_frames is not None else len(mask_image)
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. PREPARE STATIC IMAGE CONDITION
        image_tensor = self.video_processor.preprocess(image, height, width).to(device).unsqueeze(0)
        conditional_latents = self._encode_video_vae(image_tensor, device)
        conditional_latents = conditional_latents / self.vae.config.scaling_factor

        # 2. PREPARE MASK MOTION CONDITION
        mask_tensor = self.video_processor.preprocess(mask_image, height, width)
        if mask_tensor.shape[1] > 1:
            mask_tensor = mask_tensor.mean(dim=1, keepdim=True)

        # Reshape for projector: (T, C, H, W)
        mask_for_projection = rearrange(mask_tensor, "f c h w -> f c h w").to(device, dtype)
        encoder_hidden_states = self.mask_projector(mask_for_projection)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # (T, 1, D)
        # Add batch dimension for UNet
        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)  # (1, T, 1, D)
        # The UNet will handle flattening this to (B*T, 1, D) where B=1
        # To be safe, we pass it pre-flattened.
        encoder_hidden_states = rearrange(encoder_hidden_states, "b f s d -> (b f) s d")

        # 3. PREPARE LATENTS
        shape = (1, num_frames, self.unet.config.out_channels, height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if noise_aug_strength > 0:
            latents += noise_aug_strength * randn_tensor(latents.shape, generator=generator, device=device,
                                                         dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        # 4. GET ADDED TIME IDS
        # For pipeline, batch size is 1
        added_time_ids = [fps - 1, motion_bucket_id, 0.0]  # noise_aug_strength for add_time_ids is 0 for inference
        added_time_ids = torch.tensor([added_time_ids], dtype=dtype, device=device)

        # 5. DENOISING LOOP
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                unet_input = torch.cat([latent_model_input, conditional_latents], dim=2)

                noise_pred = self.unet(
                    unet_input, t, encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids
                ).sample

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.update()

        # 6. DECODE
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)

        if not return_dict:
            return (frames,)
        return StableVideoDiffusionPipelineOutput(frames=frames)


# pipeline.py

import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from torchvision import transforms
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class VideoInferencePipeline:
    """
    A reusable pipeline for single-step video diffusion inference.

    This class encapsulates the models and the core inference logic,
    separating it from data loading and saving, which can vary between tasks.
    """

    def __init__(self, base_model_path: str, unet_checkpoint_path: str, device: str = "cuda",
                 weight_dtype: torch.dtype = torch.float16):
        """
        Loads all necessary models into memory.

        Args:
            base_model_path (str): Path to the base Stable Video Diffusion model.
            unet_checkpoint_path (str): Path to the fine-tuned UNet checkpoint.
            device (str): The device to run models on ('cuda' or 'cpu').
            weight_dtype (torch.dtype): The precision for model weights (float16 or bfloat16).
        """
        logger.info("--- Initializing Inference Pipeline and Loading Models ---")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weight_dtype = weight_dtype

        # Load models from pretrained paths
        try:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_model_path, subfolder="feature_extractor")
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_model_path,
                                                                               subfolder="image_encoder",
                                                                               variant="fp16")
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(base_model_path, subfolder="vae", variant="fp16")
            self.unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_checkpoint_path, subfolder="unet")
        except Exception as e:
            raise IOError(f"Fatal error loading models: {e}")

        # Move models to the specified device and set to evaluation mode
        # CLIP must run in FP32 to avoid CUBLAS errors
        self.image_encoder.to(self.device, dtype=torch.float32).eval()
        # VAE must also run in FP32 to avoid CUBLAS errors
        self.vae.to(self.device, dtype=torch.float32).eval()
        self.unet.to(self.device, dtype=self.weight_dtype).eval()

        logger.info("--- Models Loaded Successfully on %s ---", self.device)

    def run(self, cond_frames, mask_frames, seed=42, mask_cond_mode="vae", fps=7, motion_bucket_id=127,
            noise_aug_strength=0.0):
        """
        Runs the core inference process on a sequence of conditioning and mask frames.

        Args:
            cond_frames (list[Image.Image]): List of PIL images for conditioning.
            mask_frames (list[Image.Image]): List of PIL images for the masks.
            seed (int): Random seed for generation.
            mask_cond_mode (str): How the mask is conditioned ("vae" or "interpolate").
            fps (int): Frames per second to condition the model with.
            motion_bucket_id (int): Motion bucket ID for conditioning.
            noise_aug_strength (float): Noise augmentation strength.

        Returns:
            list[Image.Image]: A list of the generated video frames as PIL Images.
        """
        # --- 1. Prepare Tensors ---
        cond_video_tensor = self._pil_to_tensor(cond_frames).to(self.device)
        mask_video_tensor = self._pil_to_tensor(mask_frames).to(self.device)

        if mask_video_tensor.shape[2] != 3:
            mask_video_tensor = mask_video_tensor.repeat(1, 1, 3, 1, 1)

        with torch.no_grad():
            # --- 2. Get CLIP Image Embeddings ---
            first_frame_tensor = cond_video_tensor[:, 0, :, :, :]
            pixel_values_for_clip = self._resize_with_antialiasing(first_frame_tensor, (224, 224))
            pixel_values_for_clip = ((pixel_values_for_clip + 1.0) / 2.0).clamp(0, 1)
            pixel_values = self.feature_extractor(images=pixel_values_for_clip, do_rescale=False, return_tensors="pt").pixel_values
            # Run CLIP in FP32
            image_embeddings = self.image_encoder(pixel_values.to(self.device, dtype=torch.float32)).image_embeds
            
            logger.debug("CLIP Embeds Max: %.4f, Mean: %.4f", image_embeddings.max().item(), image_embeddings.mean().item())

            # Setup for UNet which uses weight_dtype (likely FP16)
            image_embeddings = image_embeddings.to(dtype=self.weight_dtype)
            encoder_hidden_states = torch.zeros_like(image_embeddings).unsqueeze(1)

            # --- 3. Prepare Latents ---
            # VAE encoding must happen in FP32
            cond_video_tensor_fp32 = cond_video_tensor.to(dtype=torch.float32)
            cond_latents = self._tensor_to_vae_latent(cond_video_tensor_fp32)
            
            logger.debug("Cond Latents Max: %.4f, Mean: %.4f", cond_latents.max().item(), cond_latents.mean().item())

            # Cast back to weight_dtype (FP16) for UNet
            cond_latents = cond_latents.to(dtype=self.weight_dtype)
            cond_latents = cond_latents / self.vae.config.scaling_factor

            if mask_cond_mode == "vae":
                mask_video_tensor_fp32 = mask_video_tensor.to(dtype=torch.float32)
                mask_latents = self._tensor_to_vae_latent(mask_video_tensor_fp32)
                logger.debug("Mask Latents Max: %.4f, Mean: %.4f", mask_latents.max().item(), mask_latents.mean().item())
                mask_latents = mask_latents.to(dtype=self.weight_dtype)
                mask_latents = mask_latents / self.vae.config.scaling_factor
            elif mask_cond_mode == "interpolate":
                target_shape = cond_latents.shape[-2:]
                b, t, c, h, w = mask_video_tensor.shape
                mask_video_reshaped = rearrange(mask_video_tensor, "b t c h w -> (b t) c h w")
                interpolated_mask = F.interpolate(mask_video_reshaped, size=target_shape, mode='bilinear',
                                                  align_corners=False)
                mask_latents = rearrange(interpolated_mask, "(b t) c h w -> b t c h w", b=b)
            else:
                raise ValueError(f"Unknown mask_cond_mode: {mask_cond_mode}")

            # --- 4. Run UNet Single-Step Inference ---
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noisy_latents = torch.randn(cond_latents.shape, generator=generator, device=self.device,
                                        dtype=self.weight_dtype)
            timesteps = torch.full((1,), 1.0, device=self.device, dtype=torch.long)
            added_time_ids = self._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, batch_size=1)

            unet_input = torch.cat([noisy_latents, cond_latents, mask_latents], dim=2)
            pred_latents = self.unet(unet_input, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample
            
            logger.debug("Pred Latents Max: %.4f, Mean: %.4f", pred_latents.max().item(), pred_latents.mean().item())

            # --- 5. Decode Latents to Video Frames ---
            pred_latents = (1 / self.vae.config.scaling_factor) * pred_latents.squeeze(0)

            frames = []
            # Process in chunks to avoid VRAM issues, especially for long videos
            # Decode in FP32
            pred_latents_fp32 = pred_latents.to(dtype=torch.float32)
            for i in range(0, pred_latents_fp32.shape[0], 8):
                chunk = pred_latents_fp32[i: i + 8]
                decoded_chunk = self.vae.decode(chunk, num_frames=chunk.shape[0]).sample
                frames.append(decoded_chunk)

            video_tensor = torch.cat(frames, dim=0)
            logger.debug("Video Tensor (Pre-Clamp) Max: %.4f, Mean: %.4f", video_tensor.max().item(), video_tensor.mean().item())
            video_tensor = (video_tensor / 2.0 + 0.5).clamp(0, 1).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            # Return a list of PIL images
            return [transforms.ToPILImage()(frame) for frame in video_tensor]

    def _pil_to_tensor(self, frames: list[Image.Image]):
        """Converts a list of PIL images to a normalized video tensor."""
        video_tensor = torch.stack([transforms.ToTensor()(f) for f in frames]).unsqueeze(0)
        return video_tensor * 2.0 - 1.0

    def _tensor_to_vae_latent(self, t: torch.Tensor):
        """Encodes a video tensor into the VAE's latent space in chunks to avoid OOM."""
        video_length = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        
        # Process in chunks of 8
        chunk_size = 8
        latents_list = []
        
        for i in range(0, t.shape[0], chunk_size):
            chunk = t[i:i + chunk_size]
            chunk_latents = self.vae.encode(chunk).latent_dist.sample()
            latents_list.append(chunk_latents)
            
        latents = torch.cat(latents_list, dim=0)
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        return latents * self.vae.config.scaling_factor

    def _get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, batch_size):
        """Creates the additional time IDs for conditioning the UNet."""
        add_time_ids_list = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids_list)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created.")
        add_time_ids = torch.tensor([add_time_ids_list], dtype=self.weight_dtype, device=self.device)
        return add_time_ids.repeat(batch_size, 1)

    def _resize_with_antialiasing(self, input_tensor, size, interpolation="bicubic", align_corners=True):
        """
        Resizes a tensor with anti-aliasing for CLIP input, mirroring k-diffusion.
        This is a direct copy of the helper function from your original scripts.
        """
        h, w = input_tensor.shape[-2:]
        factors = (h / size[0], w / size[1])
        sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
        if (ks[0] % 2) == 0: ks = ks[0] + 1, ks[1]
        if (ks[1] % 2) == 0: ks = ks[0], ks[1] + 1

        def _compute_padding(kernel_size):
            computed = [k - 1 for k in kernel_size]
            out_padding = 2 * len(kernel_size) * [0]
            for i in range(len(kernel_size)):
                computed_tmp = computed[-(i + 1)]
                pad_front = computed_tmp // 2
                pad_rear = computed_tmp - pad_front
                out_padding[2 * i + 0] = pad_front
                out_padding[2 * i + 1] = pad_rear
            return out_padding

        def _filter2d(input_tensor, kernel):
            b, c, h, w = input_tensor.shape
            tmp_kernel = kernel[:, None, ...].to(device=input_tensor.device, dtype=input_tensor.dtype)
            tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
            height, width = tmp_kernel.shape[-2:]
            padding_shape = _compute_padding([height, width])
            input_tensor_padded = F.pad(input_tensor, padding_shape, mode="reflect")
            tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
            input_tensor_padded = input_tensor_padded.view(-1, tmp_kernel.size(0), input_tensor_padded.size(-2),
                                                           input_tensor_padded.size(-1))
            output = F.conv2d(input_tensor_padded, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
            return output.view(b, c, h, w)

        def _gaussian(window_size, sigma):
            if isinstance(sigma, float):
                sigma = torch.tensor([[sigma]])
            x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(
                sigma.shape[0], -1)
            if window_size % 2 == 0:
                x = x + 0.5
            gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
            return gauss / gauss.sum(-1, keepdim=True)

        def _gaussian_blur2d(input_tensor, kernel_size, sigma):
            if isinstance(sigma, tuple):
                sigma = torch.tensor([sigma], dtype=input_tensor.dtype)
            else:
                sigma = sigma.to(dtype=input_tensor.dtype)
            ky, kx = int(kernel_size[0]), int(kernel_size[1])
            bs = sigma.shape[0]
            kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
            kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
            out_x = _filter2d(input_tensor, kernel_x[..., None, :])
            return _filter2d(out_x, kernel_y[..., None])

        blurred_input = _gaussian_blur2d(input_tensor, ks, sigmas)
        return F.interpolate(blurred_input, size=size, mode=interpolation, align_corners=align_corners)