import torch
import tqdm
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import (
    BaseOutput, 
    USE_PEFT_BACKEND,     
    is_peft_available,
    is_peft_version,
    is_torch_version,
    logging
)
from diffusers.loaders.lora_pipeline import (
    _LOW_CPU_MEM_USAGE_DEFAULT_LORA,
    StableDiffusionLoraLoaderMixin
)
from peft import LoraConfig, LoraModel, set_peft_model_state_dict
import os

import matplotlib
from typing import Union, Dict
logger = logging.get_logger(__name__)


class GVMLoraLoader(StableDiffusionLoraLoaderMixin):
    _lora_loadable_modules = ["unet"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_lora_weights(
        self, 
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], 
        adapter_name=None, 
        hotswap: bool = False,
        **kwargs
    ):

        unet_lora_config = LoraConfig.from_pretrained(pretrained_model_name_or_path_or_dict)
        checkpoint_st = os.path.join(pretrained_model_name_or_path_or_dict, "pytorch_lora_weights.safetensors")
        checkpoint_pt = os.path.join(pretrained_model_name_or_path_or_dict, "pytorch_lora_weights.pt")
        if os.path.exists(checkpoint_st):
            from safetensors.torch import load_file as _st_load

            unet_lora_ckpt = _st_load(checkpoint_st)
        else:
            unet_lora_ckpt = torch.load(checkpoint_pt)
        self.unet = LoraModel(self.unet, unet_lora_config, "default")
        set_peft_model_state_dict(self.unet, unet_lora_ckpt)


class GVMOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """
    alpha: np.ndarray
    image: np.ndarray

class GVMPipeline(DiffusionPipeline, GVMLoraLoader):
    def __init__(self, vae, unet, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae, unet=unet, scheduler=scheduler
        )

    def encode(self, input):
        num_frames = input.shape[1]
        input = input.flatten(0, 1)
        latent = self.vae.encode(input.to(self.vae.dtype)).latent_dist.mode()
        latent = latent * self.vae.config.scaling_factor
        latent = latent.reshape(-1, num_frames, *latent.shape[1:])
        return latent

    def decode(self, latents, decode_chunk_size=16):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        num_frames = latents.shape[1]
        latents = latents.flatten(0, 1)
        latents = latents / self.vae.config.scaling_factor

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            frame = self.vae.decode(
                latents[i : i + decode_chunk_size].to(self.vae.dtype),
                num_frames=num_frames_in,
            ).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch, frames, channels, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:])
        return frames.to(torch.float32)

    
    def single_infer(self, rgb, position_ids=None, num_inference_steps=None, class_labels=None, noise_type="gaussian"):
        rgb_latent = self.encode(rgb)

        self.scheduler.set_timesteps(num_inference_steps, device=rgb.device)

        if noise_type == "gaussian":
            noise_latent = torch.randn_like(rgb_latent)
            timesteps = self.scheduler.timesteps
        elif noise_type == "zeros":
            noise_latent = torch.zeros_like(rgb_latent)
            timesteps = torch.ones_like(self.scheduler.timesteps) * (self.scheduler.config.num_train_timesteps - 1) # 999
            timesteps = timesteps.long()
        else:
            raise NotImplementedError
            
        image_embeddings = torch.zeros((noise_latent.shape[0], 1, 1024)).to(
            noise_latent
        )

        for i, t in enumerate(timesteps):
            latent_model_input = noise_latent
            latent_model_input = torch.cat([latent_model_input, rgb_latent], dim=2)
            # [batch_size, num_frame, 4, h, w]
            model_output = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                position_ids=position_ids,
                class_labels=class_labels,
            ).sample

            if noise_type == 'zeros':
                noise_latent = model_output
            else:
                # compute the previous noisy sample x_t -> x_t-1
                noise_latent = self.scheduler.step(
                    model_output, t, noise_latent
                ).prev_sample

        return noise_latent

    
    def __call__(
        self,
        image,
        num_frames,
        num_overlap_frames,
        num_interp_frames,
        decode_chunk_size,
        num_inference_steps,
        use_clip_img_emb=False,
        noise_type='zeros',
        mode='matte',
        ensemble_size: int = 3,
    ):

        assert ensemble_size >= 1
        self.vae.to(dtype=torch.float16)
        class_embedding = None
        
        # (1, N, 3, H, W)
        image = image.unsqueeze(0)
        B, N = image.shape[:2]
        rgb_norm = image * 2 - 1  # [-1, 1]

        rgb = rgb_norm.expand(ensemble_size, -1, -1, -1, -1)
        if N <= num_frames:
            position_ids = torch.arange(N).unsqueeze(0).repeat(B, 1).to(rgb.device)
            position_ids = torch.zeros_like(position_ids)
            position_ids = None

            latent_all = self.single_infer(
                rgb,
                num_inference_steps=num_inference_steps,
                class_labels=class_embedding,
                position_ids=position_ids,
                noise_type=noise_type
            )
        else:
            # assert 2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
            assert num_frames % 2 == 0
            # num_interp_frames = num_frames - 2
            key_frame_indices = []
            for i in range(0, N, num_frames - num_overlap_frames):
                if (
                    i + num_frames - 1 >= N
                    or len(key_frame_indices) >= num_frames
                ):

                    # print(i)
                    pass

                key_frame_indices.append(i)
                key_frame_indices.append(min(N - 1, i + num_frames - 1))

            key_frame_indices = torch.tensor(key_frame_indices, device=rgb.device)
            
            latent_all = None
            pre_latent = None

            for i in tqdm.tqdm(range(0, len(key_frame_indices), 2)):
                position_ids = torch.arange(0, key_frame_indices[i + 1] - key_frame_indices[i] + 1).to(rgb.device)
                position_ids = position_ids.unsqueeze(0).repeat(B, 1)
                position_ids = None
                latent = self.single_infer(
                    rgb[:, key_frame_indices[i] : key_frame_indices[i + 1] + 1],
                    position_ids=position_ids,
                    num_inference_steps=num_inference_steps,
                    class_labels=class_embedding
                )

                if pre_latent is not None:
                    ratio = (
                        torch.linspace(0, 1, num_overlap_frames)
                        .to(latent)
                        .view(1, -1, 1, 1, 1)
                    )
                    try:
                        latent_all[:, -num_overlap_frames:] = latent[:,:num_overlap_frames] * ratio + latent_all[:, -num_overlap_frames:] * (1 - ratio)
                    except:
                        num_overlap_frames = min(num_overlap_frames, latent.shape[1])
                        ratio = (
                                torch.linspace(0, 1, num_overlap_frames)
                                .to(latent)
                                .view(1, -1, 1, 1, 1)
                        )
                        latent_all[:, -num_overlap_frames:] = latent[:,:num_overlap_frames] * ratio + latent_all[:, -num_overlap_frames:] * (1 - ratio)
                    latent_all = torch.cat([latent_all, latent[:,num_overlap_frames:]], dim=1)
                else:
                    latent_all = latent.clone()

                pre_latent = latent
                if rgb.device.type == "cuda":
                    torch.cuda.empty_cache()

            assert latent_all.shape[1] == image.shape[1]

        alpha = self.decode(latent_all, decode_chunk_size=decode_chunk_size)

        # (N_videos, num_frames, H, W, 3)
        alpha = alpha.mean(dim=2, keepdim=True)
        alpha, _ = torch.max(alpha, dim=0)
        alpha = torch.clamp(alpha * 0.5 + 0.5, 0.0, 1.0)

        if alpha.dim() == 5:
            alpha = alpha.squeeze(0)
        
        # (N, H, W, 3)
        image = image.squeeze(0)

        return GVMOutput(
            alpha=alpha,
            image=image,
        )