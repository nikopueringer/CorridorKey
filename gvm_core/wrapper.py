import os
import os.path as osp
import cv2
import random
import logging
import time
from pathlib import Path

from easydict import EasyDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

# Relative imports from the internal gvm package
# Assuming this file is inside gvm_core/
from .gvm.pipelines.pipeline_gvm import GVMPipeline
from .gvm.utils.inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
from .gvm.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel


def seed_all(seed: int = 0):
    """Seed all random number generators for reproducibility.

    WARNING: This mutates global state — Python's random, numpy's RNG,
    and all PyTorch CUDA RNGs. Called from GVMProcessor.__init__.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def impad_multi(img, multiple=32):
    # img: (N, C, H, W)
    h, w = img.shape[2], img.shape[3]
    
    target_h = int(np.ceil(h / multiple) * multiple)
    target_w = int(np.ceil(w / multiple) * multiple)

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    # F.pad expects (padding_left, padding_right, padding_top, padding_bottom)
    padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    return padded, (pad_top, pad_left, pad_bottom, pad_right)

def sequence_collate_fn(examples):
    rgb_values = torch.stack([example["image"] for example in examples])
    rgb_values = rgb_values.to(memory_format=torch.contiguous_format).float()
    rgb_names = [example["filename"] for example in examples]
    return {'rgb_values': rgb_values, 'rgb_names': rgb_names}

class GVMProcessor:
    def __init__(self, 
                 model_base=None,
                 unet_base=None,
                 lora_base=None,
                 device="cpu",
                 seed=None):
        self.device = torch.device(device)
        
        # Resolve default weights path relative to this file
        if model_base is None:
            model_base = osp.join(osp.dirname(__file__), "weights")
            
        self.model_base = model_base
        self.unet_base = unet_base
        self.lora_base = lora_base
        
        if seed is None:
            seed = int(time.time())
        seed_all(seed)
        
        logging.info(f"Loading GVM models from {model_base}...")
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(model_base, subfolder="vae", torch_dtype=torch.float16)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_base, subfolder="scheduler")
        
        unet_folder = unet_base if unet_base is not None else model_base
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            unet_folder, 
            subfolder="unet", 
            class_embed_type=None,
            torch_dtype=torch.float16
        )

        self.pipe = GVMPipeline(vae=self.vae, unet=self.unet, scheduler=self.scheduler)
        if lora_base:
            # Check if lora_base is None or points to valid path, otherwise try default
            if lora_base is None and osp.exists(osp.join(model_base, "unet")):
                 # Often lora weights are just the unet weights in this codebase based on demo.py usage
                 pass 
            elif lora_base:
                self.pipe.load_lora_weights(lora_base)
                
        self.pipe = self.pipe.to(self.device, dtype=torch.float16)
        logging.info("Models loaded.")

    def process_sequence(self, input_path, output_dir, 
                         num_frames_per_batch=8,
                         denoise_steps=1,
                         max_frames=None,
                         decode_chunk_size=8,
                         num_interp_frames=1,
                         num_overlap_frames=1,
                         use_clip_img_emb=False,
                         noise_type='zeros',
                         mode='matte',
                         write_video=True,
                         direct_output_dir=None):
        """
        Process a single video or directory of images.
        """
        input_path = Path(input_path)
        file_name = input_path.stem
        is_video = input_path.suffix.lower() in ['.mp4', '.mkv', '.gif', '.mov', '.avi']
        
        # --- Determine Resolution & Upscaling ---
        if is_video:
            cap = cv2.VideoCapture(str(input_path))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            image_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.exr']])
            if not image_files:
                logging.warning(f"No images found in {input_path}")
                return
            # Use cv2 for EXR support if needed
            first_img_path = str(image_files[0])
            if first_img_path.lower().endswith('.exr'):
                 # import cv2 # Global import used
                 if "OPENCV_IO_ENABLE_OPENEXR" not in os.environ:
                     os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                 img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
            else:
                 img = cv2.imread(first_img_path)
                 
            if img is not None:
                orig_h, orig_w = img.shape[:2]
            else:
                orig_h, orig_w = 1080, 1920 # Fallback

        target_h = orig_h
        if target_h < 1024:
            scale_ratio = 1024 / target_h
            target_h = 1024
        
        # Calculate max resolution / long edge
        if orig_h < orig_w: # Landscape
            ratio = orig_w / orig_h
            new_long = int(1024 * ratio)
        else:
            ratio = orig_h / orig_w
            new_long = int(1024 * ratio)
            
        scale_cap = 1920
        if new_long > scale_cap:
            new_long = scale_cap
        
        max_res_param = new_long 

        transform = Compose([
            ToTensor(),
            Resize(size=1024, max_size=max_res_param, antialias=True)
        ])

        if is_video:
            reader = VideoReader(
                str(input_path), 
                max_frames=max_frames,
                transform=transform
            )
        else:
            reader = ImageSequenceReader(
                str(input_path), 
                transform=transform
            )

        # Get upscaled shape from first frame
        first_frame = reader[0]
        if isinstance(first_frame, dict):
             first_frame = first_frame['image']
        
        current_upscaled_shape = list(first_frame.shape[1:]) # H, W
        if current_upscaled_shape[0] % 2 != 0: current_upscaled_shape[0] -= 1
        if current_upscaled_shape[1] % 2 != 0: current_upscaled_shape[1] -= 1
        current_upscaled_shape = tuple(current_upscaled_shape)

        # Output preparation
        fps = reader.frame_rate if hasattr(reader, 'frame_rate') else 24.0
        
        if direct_output_dir:
            # Write directly to this folder
            os.makedirs(direct_output_dir, exist_ok=True)
            writer_alpha_seq = ImageSequenceWriter(direct_output_dir, extension='png')
            writer_alpha = None
            if write_video:
                 # Warning: direct mode might not support video naming nicely without logic
                 # Let's write video into the directory with fixed name
                 writer_alpha = VideoWriter(osp.join(direct_output_dir, f"{file_name}_alpha.mp4"), frame_rate=fps)
        else:
            # Create output directory for this specific file
            file_output_dir = osp.join(output_dir, file_name)
            os.makedirs(file_output_dir, exist_ok=True)
            logging.info(f"Processing {input_path} -> {file_output_dir}")
            
            writer_alpha = VideoWriter(osp.join(file_output_dir, f"{file_name}_alpha.mp4"), frame_rate=fps) if write_video else None
            writer_alpha_seq = ImageSequenceWriter(osp.join(file_output_dir, "alpha_seq"), extension='png')
        
        # Dataloader
        if is_video:
            dataloader = DataLoader(reader, batch_size=num_frames_per_batch)
        else:
            dataloader = DataLoader(reader, batch_size=num_frames_per_batch, collate_fn=sequence_collate_fn)

        upper_bound = 240./255.
        lower_bound = 25./ 255.

        for batch_id, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Inferencing {file_name}"):
            filenames = []
            if is_video:
                b, _, h, w = batch.shape
                for i in range(b):
                    file_id = batch_id * b + i
                    filenames.append(f"{file_id:05d}.jpg")
            else:
                filenames = batch['rgb_names']
                batch = batch['rgb_values']

            # Pad (Reflective)
            batch, pad_info = impad_multi(batch)

            # Inference
            with torch.no_grad():
                pipe_out = self.pipe(
                    batch.to(self.device, dtype=torch.float16),
                    num_frames=num_frames_per_batch,
                    num_overlap_frames=num_overlap_frames,
                    num_interp_frames=num_interp_frames,
                    decode_chunk_size=decode_chunk_size,
                    num_inference_steps=denoise_steps,
                    mode=mode,
                    use_clip_img_emb=use_clip_img_emb,
                    noise_type=noise_type,
                    ensemble_size=1,
                )
            image = pipe_out.image
            alpha = pipe_out.alpha

            # Crop padding
            out_h, out_w = image.shape[2:]
            pad_t, pad_l, pad_b, pad_r = pad_info
            
            end_h = out_h - pad_b
            end_w = out_w - pad_r
            
            image = image[:, :, pad_t:end_h, pad_l:end_w]
            alpha = alpha[:, :, pad_t:end_h, pad_l:end_w]

            # Resize to ensure exact match if there's any discrepancy
            alpha = F.interpolate(alpha, current_upscaled_shape, mode='bilinear')
            
            # Threshold
            alpha[alpha>=upper_bound] = 1.0
            alpha[alpha<=lower_bound] = 0.0

            if writer_alpha: writer_alpha.write(alpha)
            writer_alpha_seq.write(alpha, filenames=filenames)
        
        if writer_alpha: writer_alpha.close()
        writer_alpha_seq.close()
        logging.info(f"Finished {file_name}")
