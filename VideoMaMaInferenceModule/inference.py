"""
VideoMaMa Inference Module
Provides functions to load the model and run inference on video inputs.
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from pathlib import Path

from CorridorKeyModule.model_assets import ensure_videomama_weights

# Add current directory to path so that pipeline.py's intra-package imports
# (e.g. "from pipeline import ...") resolve when this module is imported from
# outside the VideoMaMaInferenceModule directory.  This is a workaround for the
# module's original structure — a cleaner fix would convert to proper relative
# imports throughout.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .pipeline import VideoInferencePipeline

def load_videomama_model(base_model_path: Optional[str] = None, unet_checkpoint_path: Optional[str] = None, device: str = "cpu") -> VideoInferencePipeline:
    """
    Load VideoMaMa pipeline with pretrained weights.

    Args:
        base_model_path (str, optional): Path to the base Stable Video Diffusion model. 
                                         Defaults to 'checkpoints/stable-video-diffusion-img2vid-xt' in module dir.
        unet_checkpoint_path (str, optional): Path to the fine-tuned UNet checkpoint.
                                              Defaults to 'checkpoints/VideoMaMa' in module dir.
        device (str): Device to run on ("cuda" or "cpu").

    Returns:
        VideoInferencePipeline: Loaded pipeline instance.
    """
    checkpoints_dir = os.path.join(current_dir, "checkpoints")
    if base_model_path is None or unet_checkpoint_path is None:
        ensured_base_path, ensured_unet_path = ensure_videomama_weights(checkpoints_dir)
        if base_model_path is None:
            base_model_path = str(ensured_base_path)
        if unet_checkpoint_path is None:
            unet_checkpoint_path = str(ensured_unet_path)

    print(f"Loading Base model from {base_model_path}...")
    print(f"Loading VideoMaMa UNet from {unet_checkpoint_path}...")
    
    # Check if paths exist
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    if not os.path.exists(unet_checkpoint_path):
        raise FileNotFoundError(f"UNet checkpoint path not found: {unet_checkpoint_path}")

    pipeline = VideoInferencePipeline(
        base_model_path=base_model_path,
        unet_checkpoint_path=unet_checkpoint_path,
        weight_dtype=torch.float16, # Use float16 for inference by default
        device=device
    )
    
    print("VideoMaMa pipeline loaded successfully!")
    return pipeline

def extract_frames_from_video(video_path: str, max_frames: Optional[int] = None) -> tuple[List[np.ndarray], float]:
    """
    Extract frames from video file.

    Args:
        video_path (str): Path to video file.
        max_frames (int, optional): Maximum number of frames to extract.

    Returns:
        tuple: (List of numpy arrays (H,W,3) uint8 RGB, FPS)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    
    cap.release()
    
    if max_frames and len(all_frames) > max_frames:
        frames = all_frames[:max_frames]
    else:
        frames = all_frames
    
    return frames, original_fps

def run_inference(
    pipeline: VideoInferencePipeline,
    input_frames: List[np.ndarray],
    mask_frames: List[np.ndarray],
    chunk_size: int = 24  # Adjusted default chunk size
) -> List[np.ndarray]:
    """
    Run VideoMaMa inference on video frames with mask conditioning.

    Args:
        pipeline (VideoInferencePipeline): Loaded pipeline instance.
        input_frames (List[np.ndarray]): List of RGB frames (H,W,3) uint8.
        mask_frames (List[np.ndarray]): List of mask frames (H,W) uint8 (0-255) grayscale.
        chunk_size (int): Number of frames to process at once to avoid OOM.

    Returns:
        List[np.ndarray]: List of output RGB frames (H,W,3) uint8.
    """
    if len(input_frames) != len(mask_frames):
        # Resize mask frames list to match input if needed (e.g. repeat or slice)
        # For strict correctness, we'll raise an error or warn.
        # But let's assume the user provides matching lengths or we might need to handle it.
        # Here we just raise for clarity.
        raise ValueError(f"Input frames ({len(input_frames)}) and mask frames ({len(mask_frames)}) must have same length.")

    # Convert numpy arrays to PIL Images
    frames_pil = [Image.fromarray(f) for f in input_frames]
    
    # Handle mask frames - ensure they are PIL "L" mode
    mask_frames_pil = []
    for m in mask_frames:
        if m.ndim == 3:
            # If RGB/BGR mask, convert to grayscale
            m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        mask_frames_pil.append(Image.fromarray(m, mode='L'))
    
    # Resize to model input size (1024x576 is standard for SVD)
    target_width, target_height = 1024, 576
    frames_resized = [f.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                     for f in frames_pil]
    masks_resized = [m.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                    for m in mask_frames_pil]
    
    print(f"Processing {len(frames_resized)} frames in chunks of {chunk_size}...")
    
    # Store original size for resizing back
    if not frames_pil:
        return []
        
    original_size = frames_pil[0].size
    
    for i in range(0, len(frames_resized), chunk_size):
        chunk_frames = frames_resized[i:i + chunk_size]
        chunk_masks = masks_resized[i:i + chunk_size]
        
        print(f"  Running inference on chunk {i//chunk_size + 1}/{len(frames_resized)//chunk_size + 1} ({len(chunk_frames)} frames)...")
        
        # Clear cache before each chunk
        if pipeline.device.type == "cuda":
            torch.cuda.empty_cache()
        
        chunk_output = pipeline.run(
            cond_frames=chunk_frames,
            mask_frames=chunk_masks,
            seed=42, # Fixed seed for reproducibility
            mask_cond_mode="vae"
        )
        
        # Resize back to original resolution immediately
        chunk_output_resized = [f.resize(original_size, Image.Resampling.BILINEAR) 
                                for f in chunk_output]
        
        # Convert back to numpy arrays
        chunk_output_np = [np.array(f) for f in chunk_output_resized]
        
        yield chunk_output_np

def save_video(frames: List[np.ndarray], output_path: str, fps: float):
    """
    Save frames as a video file.

    Args:
        frames (List[np.ndarray]): List of frames (RGB).
        output_path (str): Output video path.
        fps (float): Frames per second.
    """
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video to {output_path}")
