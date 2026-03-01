# Python API Reference

This document covers the public Python APIs for all three CorridorKey modules. Use these APIs to integrate CorridorKey into custom scripts, pipeline tools, or automation workflows.

---

## Table of Contents

1. [CorridorKeyModule](#corridorkeymodule)
2. [gvm_core](#gvm_core)
3. [VideoMaMaInferenceModule](#videomamainferencemodule)
4. [color_utils](#color_utils)
5. [Complete Examples](#complete-examples)

---

## CorridorKeyModule

### Import

```python
from CorridorKeyModule import CorridorKeyEngine
```

### CorridorKeyEngine

The main inference class. Load it once, then call `process_frame()` for each frame.

#### Constructor

```python
CorridorKeyEngine(
    checkpoint_path: str,
    device: str = 'cuda',
    img_size: int = 2048,
    use_refiner: bool = True
)
```

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_path` | str | Path to the `.pth` checkpoint file |
| `device` | str | PyTorch device (`'cuda'`, `'cuda:0'`, `'cpu'`) |
| `img_size` | int | Processing resolution (model trained at 2048) |
| `use_refiner` | bool | Enable CNN refiner module. Disable for backbone-only mode |

**Example:**
```python
engine = CorridorKeyEngine(
    checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey.pth",
    device='cuda',
    img_size=2048
)
```

#### process_frame()

```python
engine.process_frame(
    image: np.ndarray,
    mask_linear: np.ndarray,
    refiner_scale: float = 1.0,
    input_is_linear: bool = False,
    fg_is_straight: bool = True,
    despill_strength: float = 1.0,
    auto_despeckle: bool = True,
    despeckle_size: int = 400
) -> dict
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | np.ndarray | — | RGB image `[H, W, 3]`. Accepts uint8 (0-255) or float32 (0.0-1.0) |
| `mask_linear` | np.ndarray | — | Alpha hint `[H, W]` or `[H, W, 1]`. Accepts uint8 or float32 |
| `refiner_scale` | float | 1.0 | Multiplier for CNN refiner deltas. >1.0 pushes more detail |
| `input_is_linear` | bool | False | If True, resize in linear space then convert to sRGB for model |
| `fg_is_straight` | bool | True | If True, FG is un-premultiplied (standard). If False, premultiplied |
| `despill_strength` | float | 1.0 | Green spill removal strength (0.0 = none, 1.0 = full) |
| `auto_despeckle` | bool | True | Enable connected-components matte cleanup |
| `despeckle_size` | int | 400 | Minimum pixel area to keep during despeckling |

**Returns:** `dict` with keys:

| Key | Shape | Color Space | Description |
|---|---|---|---|
| `'alpha'` | `[H, W, 1]` | Linear | Raw predicted alpha matte |
| `'fg'` | `[H, W, 3]` | sRGB | Raw predicted straight foreground color |
| `'comp'` | `[H, W, 3]` | sRGB | Preview composite over checkerboard |
| `'processed'` | `[H, W, 4]` | Linear premul | Despilled, despecked RGBA for EXR export |

**Example:**
```python
import cv2
import numpy as np

# Load sRGB image
img = cv2.imread("frame.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load alpha hint
mask = cv2.imread("alpha_hint.png", cv2.IMREAD_GRAYSCALE)

# Process
result = engine.process_frame(
    img_rgb,
    mask,
    despill_strength=0.8,
    auto_despeckle=True,
    despeckle_size=400
)

# Access results
alpha = result['alpha']       # [H, W, 1] float32
fg_srgb = result['fg']       # [H, W, 3] float32
rgba_linear = result['processed']  # [H, W, 4] float32
```

---

## gvm_core

### Import

```python
from gvm_core import GVMProcessor
```

### GVMProcessor

High-level wrapper for Generative Video Matting inference.

#### Constructor

```python
GVMProcessor(
    model_base: str = None,
    unet_base: str = None,
    lora_base: str = None,
    device: str = "cuda",
    seed: int = None
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_base` | str | `gvm_core/weights/` | Path to model weights directory |
| `unet_base` | str | None | Override path for UNet weights |
| `lora_base` | str | None | Path to LoRA weights (optional) |
| `device` | str | `"cuda"` | PyTorch device |
| `seed` | int | `time.time()` | Random seed for reproducibility |

#### process_sequence()

```python
processor.process_sequence(
    input_path: str,
    output_dir: str,
    num_frames_per_batch: int = 8,
    denoise_steps: int = 1,
    max_frames: int = None,
    decode_chunk_size: int = 8,
    num_interp_frames: int = 1,
    num_overlap_frames: int = 1,
    use_clip_img_emb: bool = False,
    noise_type: str = 'zeros',
    mode: str = 'matte',
    write_video: bool = True,
    direct_output_dir: str = None
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_path` | str | — | Path to video file or image sequence directory |
| `output_dir` | str | — | Output root directory |
| `num_frames_per_batch` | int | 8 | Frames per batch (reduce if OOM) |
| `denoise_steps` | int | 1 | Denoising steps (1 is standard) |
| `max_frames` | int | None | Limit number of frames processed |
| `decode_chunk_size` | int | 8 | VAE decode batch size (reduce if OOM) |
| `num_interp_frames` | int | 1 | Interpolation frames between batches |
| `num_overlap_frames` | int | 1 | Overlap frames for temporal consistency |
| `use_clip_img_emb` | bool | False | Use CLIP image embeddings for conditioning |
| `noise_type` | str | `'zeros'` | Noise initialization type for latents |
| `mode` | str | `'matte'` | Output mode |
| `write_video` | bool | True | Also write MP4 video output |
| `direct_output_dir` | str | None | Write PNGs directly to this directory |

**Example:**
```python
processor = GVMProcessor(device="cuda")

processor.process_sequence(
    input_path="path/to/video.mp4",
    output_dir="path/to/output",
    num_frames_per_batch=4,  # Reduce for less VRAM
    denoise_steps=1
)
```

---

## VideoMaMaInferenceModule

### Import

```python
from VideoMaMaInferenceModule import (
    load_videomama_model,
    run_inference,
    extract_frames_from_video,
    save_video,
    VideoInferencePipeline
)
```

### load_videomama_model()

```python
load_videomama_model(
    base_model_path: str = None,
    unet_checkpoint_path: str = None,
    device: str = "cuda"
) -> VideoInferencePipeline
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_model_path` | str | `checkpoints/stable-video-diffusion-img2vid-xt` | Base SVD model path |
| `unet_checkpoint_path` | str | `checkpoints/VideoMaMa` | Fine-tuned UNet path |
| `device` | str | `"cuda"` | PyTorch device |

### run_inference()

```python
run_inference(
    pipeline: VideoInferencePipeline,
    input_frames: List[np.ndarray],
    mask_frames: List[np.ndarray],
    chunk_size: int = 24
) -> Generator[List[np.ndarray]]
```

**Important:** This is a **generator** that yields chunks of output frames, not a single list.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pipeline` | VideoInferencePipeline | — | Loaded pipeline from `load_videomama_model()` |
| `input_frames` | List[np.ndarray] | — | List of RGB frames `[H,W,3]` uint8 |
| `mask_frames` | List[np.ndarray] | — | List of grayscale masks `[H,W]` uint8 |
| `chunk_size` | int | 24 | Frames per inference chunk |

**Yields:** `List[np.ndarray]` — chunk of output RGB frames `[H,W,3]` uint8

### extract_frames_from_video()

```python
extract_frames_from_video(
    video_path: str,
    max_frames: int = None
) -> tuple[List[np.ndarray], float]
```

Returns a tuple of `(frames, fps)` where frames are RGB uint8 numpy arrays.

### save_video()

```python
save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float
)
```

Saves a list of RGB uint8 frames as an MP4 video.

**Complete Example:**
```python
from VideoMaMaInferenceModule import load_videomama_model, run_inference, extract_frames_from_video

# Load model
pipeline = load_videomama_model(device="cuda")

# Extract frames
input_frames, fps = extract_frames_from_video("input.mp4")

# Load masks (must match frame count)
mask_frames = [cv2.imread(f"masks/{i:05d}.png", cv2.IMREAD_GRAYSCALE)
               for i in range(len(input_frames))]

# Run inference (generator)
all_output = []
for chunk in run_inference(pipeline, input_frames, mask_frames, chunk_size=24):
    all_output.extend(chunk)
```

---

## color_utils

**File:** `CorridorKeyModule/core/color_utils.py`

All functions support both NumPy arrays and PyTorch tensors unless noted otherwise.

### Gamma Conversion

```python
from CorridorKeyModule.core import color_utils as cu

# sRGB ↔ Linear (piecewise transfer function)
linear = cu.srgb_to_linear(srgb_image)
srgb = cu.linear_to_srgb(linear_image)
```

### Compositing

```python
# Premultiply / unpremultiply
fg_premul = cu.premultiply(fg, alpha)
fg_straight = cu.unpremultiply(fg_premul, alpha)

# Composite straight FG over background
result = cu.composite_straight(fg, bg, alpha)  # FG * A + BG * (1-A)

# Composite premultiplied FG over background
result = cu.composite_premul(fg_premul, bg, alpha)  # FG + BG * (1-A)
```

### Color Processing

```python
# Green spill removal (luminance-preserving)
despilled = cu.despill(image, green_limit_mode='average', strength=1.0)
# green_limit_mode: 'average' = (R+B)/2, 'max' = max(R,B)

# RGB to YUV (Rec. 601) — PyTorch tensors only
yuv = cu.rgb_to_yuv(rgb_tensor)
```

### Morphological Operations

```python
# Clean matte (remove small disconnected components)
cleaned = cu.clean_matte(alpha_np, area_threshold=300, dilation=15, blur_size=5)

# Dilate mask
dilated = cu.dilate_mask(mask, radius=10)

# Apply garbage matte
result = cu.apply_garbage_matte(predicted_matte, garbage_matte, dilation=10)
```

### Utilities

```python
# Generate checkerboard pattern
checkerboard = cu.create_checkerboard(width, height, checker_size=64, color1=0.2, color2=0.4)
# Returns: [H, W, 3] float32
```

---

## Complete Examples

### Example 1: Single Frame Processing

```python
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from CorridorKeyModule import CorridorKeyEngine

# Initialize
engine = CorridorKeyEngine(
    checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey.pth",
    device='cuda',
    img_size=2048
)

# Load inputs
img = cv2.imread("greenscreen.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread("rough_mask.png", cv2.IMREAD_GRAYSCALE)

# Process
result = engine.process_frame(img_rgb, mask, despill_strength=0.8)

# Save EXR outputs
exr_flags = [
    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]

# Processed RGBA (ready for compositing)
proc_bgra = cv2.cvtColor(result['processed'], cv2.COLOR_RGBA2BGRA)
cv2.imwrite("output_rgba.exr", proc_bgra, exr_flags)

# Matte only
alpha = result['alpha']
if alpha.ndim == 3:
    alpha = alpha[:, :, 0]
cv2.imwrite("output_matte.exr", alpha, exr_flags)

# Preview PNG
comp_bgr = cv2.cvtColor(
    (np.clip(result['comp'], 0, 1) * 255).astype(np.uint8),
    cv2.COLOR_RGB2BGR
)
cv2.imwrite("preview.png", comp_bgr)
```

### Example 2: Batch Sequence Processing

```python
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from CorridorKeyModule import CorridorKeyEngine

engine = CorridorKeyEngine(
    checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey.pth",
    device='cuda'
)

input_dir = "shot/Input"
alpha_dir = "shot/AlphaHint"
output_dir = "shot/Output/Processed"
os.makedirs(output_dir, exist_ok=True)

input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
alpha_files = sorted([f for f in os.listdir(alpha_dir) if f.endswith('.png')])

exr_flags = [
    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]

for img_file, mask_file in zip(input_files, alpha_files):
    # Load
    img = cv2.imread(os.path.join(input_dir, img_file))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(alpha_dir, mask_file), cv2.IMREAD_GRAYSCALE)

    # Process
    result = engine.process_frame(img_rgb, mask)

    # Save
    stem = os.path.splitext(img_file)[0]
    proc_bgra = cv2.cvtColor(result['processed'], cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(os.path.join(output_dir, f"{stem}.exr"), proc_bgra, exr_flags)

    print(f"Processed: {img_file}")
```

### Example 3: Linear EXR Input

```python
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
from CorridorKeyModule import CorridorKeyEngine

engine = CorridorKeyEngine(
    checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey.pth",
    device='cuda'
)

# Load Linear EXR (32-bit float, 0.0-1.0+)
img_linear = cv2.imread("render.exr", cv2.IMREAD_UNCHANGED)
img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)

# Load mask
mask = cv2.imread("mask.exr", cv2.IMREAD_UNCHANGED)
if mask.ndim == 3:
    mask = mask[:, :, 0]

# Process with linear input flag
result = engine.process_frame(
    img_linear_rgb,
    mask,
    input_is_linear=True  # Critical: tells engine to resize in linear, then convert to sRGB
)
```

### Example 4: GVM + CorridorKey Pipeline

```python
from gvm_core import GVMProcessor
from CorridorKeyModule import CorridorKeyEngine

# Step 1: Generate alpha hints
gvm = GVMProcessor(device="cuda")
gvm.process_sequence(
    input_path="shot/Input",
    output_dir=None,
    num_frames_per_batch=1,
    write_video=False,
    direct_output_dir="shot/AlphaHint"
)

# Step 2: Run CorridorKey
engine = CorridorKeyEngine(
    checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey.pth",
    device='cuda'
)

# ... process frames as in Example 2
```
