# CorridorKeyModule

A self-contained, high-performance AI Chroma Keying engine. This module provides a simple API to access the `CorridorKey` architecture (Hiera Backbone + CNN Refiner) for processing green screen footage.

## Features
*   **Resolution Independent:** Automatically resizes input images to match the native training resolution of the model (2048x2048).
*   **High Fidelity:** Preserves original input resolution using Lanczos4 resampling for final output.
*   **Robust:** Supports explicit configurations for Linear (EXR) and sRGB (PNG/MP4) source inputs.

## Installation

Dependencies for the engine are managed in the main project root `requirements.txt`.  
*(Requires PyTorch, NumPy, OpenCV, Timm)*

## Usage (GUI Wizard)

For most users, the easiest way to interact with the module is through the included wizard:
`clip_manager.py` (or dragging and dropping folders onto the `.bat` / `.sh` scripts).
The wizard handles finding the latest checkpoint automatically (either `.safetensors` or legacy `.pth`), prompting for configuration (gamma, despill strength, despeckling), and batch processing entire sequences.

## Usage (Python API)

### 1. Initialization
Initialize the engine once. Point it at your checkpoint — either the preferred `.safetensors` file or a legacy `.pth`. The engine is hardcoded to process at 2048x2048, representing the data it was trained on.

```python
from CorridorKeyModule import CorridorKeyEngine

# Initialize standard engine (CUDA)
engine = CorridorKeyEngine(
    checkpoint_path="models/latest_model.safetensors",
    device='cuda',
    img_size=2048
)
```

### 2. Processing a Frame
The engine expects inputs as Numpy Arrays (`H, W, Channels`).
*   It natively processes in **32-bit float** (`0.0 - 1.0`).
*   If you pass an **8-bit integer** (`0 - 255`) array, the engine will automatically normalize it to `0.0 - 1.0` floats for you. 
*   If you pass a **16-bit or 32-bit float** array (like an EXR), it will process it at full precision without downgrading.

```python
import cv2
import os

# Enable EXR Support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Load Image (Linear EXR - Read as 32-bit Float)
img_linear = cv2.imread("input.exr", cv2.IMREAD_UNCHANGED)
img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)

# Load Coarse Mask (Linear EXR - Read as 32-bit Float)
mask = cv2.imread("mask.exr", cv2.IMREAD_UNCHANGED)
if mask.ndim == 3: 
    mask = mask[:,:,0] # Keep single channel

# Process
result = engine.process_frame(
    img_linear_rgb, 
    mask,
    input_is_linear=True, # Critical: Tell the engine this is a Linear EXR
)

# Save Results (Preserving Float Precision as EXR)
# 'processed' contains the final RGBA composite (Linear 0-1 float)
proc_rgba = result['processed']
proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)

exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF, cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24]
cv2.imwrite("output_processed.exr", proc_bgra, exr_flags)
```

## Module Structure
*   `inference_engine.py`: The main API wrapper class `CorridorKeyEngine`. Handles automated input normalization (uint8 to float), tensor conversions, memory transfer, resizing to/from the 2K processing resolution, and packing the final analytical passes (RG, Alpha, Processed EXR, and Comp overlays).
*   `core/model_transformer.py`: The architecture definition for the PyTorch model, combining the Hiera backbone and the convolutional refiner head.
*   `core/color_utils.py`: Custom digital compositing math utilities, including logic for luminance-preserving despilling, straight/premultiplied compositing algorithms, true sRGB gamma conversions, and connected-components morphological matte cleaning.
