# Troubleshooting

Common issues, their causes, and solutions when running CorridorKey.

> **Quick fixes for the most common problems:**
> | What you're seeing | Jump to |
> |---|---|
> | Can't get it installed | [Installation Issues](#installation-issues) |
> | "CUDA out of memory" | [CUDA Out of Memory](#cuda-out-of-memory-oom) |
> | Output looks dark or wrong | [Visual Artifacts](#visual-artifacts) |
> | Green edges on subject | [Green fringing on edges](#green-fringing-on-edges) |
> | Dots/specks in the key | [Tracking markers or debris](#tracking-markers-or-debris-in-the-matte) |
> | Drag-and-drop doesn't work | [No target folder provided](#no-target-folder-provided-launcher-script) |

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Visual Artifacts](#visual-artifacts)
4. [Performance Issues](#performance-issues)
5. [Alpha Hint Generator Issues](#alpha-hint-generator-issues)
6. [Output / Compositing Issues](#output--compositing-issues)

---

## Installation Issues

### uv not found / not in PATH

**Symptom:** `Install_CorridorKey_Windows.bat` shows an error about uv not being recognized, or `uv: command not found` on Mac/Linux.

**Fix:**
1. **Windows:** The installer tries to install uv automatically. If it fails, close and reopen your terminal — uv adds itself to PATH via the registry, but existing terminal windows won't see it until restarted.
2. **Mac/Linux:** Install uv manually:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Close and reopen your terminal, then try again.

### PyTorch CUDA not available

**Symptom:** `torch.cuda.is_available()` returns `False`

**Fix:**
1. Verify you have an NVIDIA GPU with CUDA support
2. Check your CUDA version:
   ```bash
   nvidia-smi
   ```
3. Re-run `uv sync` — the `pyproject.toml` pins the correct PyTorch version with CUDA support. If you continue to have issues, try clearing the cache:
   ```bash
   uv cache clean
   uv sync
   ```

### OpenEXR import errors

**Symptom:** `cv2.imread()` returns `None` for EXR files, or errors about OpenEXR support

**Fix:** The environment variable `OPENCV_IO_ENABLE_OPENEXR` must be set **before** importing cv2:
```python
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # Must come AFTER setting the env var
```

`clip_manager.py` already does this at the top of the file. If you're writing custom scripts, ensure you follow the same order.

### timm version conflicts

**Symptom:** Errors about missing Hiera model or incompatible timm API

**Fix:** CorridorKey requires `timm==1.0.24`, which is pinned in `pyproject.toml`. Re-running `uv sync` should resolve version conflicts. If you've manually installed a different version outside of uv:
```bash
uv sync --reinstall-package timm
```

---

## Runtime Errors

### CUDA Out of Memory (OOM)

**Symptom:** `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Causes and fixes:**

1. **Not enough VRAM:** CorridorKey requires ~22.7 GB at 2048x2048.
   - Use a 24GB+ GPU (RTX 3090, 4090, 5090, A6000, etc.)
   - Close other GPU-intensive applications
   - If your GPU also drives your display, that uses ~1-2 GB of VRAM

2. **GPU driving displays:** Running on the same GPU as your OS/desktop manager leaves less free VRAM.
   - Use a secondary GPU for inference: `device='cuda:1'`
   - Or use a cloud GPU instance (Runpod, Google Colab, Lambda)

3. **Previous failed run left memory allocated:** PyTorch may not release GPU memory after an error.
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
   Or restart your Python process entirely.

4. **Running GVM or VideoMaMa:** These have their own massive VRAM requirements (80GB+ and 24GB+ respectively).

### Checkpoint not found

**Symptom:** `FileNotFoundError: Checkpoint not found: ...`

**Fix:**
1. Ensure the checkpoint file exists:
   ```bash
   ls -la CorridorKeyModule/checkpoints/
   ```
2. The file should be named `CorridorKey.pth` (or any `.pth` file — the wizard auto-detects the single checkpoint)
3. Re-download if needed:
   ```bash
   curl -L -o CorridorKeyModule/checkpoints/CorridorKey.pth \
     https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth
   ```

### Multiple checkpoints found

**Symptom:** `ValueError: Multiple checkpoints found in ...`

**Fix:** The wizard expects exactly one `.pth` file in `CorridorKeyModule/checkpoints/`. Remove or move extra checkpoint files:
```bash
# Move extras to the ignored directory
mv CorridorKeyModule/checkpoints/old_model.pth CorridorKeyModule/IgnoredCheckpoints/
```

### No target folder provided (launcher script)

**Symptom:** `[ERROR] No target folder provided.` when running the .bat or .sh script

**Fix:**
- **Windows:** Drag and drop a folder or video file onto `CorridorKey_DRAG_CLIPS_HERE_local.bat`. Do NOT double-click the .bat directly.
- **Linux/macOS:** Provide a path argument:
  ```bash
  ./CorridorKey_DRAG_CLIPS_HERE_local.sh /path/to/your/footage
  ```

### Frame count mismatch

**Symptom:** `ValueError: Clip '...': Frame count mismatch! Input: X, Alpha: Y`

**Fix:** Your Input and AlphaHint sequences have different numbers of frames. Ensure they match exactly:
```bash
ls Input/ | wc -l
ls AlphaHint/ | wc -l
```

If AlphaHint was generated by GVM/VideoMaMa and has fewer frames, re-generate it or manually pad with copies of the last frame.

---

## Visual Artifacts

### Dark fringes / crushed shadows around edges

**Likely cause:** sRGB-to-Linear conversion applied in the wrong order, or the FG pass is being treated as linear when it's actually sRGB.

**Fix:** When compositing the raw `FG/` and `Matte/` passes:
1. Convert FG from sRGB to linear **first**
2. Then premultiply by the alpha
3. Then composite over your background (also in linear)
4. Convert the final result to sRGB for display

Or simply use the `Processed/` pass, which has already been correctly converted and premultiplied.

### Bright halos / light edges around subject

**Likely cause:** Compositing in sRGB space instead of linear.

**Fix:** All compositing math (`FG * alpha + BG * (1-alpha)`) must happen in linear color space. Convert to linear before compositing, convert back after.

### Green fringing on edges

**Likely cause:** Despill strength set too low.

**Fix:**
- During the wizard, increase the despill strength (default is 10/10)
- If using the API: `engine.process_frame(..., despill_strength=1.0)`

### Tracking markers or debris in the matte

**Likely cause:** Auto-despeckle is disabled or the size threshold is too small.

**Fix:**
- Enable auto-despeckle during the wizard
- Increase the despeckle size threshold (default 400px): smaller tracking markers might need a lower threshold, larger debris needs higher

### Blocky / patchy artifacts in fine detail

**Likely cause:** The CNN refiner is disabled or its strength is too low.

**Fix:**
- Ensure `use_refiner=True` (default)
- Try increasing refiner strength above 1.0 (experimental): `refiner_scale=1.5`

### Washed out / too bright output

**Likely cause:** Double gamma correction — converting already-sRGB data through `linear_to_srgb()` again.

**Fix:** Trace the color space through your pipeline. The `FG/` pass is sRGB. If you convert it to linear and then back to sRGB for display, that's correct. If you skip the linear step and apply `linear_to_srgb()` directly, it will be too bright.

### Output is entirely black

**Likely cause:** Checkpoint not loaded correctly, or the alpha hint is entirely black (no subject detected).

**Fix:**
1. Check that the checkpoint loaded without warnings:
   ```
   Loading CorridorKey from .../CorridorKey.pth...
   ```
   If you see `[Warning] Missing keys:` with many entries, the checkpoint is wrong.
2. Verify your alpha hint actually contains white regions where the subject is.

---

## Performance Issues

### Slow inference speed

**Typical performance:** 2-5 seconds per 4K frame on an RTX 4090.

**If significantly slower:**
1. Ensure you're running on GPU, not CPU
2. Check that FP16 autocast is enabled (it is by default in the engine)
3. Close other GPU-intensive processes
4. The first frame may be slower due to CUDA kernel compilation

### High CPU usage during inference

**Expected:** The CPU is active during I/O (reading/writing frames, OpenCV operations). The GPU handles the neural network inference. This is normal for a frame-by-frame pipeline.

---

## Alpha Hint Generator Issues

### GVM: Out of Memory

**Fix:** Reduce batch parameters:
```python
processor.process_sequence(
    input_path="...",
    output_dir="...",
    num_frames_per_batch=1,    # Process one frame at a time
    decode_chunk_size=1,       # Decode one frame at a time
)
```

GVM natively requires ~80 GB VRAM. There is no way to run it on consumer GPUs without significant architecture changes.

### GVM: Missing weights

**Symptom:** Errors about missing model files in `gvm_core/weights/`

**Fix:** Download the complete weights:
```bash
uv run hf download geyongtao/gvm --local-dir gvm_core/weights
```

The weights directory should contain `vae/`, `unet/`, `scheduler/`, and other subdirectories.

### VideoMaMa: No output / empty AlphaHint

**Likely cause:** The VideoMamaMaskHint is missing, empty, or in the wrong format.

**Fix:**
1. Ensure `VideoMamaMaskHint/` exists in the shot folder and contains images
2. Masks should be binary (black/white) — the code thresholds at value 10
3. Masks must have the same frame count as the input

### VideoMaMa: Import error

**Symptom:** `Failed to import VideoMaMa`

**Fix:** VideoMaMa dependencies are included in the project's `pyproject.toml`. Run `uv sync` to ensure everything is installed, then ensure checkpoints are downloaded:
```bash
uv sync
uv run hf download SammyLim/VideoMaMa --local-dir VideoMaMaInferenceModule/checkpoints
```

---

## Output / Compositing Issues

### EXR files look wrong in Photoshop / generic viewers

**Expected:** EXR files are linear float data. Most image viewers (including Photoshop's default) don't apply the correct sRGB transform when displaying. The image will look dark/washed out.

**Fix:**
- In Photoshop: Use `Edit > Color Settings` and set the working space to Linear
- In Nuke/Fusion/Resolve: These tools handle linear EXR natively
- For quick preview, use the `Comp/` PNG output instead

### Processed pass looks correct but FG/Matte separate passes don't composite cleanly

**Likely cause:** You're compositing the sRGB FG directly with the linear alpha without converting.

**Fix:** The FG pass is in sRGB gamut. You must:
1. Convert FG to linear (`srgb_to_linear`)
2. Premultiply by the matte
3. Composite over your background (in linear)
4. Convert result to sRGB for output

### Alpha appears to have hard edges when it shouldn't

**Likely cause:** The alpha hint you provided was too precise/hard-edged, or auto-despeckle is aggressively cleaning up legitimate semi-transparent detail.

**Fix:**
- Try a softer, more feathered alpha hint
- Reduce or disable auto-despeckle
- The model works best with coarse, blurry hints — it fills in the fine detail itself

---

## Still Stuck?

1. Check the [Corridor Creates Discord](https://discord.gg/zvwUrdWXJm) for community help
2. File an issue on [GitHub](https://github.com/nikopueringer/CorridorKey/issues)
3. Review the [Color Pipeline](COLOR_PIPELINE.md) and [Architecture](ARCHITECTURE.md) docs for deeper understanding
