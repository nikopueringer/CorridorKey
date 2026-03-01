# CLAUDE.md - CorridorKey AI Assistant Guide

## Project Identity

**CorridorKey** is a neural-network-based green screen removal (chroma keying) tool built for professional VFX pipelines. Created by Niko Pueringer at Corridor Digital, it solves the *unmixing* problem: for every pixel (including highly transparent ones like motion blur, hair, and out-of-focus edges), the model predicts the true un-multiplied straight foreground color alongside a clean linear alpha channel. This is not a binary mask — it is physically accurate color separation.

**Repository:** `nikopueringer/CorridorKey`
**License:** Custom CC BY-NC-SA 4.0 variant (commercial *use* allowed; repackaging/reselling/paid API forbidden)
**Community:** Corridor Creates Discord — https://discord.gg/zvwUrdWXJm

**Target Audience:** Two distinct groups:
- **VFX Artists / Editors** — After Effects, DaVinci Resolve, Nuke, Premiere users who may have little or no Python experience. Route them to `docs/QUICKSTART_ARTISTS.md`.
- **Developers / ML Engineers** — Comfortable with Python, PyTorch, and command line. Route them to `docs/GETTING_STARTED.md` and `docs/API_REFERENCE.md`.

---

## Quick Reference

| Item | Detail |
|---|---|
| Language | Python 3.11+ |
| Framework | PyTorch 2.9.1, Timm 1.0.24, Diffusers, OpenCV |
| Entry point | `clip_manager.py` (CLI wizard) |
| Core engine | `CorridorKeyModule/inference_engine.py` |
| Model arch | `CorridorKeyModule/core/model_transformer.py` |
| Color math | `CorridorKeyModule/core/color_utils.py` |
| Alpha generators | `gvm_core/` (GVM) and `VideoMaMaInferenceModule/` (VideoMaMa) |
| Checkpoint | `CorridorKeyModule/checkpoints/CorridorKey.pth` (~300MB) |
| Native resolution | 2048x2048 (bilinear resize in, Lanczos4 resize out) |
| Min VRAM | ~22.7 GB (CorridorKey alone) |
| Package manager | [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`) |
| Tests | `test_vram.py` (VRAM usage verification only) |
| CI/CD | None |

---

## Architecture Overview

### The GreenFormer (`model_transformer.py`)

```
Input [B, 4, 2048, 2048]     (RGB + Coarse Alpha Hint)
         │
    ┌────▼────┐
    │  Hiera  │  timm hiera_base_plus_224, patched to 4-channel input
    │ Backbone │  Feature channels: [112, 224, 448, 896]
    └────┬────┘
         │ (4 multiscale feature maps)
    ┌────▼────┐
    │ Decoder  │  Two parallel DecoderHead instances:
    │  Heads   │  Alpha (1ch logits) + FG (3ch logits)
    └────┬────┘
         │ (Bilinear upsample to full res → Coarse Logits)
    ┌────▼────────────────┐
    │  CNN Refiner         │  7-in (3 RGB + 4 coarse probs) → 4-out delta logits
    │  (CNNRefinerModule)  │  Dilated residual blocks (d=1,2,4,8), ~65px RF
    │                      │  Output scaled 10x, whisper-initialized
    └────┬────────────────┘
         │ (Residual addition in logit space)
    ┌────▼────┐
    │ Sigmoid  │  Final alpha [0,1] + FG color [0,1] in sRGB
    └─────────┘
```

### Three-Module System

1. **CorridorKeyModule** — Core inference engine. Takes RGB + Alpha Hint → outputs straight FG (sRGB), linear alpha, premultiplied RGBA (linear EXR), and preview composite.
2. **gvm_core** — Generative Video Matting. Automatic alpha hint generation using Stable Video Diffusion. ~80GB VRAM. Licensed CC BY-NC-SA 4.0 (Zhejiang University AIM lab).
3. **VideoMaMaInferenceModule** — Mask-conditioned alpha hint generation using SVD. Requires user-drawn VideoMamaMaskHint. ~24GB+ VRAM. Licensed CC BY-NC 4.0 + Stability AI Community License (KAIST CVLAB).

---

## Critical Color Pipeline Rules

These invariants **must not be broken**. When debugging compositing issues, check these first:

1. **Model I/O is strictly `[0.0, 1.0]` float tensors.**
   - Input: sRGB color space (ImageNet-normalized: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   - Output FG (`res['fg']`): sRGB, straight (un-premultiplied)
   - Output Alpha (`res['alpha']`): Linear

2. **EXR outputs are Linear float, premultiplied.**
   - The `Processed` pass: `srgb_to_linear(fg) * alpha` packed as RGBA half-float EXR
   - Uses PXR24 compression (verified smallest working format)
   - Never apply a pure `gamma 2.2` curve — use the piecewise sRGB transfer functions in `color_utils.py`

3. **Inference resolution is fixed at 2048x2048.**
   - Input is resized via bilinear interpolation → processed at 2048x2048 → resized back via Lanczos4
   - Linear inputs are resized in linear space, then converted to sRGB before the model
   - sRGB inputs are resized directly in sRGB space

4. **Despill is luminance-preserving.**
   - Green excess `= max(0, G - (R+B)/2)` redistributed equally to R and B channels
   - Applied in sRGB space before linear conversion

5. **Auto-despeckle uses connected-components morphology.**
   - Threshold at 0.5 → find components → keep areas >= threshold → dilate → blur → multiply

---

## Directory Structure

```
CorridorKey/
├── clip_manager.py                    # CLI wizard — entry point
├── test_vram.py                       # VRAM benchmark utility
├── pyproject.toml                     # Dependencies and project metadata (uv)
│
├── CorridorKey_DRAG_CLIPS_HERE_local.sh   # Linux/macOS launcher
├── CorridorKey_DRAG_CLIPS_HERE_local.bat  # Windows launcher
├── Install_CorridorKey_Windows.bat        # Windows auto-installer
├── Install_GVM_Windows.bat                # GVM module installer
├── Install_VideoMaMa_Windows.bat          # VideoMaMa module installer
├── RunGVMOnly.sh                          # Alpha generation only
├── RunInferenceOnly.sh                    # Inference only
│
├── CorridorKeyModule/                 # Core engine
│   ├── __init__.py                    # Exports CorridorKeyEngine
│   ├── inference_engine.py            # CorridorKeyEngine class
│   ├── README.md
│   └── core/
│       ├── model_transformer.py       # GreenFormer architecture
│       └── color_utils.py             # Compositing math
│
├── gvm_core/                          # GVM alpha hint generator
│   ├── __init__.py                    # Exports GVMProcessor
│   ├── wrapper.py                     # GVMProcessor class
│   ├── LICENSE.md                     # CC BY-NC-SA 4.0
│   ├── README.md
│   └── gvm/                           # Internal diffusion package
│       ├── models/unet_spatio_temporal_condition.py
│       ├── pipelines/pipeline_gvm.py
│       └── utils/inference_utils.py
│
├── VideoMaMaInferenceModule/          # VideoMaMa alpha hint generator
│   ├── __init__.py                    # Exports load_videomama_model, etc.
│   ├── inference.py                   # Inference API
│   ├── pipeline.py                    # SVD-based pipeline
│   ├── LICENSE.md                     # CC BY-NC 4.0 + Stability AI
│   └── README.md
│
├── ClipsForInference/                 # User input staging area
├── Output/                            # Output destination
├── IgnoredClips/                      # Excluded clips
└── docs/                              # Documentation (see index below)
```

---

## Key Classes and Functions

### `CorridorKeyEngine` (`CorridorKeyModule/inference_engine.py`)
- `__init__(checkpoint_path, device='cuda', img_size=2048, use_refiner=True)`
- `process_frame(image, mask_linear, refiner_scale=1.0, input_is_linear=False, fg_is_straight=True, despill_strength=1.0, auto_despeckle=True, despeckle_size=400)` → dict with keys: `alpha`, `fg`, `comp`, `processed`

### `GreenFormer` (`CorridorKeyModule/core/model_transformer.py`)
- `forward(x)` where x is `[B, 4, H, W]` → dict with keys: `alpha`, `fg`
- Contains: `DecoderHead`, `CNNRefinerModule`, `RefinerBlock`, `MLP`

### `color_utils` (`CorridorKeyModule/core/color_utils.py`)
- `srgb_to_linear(x)` / `linear_to_srgb(x)` — piecewise sRGB transfer, supports NumPy and PyTorch
- `premultiply(fg, alpha)` / `unpremultiply(fg, alpha)`
- `composite_straight(fg, bg, alpha)` / `composite_premul(fg, bg, alpha)`
- `despill(image, green_limit_mode='average', strength=1.0)` — luminance-preserving
- `clean_matte(alpha_np, area_threshold, dilation, blur_size)` — morphological cleanup
- `dilate_mask(mask, radius)` — supports NumPy (cv2) and PyTorch (MaxPool)

### `GVMProcessor` (`gvm_core/wrapper.py`)
- `__init__(model_base=None, device="cuda")`
- `process_sequence(input_path, output_dir, num_frames_per_batch=8, denoise_steps=1, ...)`

### `VideoMaMaInferenceModule` (`VideoMaMaInferenceModule/inference.py`)
- `load_videomama_model(base_model_path=None, unet_checkpoint_path=None, device="cuda")`
- `run_inference(pipeline, input_frames, mask_frames, chunk_size=24)` — generator yielding frame chunks
- `extract_frames_from_video(video_path, max_frames=None)`

---

## Clip Manager Wizard Flow (`clip_manager.py`)

The wizard (`--action wizard`) follows this loop:

1. **Map Path** — Accepts Windows `V:\` paths and converts to Linux `/mnt/ssd-storage/`
2. **Analyze** — Detects if target is a single shot or batch of shots
3. **Organize** — Creates `Input/`, `AlphaHint/`, `VideoMamaMaskHint/` folder structure
4. **Status Loop** — Categorizes clips as READY / MASKED / RAW
5. **Actions** — `[v]` VideoMaMa, `[g]` GVM, `[i]` Inference, `[r]` Re-Scan, `[q]` Quit

The expected shot folder structure:
```
MyShot/
├── Input/              # RGB frames (PNG, EXR, etc.) or Input.mp4
├── AlphaHint/          # Coarse alpha masks (generated or manual)
├── VideoMamaMaskHint/  # Rough binary mask for VideoMaMa (optional)
└── Output/             # Created by inference
    ├── FG/             # Straight foreground color (EXR, sRGB gamut)
    ├── Matte/          # Linear alpha channel (EXR)
    ├── Processed/      # Premultiplied RGBA (EXR, Linear)
    └── Comp/           # Preview composite over checkerboard (PNG)
```

---

## Build and Run Commands

```bash
# Setup (uv handles Python installation, venv creation, and dependency resolution)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Install uv (if not already installed)
uv sync                                             # Install all dependencies

# Download model (~300MB)
# Place as: CorridorKeyModule/checkpoints/CorridorKey.pth

# Run wizard (primary usage)
uv run python clip_manager.py --action wizard --win_path /path/to/clips

# Run via launcher scripts
./CorridorKey_DRAG_CLIPS_HERE_local.sh /path/to/clips

# Generate alpha hints only (uses GVM)
uv run python clip_manager.py --action generate_alphas

# Run inference only (clips must have Input + AlphaHint)
uv run python clip_manager.py --action run_inference

# List/validate clips
uv run python clip_manager.py --action list

# Test VRAM usage
uv run python test_vram.py
```

---

## Common Pitfalls

1. **"Crushed shadows" or "dark fringes"** — Almost always an sRGB↔Linear conversion in the wrong order. Check `color_utils.py` call sequence in `inference_engine.py`.
2. **EXR not loading** — `OPENCV_IO_ENABLE_OPENEXR` must be set *before* `import cv2`. The codebase does this at the top of `clip_manager.py`.
3. **OOM errors** — The model requires ~22.7GB VRAM at 2048x2048. Reduce `img_size` if experimenting, but quality will degrade (model was trained at 2048).
4. **Checkpoint naming** — The wizard auto-detects the `.pth` file in `CorridorKeyModule/checkpoints/` but expects exactly one file there.
5. **Frame count mismatch** — Input and AlphaHint sequences must have the same number of frames.
6. **FG pass is sRGB** — The `FG/` output directory contains sRGB-gamut foreground. You must convert to linear before compositing in Nuke/Fusion.

---

## Design Principles

- **VFX-first:** All outputs target professional compositing workflows (EXR half-float, Linear color, proper premultiplication)
- **Resolution independent:** Arbitrary input → 2048x2048 processing → original resolution output
- **Offline capable:** No network calls during inference; model weights are local files
- **Modular:** The three modules (CorridorKey, GVM, VideoMaMa) are independently installable
- **Color-correct:** Uses piecewise sRGB transfer functions, never approximated gamma 2.2 curves

---

## Documentation Index

**For Artists (No Coding Required):**
- [Artist Quickstart](docs/QUICKSTART_ARTISTS.md) — zero to first key in 30 minutes, no code

**For Developers:**
- [Getting Started (Developer Guide)](docs/GETTING_STARTED.md) — installation, setup, and Python usage
- [Python API Reference](docs/API_REFERENCE.md) — full API docs for all three modules
- [Architecture Deep Dive](docs/ARCHITECTURE.md) — GreenFormer model, inference pipeline, module design

**For Everyone:**
- [Color Pipeline & Compositing Math](docs/COLOR_PIPELINE.md) — sRGB/Linear, premultiplication, output specs
- [Troubleshooting](docs/TROUBLESHOOTING.md) — common issues and fixes
- [Contributing](docs/CONTRIBUTING.md) — how to help improve the project
- [LLM Handover Guide](docs/LLM_HANDOVER.md) — technical reference for AI assistants
