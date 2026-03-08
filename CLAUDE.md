# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

CorridorKey is a neural-network-based green screen keyer for professional VFX. It takes an RGB image + coarse alpha hint and produces physically unmixed straight foreground color + linear alpha channel. Native inference at 2048x2048 (Hiera backbone). Requires ~22.7GB VRAM (CUDA), also supports MPS (Apple Silicon) and an experimental MLX backend.

## Important

Be extremely concise in all interactions and commit messages. Sacrifice grammar for the sake of being concise.

## Commands

```bash
# Setup (uses uv, not pip)
uv sync --group dev

# Tests
uv run pytest                  # all tests
uv run pytest -v               # verbose
uv run pytest -m "not gpu"     # skip GPU tests (what CI runs)
uv run pytest tests/test_color_utils.py  # single file
uv run pytest -k "test_name"   # single test by name
uv run pytest --cov            # with coverage

# Lint & format
uv run ruff check              # lint
uv run ruff format --check     # format check
uv run ruff format             # auto-format

# Run the keyer
uv run python corridorkey_cli.py --action wizard --win_path <path>
# Or drag files onto CorridorKey_DRAG_CLIPS_HERE_local.sh / .bat
```

Ruff config: line-length 120, rules `E,F,W,I,B`. `gvm_core/` and `VideoMaMaInferenceModule/` are excluded from linting. CI runs lint + tests (Python 3.10, 3.13) on every PR to `main`.

## Architecture

### Entry Points
- `corridorkey_cli.py` — CLI arg parsing, env setup, interactive wizard. Imports pipeline logic from `clip_manager.py`.
- `clip_manager.py` — Core pipeline: scans directories for `Input/` (RGB) + `AlphaHint/` (BW masks), prompts for config (gamma, despill, despeckle, refiner), loops frame-by-frame through the engine.
- Launcher scripts: `.sh` / `.bat` files that invoke `corridorkey_cli.py`.

### CorridorKeyModule (inference engine)
- `inference_engine.py` — `CorridorKeyEngine`: loads model, resizes to/from 2048x2048 (Lanczos4), normalizes inputs (uint8->float), packs output passes.
- `backend.py` — Backend factory: selects Torch or MLX engine. Configured via `CORRIDORKEY_BACKEND` env var or CLI flag (`auto`/`torch`/`mlx`).
- `core/model_transformer.py` — `GreenFormer`: Hiera backbone (timm, 4-channel: RGB + alpha hint), multiscale decoders, CNN refiner head (`CNNRefinerModule`) with additive delta logits.
- `core/color_utils.py` — Compositing math: piecewise sRGB transfer functions, premultiply, luminance-preserving despill, morphological matte cleanup.

### backend/ (service layer)
Higher-level service abstraction (`CorridorKeyService`), project/clip state management, GPU job queue, ffmpeg tools, frame I/O. This layer sits above the raw inference engine.

### Optional Alpha Hint Generators
- `gvm_core/` — GVM (Generative Video Matting). Automatic, no user mask needed. ~80GB VRAM.
- `VideoMaMaInferenceModule/` — VideoMaMa. Requires user-provided `VideoMamaMaskHint/`. ~80GB VRAM.

Both invoked through `clip_manager.py` (`--action generate_alphas`). These are third-party research repos kept close to upstream — excluded from ruff enforcement.

### Device Selection
`device_utils.py` — Centralized device resolution: CLI flag > `CORRIDORKEY_DEVICE` env var > auto-detect (CUDA > MPS > CPU).

### Output Structure (per shot)
- `/Matte` — Linear alpha (half-float EXR)
- `/FG` — Straight foreground color, sRGB gamut (half-float EXR)
- `/Processed` — Linear premultiplied RGBA (half-float EXR)
- `/Comp` — Checkerboard preview (8-bit PNG)

## Philosophy

This codebase will outlive you. Every shortcut you take becomes
someone else's burden. Every hack compounds into technical debt
that slows the whole team down.

You are not just writing code. You are shaping the future of this
project. The patterns you establish will be copied. The corners
you cut will be cut again.

Fight entropy. Leave the codebase better than you found it.


## Critical Rules

1. **Color math is sacred.** Model outputs: FG is sRGB, alpha is linear. Use piecewise sRGB transfer functions from `color_utils.py`, never `pow(x, 2.2)`. "Crushed shadows" or "dark fringes" = check sRGB-to-linear conversion ordering.
2. **Model I/O is `[0.0, 1.0]` float tensors.** Always.
3. **Performance matters.** 4K video frame-by-frame. Minimize `.numpy()` transfers and unnecessary `cv2.resize` in hot loops.
4. **OpenEXR requires** `os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"` before importing cv2.
5. **Folder structure is meaningful.** The wizard expects `Input/`, `AlphaHint/`, and optionally `VideoMamaMaskHint/` subdirectories per shot.
6. **Model weights are gitignored.** `.pth`, `.safetensors`, `.ckpt`, `.bin` are all in `.gitignore`. Most tests don't need them.
