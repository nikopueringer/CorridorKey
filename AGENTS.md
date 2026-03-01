# AGENTS.md

CorridorKey is an AI green screen keyer for professional VFX pipelines. It takes an RGB frame + a coarse alpha hint and produces physically accurate unmixed foreground color and linear alpha, preserving hair, motion blur, and translucency.

## Architecture

**GreenFormer** (`CorridorKeyModule/core/model_transformer.py`):
- Backbone: `hiera_base_plus_224` (timm), patched to accept 4 channels (RGB + alpha hint)
- Dual decoders: `DecoderHead` for alpha (1ch) and foreground (3ch) — multiscale feature fusion producing coarse logits
- Refiner: `CNNRefinerModule` — dilated residual blocks producing additive delta logits applied before final sigmoid
- Trained at 2048x2048

**Key files:**

| File | Role |
|---|---|
| `CorridorKeyModule/inference_engine.py` | `CorridorKeyEngine` — loads weights, handles resize + normalize + inference + post-process |
| `CorridorKeyModule/core/color_utils.py` | Compositing math: sRGB transfer, premultiply, despill, matte cleanup |
| `clip_manager.py` | Library: clip scanning, GVM/VideoMaMa orchestration, frame-by-frame inference |
| `corridorkey_cli.py` | CLI entry point: argparse commands, interactive wizard, user prompts |
| `device_utils.py` | Device selection: CUDA > MPS > CPU auto-detect, `--device` flag, `CORRIDORKEY_DEVICE` env var |

**Three modules:**
- `CorridorKeyModule/` — project-authored, the core keying engine
- `gvm_core/` — upstream-derived from [aim-uofa/GVM](https://github.com/aim-uofa/GVM), generates alpha hints automatically
- `VideoMaMaInferenceModule/` — upstream-derived from [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa), generates alpha hints from mask conditioning

## Critical Dataflow Rules

These are the rules that, when violated, produce subtle compositing bugs (crushed shadows, dark fringes, wrong premultiplication). Check these first when debugging visual artifacts.

- Model input is sRGB-normalized float `[0, 1]`. Model output: FG is sRGB (straight/unpremultiplied), alpha is linear.
- EXR output (`Processed/`) is **linear float, premultiplied**. The pipeline is: sRGB FG -> `srgb_to_linear()` -> `premultiply()` -> write EXR half-float.
- **Always use piecewise sRGB transfer** (`color_utils.linear_to_srgb` / `srgb_to_linear`), never the gamma 2.2 approximation (`x ** (1/2.2)`). Known inconsistency: `clip_manager.py` and `gvm_core/utils/inference_utils.py` still use gamma 2.2 for VideoMaMa/GVM preprocessing — documented in `tests/test_gamma_consistency.py`.
- Inference resolution is fixed at 2048x2048. `process_frame()` resizes input down, runs the model, resizes predictions back with Lanczos4. Frame data is passed as `[H, W, 3]` numpy arrays.
- `os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"` must be set before any `cv2` import.
- **Output directories per clip:** `FG/` (half-float EXR, sRGB straight), `Matte/` (half-float EXR, grayscale linear), `Processed/` (half-float EXR, RGBA linear premultiplied), `Comp/` (8-bit PNG, sRGB over checkerboard).
- **Debugging "crushed shadows" or "dark fringes":** almost certainly an sRGB-to-linear conversion happening in the wrong order. Check the call sequence in `color_utils.py` and `inference_engine.py`.

## Dev Workflow

```bash
uv sync --group dev          # install all deps + dev tools
uv run pytest                # run tests (~3s, no GPU needed)
uv run pytest -m "not gpu"   # explicit skip of GPU tests
uv run ruff check            # lint
uv run ruff format --check   # format check
```

Entry point: `uv run python corridorkey_cli.py <subcommand>` or launcher scripts (`.bat`/`.sh`).
Model weights (~300MB) are not in the repo. Most tests don't need them.

## Code Conventions

- Python 3.10+, `from __future__ import annotations` in all CorridorKey-authored files
- Modern typing: `str | None` not `Optional[str]`, `list[str]` not `List[str]`
- Google-style docstrings with `Args:` and `Returns:` sections
- Ruff: `select = ["E", "F", "W", "I", "B"]`, line length 120
- Third-party code (`gvm_core/`, `VideoMaMaInferenceModule/`) excluded from ruff
- `logging` module in library code, not `print()`. Raw `print()` is acceptable in CLI layer only.
- No `sys.exit()` in library code — raise exceptions instead
- `os.path` throughout (not `pathlib`) — this is the established convention
- Diagrams: Mermaid only, no ASCII art
- **Performance-sensitive code:** The inference loop processes every frame at 2048x2048. Avoid unnecessary `.numpy()` transfers, redundant `cv2.resize` calls, or device round-trips in hot loops.

## Landmines

- **Never modify `gvm_core/` or `VideoMaMaInferenceModule/`** unless fixing a crash or guarding platform-specific calls (e.g., `torch.cuda.empty_cache()`). These are upstream-derived. See `.claude/rules/third-party-code.md`.
- **Three functions named `run_inference`**: (1) `clip_manager.run_inference()` — library, processes clips; (2) `VideoMaMaInferenceModule.inference.run_inference` — locally imported inside `run_videomama()`; (3) CLI subcommand in `corridorkey_cli.py`. The CLI function must use a different name (e.g., `run_inference_cmd`).
- **`run_inference()` contains `input()` calls** for user settings (gamma, despill, despeckle, refiner). For non-interactive use, pass an `InferenceSettings` dataclass. The `input()` calls should live in the CLI layer, not the library.
- **Hardcoded path mapping**: `clip_manager.py` maps `V:\` to `/mnt/ssd-storage` (Corridor Digital's studio setup). Don't remove it, but don't assume it's universal.
- **`tqdm` stays** — used by `gvm_core/`. New CLI-layer progress should use `rich.progress`.
- **Known gamma inconsistency**: `clip_manager.py` and `gvm_core/utils/inference_utils.py` use gamma 2.2 approximation for VideoMaMa/GVM preprocessing, while `color_utils.py` uses correct piecewise sRGB. Documented in `tests/test_gamma_consistency.py`.
- **"PointRend" in old commits:** entirely replaced by the CNN Refiner. No PointRend code remains.
- **`model_transformer.py` is inference-only.** Training logic (coarse logits, `.detach()`, gradient checkpointing) was deliberately stripped. Do not add training hooks back into this file.

## Hardware

- CorridorKey: ~22.7 GB VRAM at 2048x2048 (needs 24GB+ GPU)
- GVM: ~80 GB VRAM
- VideoMaMa: ~80 GB VRAM (community optimizations not yet integrated)
- GPU tests use `@pytest.mark.gpu` — auto-skipped when neither CUDA nor MPS is available
- MPS (Apple Silicon) support is experimental; `PYTORCH_ENABLE_MPS_FALLBACK=1` may be needed
