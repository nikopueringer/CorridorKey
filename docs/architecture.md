# Architecture

CorridorKey is a neural-network-based green screen removal tool. It takes an
RGB image and a "Coarse Alpha Hint" (a rough mask isolating the subject) and
produces mathematically perfect, physically unmixed Alpha and Foreground
Straight color — with the green screen unmixed from semi-transparent pixels.

For the full technical handover document aimed at AI assistants, see the
[LLM Handover](LLM_HANDOVER.md) page.

---

## The GreenFormer Model

The core architecture is called the **GreenFormer**. It combines a
vision-transformer backbone with a convolutional refiner head.

### Backbone — Hiera

The backbone is a [timm](https://github.com/huggingface/pytorch-image-models)
implementation of `hiera_base_plus_224.mae_in1k_ft_in1k`. The first layer is
patched to accept **4 input channels** (RGB + Coarse Alpha Hint) instead of the
standard 3.

### Decoders

Multiscale feature-fusion heads sit on top of the backbone and predict:

- **Coarse Alpha** (1 channel)
- **Coarse Foreground** (3 channels)

### CNN Refiner (`CNNRefinerModule`)

A custom CNN head built from dilated residual blocks. It receives the original
RGB input together with the coarse predictions from the backbone and outputs
purely **additive Delta Logits**. These deltas are applied directly to the
backbone's outputs before the final Sigmoid activation, refining edge detail
without replacing the backbone's predictions.

---

## Critical Dataflow Properties

The biggest challenge in this codebase is **color space** and **gamma math**.
When debugging compositing issues, check these rules first.

### 1. Model Input / Output — Strictly `[0.0, 1.0]` Float Tensors

- The model assumes inputs are **sRGB**.
- The predicted **Foreground** (`res['fg']`) is natively sRGB — the model is
  trained to predict the un-multiplied straight-color foreground element.
- The predicted **Alpha** (`res['alpha']`) is inherently **Linear**.

### 2. EXR Handling (the `Processed` Output Pass)

EXR files store Linear float data, premultiplied. To build the `Processed` EXR:

1. Take the sRGB foreground.
2. Convert it through `srgb_to_linear()` (the piecewise real sRGB transfer
   function defined in `color_utils.py` — **not** a pure mathematical
   γ = 2.2 curve).
3. Premultiply by the Linear Alpha.
4. Save via OpenCV with `cv2.IMWRITE_EXR_TYPE_HALF`.

!!! warning "Bug History"
    Do **not** apply a pure γ 2.2 curve. Always use the piecewise sRGB
    transfer functions in `color_utils.py`.

### 3. Inference Resizing (`img_size`)

The engine is strictly trained on **2048 × 2048** crops. In
`inference_engine.py`, `process_frame()` uses OpenCV (Lanczos4) to
upscale/downscale the user's arbitrary input resolution to 2048 × 2048, feeds
the model, and then resizes the predictions back to the original resolution.

---

## Key Source Files

| File | Responsibility |
|------|----------------|
| `CorridorKeyModule/core/model_transformer.py` | PyTorch architecture definition — Hiera backbone + CNN Refiner head. |
| `CorridorKeyModule/inference_engine.py` | `CorridorKeyEngine` class — loads weights, handles resize API, packs output passes. |
| `CorridorKeyModule/core/color_utils.py` | Pure-math compositing utilities: `srgb_to_linear()`, `premultiply()`, luminance-preserving `despill()`, morphological matte cleaning. |
| `clip_manager.py` | User-facing CLI wizard — scans directories, prompts for settings, pipes frames into the engine. |

---

## Inference Pipeline Overview

Users typically launch the system via the shell scripts
(`CorridorKey_DRAG_CLIPS_HERE_local.bat` / `.sh`) which boot the
`clip_manager.py` wizard.

1. **Scan** — Looks for folders containing an `Input` sequence (RGB) and an
   `AlphaHint` sequence (BW).
2. **Config** — Prompts for settings (gamma space, despill strength,
   auto-despeckle threshold, refiner strength).
3. **Execution** — Loops frame-by-frame, passing `[H, W, 3]` NumPy arrays to
   `engine.process_frame()`.
4. **Export** — Writes four output folders:

    | Folder | Format | Color Space |
    |--------|--------|-------------|
    | `FG/` | Half-float EXR, RGB | sRGB (convert to linear before compositing) |
    | `Matte/` | Half-float EXR, Grayscale | Linear |
    | `Processed/` | Half-float EXR, RGBA | Linear, Premultiplied |
    | `Comp/` | 8-bit PNG | sRGB composite over checkerboard (preview) |

For deeper implementation details, see the
[CorridorKeyModule README](https://github.com/nikopueringer/CorridorKey/tree/main/CorridorKeyModule)
and the [LLM Handover](LLM_HANDOVER.md) document.
