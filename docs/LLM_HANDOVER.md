# CorridorKey: LLM Handover Guide

This document supplements [CLAUDE.md](../CLAUDE.md) with deeper technical detail for AI coding assistants. Read CLAUDE.md first — it has the architecture overview, color pipeline rules, directory structure, API signatures, and common pitfalls.

This document covers what CLAUDE.md does not: forward pass pseudocode, alpha generator internals, the clip manager's data model, and debugging/directive guidance.

---

## 1. Forward Pass Detail

**File:** `CorridorKeyModule/core/model_transformer.py`

```python
# Input: x [B, 4, 2048, 2048]
features = encoder(x)                    # 4 multiscale feature maps
alpha_logits = alpha_decoder(features)   # [B, 1, H/4, W/4]
fg_logits = fg_decoder(features)         # [B, 3, H/4, W/4]

# Upsample to full resolution
alpha_logits_up = interpolate(alpha_logits, input_size)
fg_logits_up = interpolate(fg_logits, input_size)

# Coarse probabilities for refiner input
alpha_coarse = sigmoid(alpha_logits_up)
fg_coarse = sigmoid(fg_logits_up)

# Refiner predicts delta logits from [RGB, alpha_coarse, fg_coarse]
delta = refiner(rgb, cat(alpha_coarse, fg_coarse))  # [B, 4, H, W] × 10.0

# Residual addition in logit space + final activation
alpha_final = sigmoid(alpha_logits_up + delta[:, 0:1])
fg_final = sigmoid(fg_logits_up + delta[:, 1:4])
```

Key detail: the refiner's output is scaled by `10×` and its weights are whisper-initialized (`Normal(0, 1e-3)` + zero bias), so it starts as near-identity and learns corrections.

---

## 2. Alpha Hint Generators

### GVM (`gvm_core/`)

- **Method:** Stable Video Diffusion with LoRA, FlowMatch scheduler (1-step denoising)
- **Input:** RGB video/sequence only — fully automatic
- **Processing:** Resize to 1024 short-edge (max 1920 long-edge), pad to multiples of 32, batch inference
- **Output post-processing:** Threshold ≥240/255 → 1.0, ≤25/255 → 0.0, resize to input resolution
- **VRAM:** ~80GB
- **Key class:** `GVMProcessor` in `gvm_core/wrapper.py`

### VideoMaMa (`VideoMaMaInferenceModule/`)

- **Method:** Masked Stable Video Diffusion with CLIP conditioning
- **Input:** RGB frames + binary `VideoMamaMaskHint` (thresholded at value 10)
- **Processing:** Resize to 1024×576 (SVD standard), chunk-based inference (default 24 frames)
- **VRAM:** ~24GB+ (FP16)
- **Key function:** `run_inference()` in `VideoMaMaInferenceModule/inference.py` — generator yielding frame chunks

---

## 3. Clip Manager Data Model

**File:** `clip_manager.py`

### Key Classes

- **`ClipAsset`** — Wraps a media source (directory of images or video file). Properties: `path`, `type` (`'sequence'`/`'video'`), `frame_count`
- **`ClipEntry`** — Represents a complete shot. Properties: `name`, `root_path`, `input_asset`, `alpha_asset`. Methods: `find_assets()` (discovers Input/AlphaHint with fallback heuristics), `validate_pair()` (ensures frame counts match)

### User-Configurable Settings (prompted at inference time)

| Setting | Default | Range | Notes |
|---|---|---|---|
| Gamma Space | sRGB | Linear or sRGB | Tells engine how to handle input gamma |
| Despill Strength | 10 (max) | 0-10 | Mapped to 0.0-1.0 internally |
| Auto-Despeckle | ON | ON/OFF | Connected-components cleanup |
| Despeckle Size | 400px | 0+ | Min pixel area to keep |
| Refiner Strength | 1.0 | float | Experimental, scales refiner output |

---

## 4. Debugging Checklist

When a user reports a visual artifact, check in this order:

1. **Dark fringes / crushed shadows?** → sRGB↔Linear conversion order error in `inference_engine.py` or downstream comp
2. **Blocky artifacts?** → Refiner not active (`use_refiner=False`) or refiner scale too low
3. **Green edges?** → Despill strength too low or applied in wrong color space
4. **Tracking markers in output?** → Auto-despeckle disabled or `despeckle_size` too small
5. **Mismatched frame counts?** → Input and AlphaHint sequences have different lengths
6. **OOM error?** → VRAM < 22.7GB, or running on GPU driving displays
7. **Black output?** → Checkpoint not found or wrong path; check `CorridorKeyModule/checkpoints/`
8. **Washed out / too bright?** → Double gamma correction (linear treated as sRGB or vice versa)

---

## 5. Directives for AI Assistants

1. **Color math is sacred.** Every compositing operation must respect the sRGB piecewise transfer functions and the straight/premultiplied distinction. When in doubt, trace the data through `color_utils.py`.
2. **Performance matters.** This processes 4K video frame-by-frame. Every `.numpy()` transfer, `cv2.resize`, or unnecessary copy matters in the hot loop.
3. **The model is fixed.** Training code is not in this repository. Do not modify `model_transformer.py` architecture unless explicitly building a new training pipeline.
4. **Test with `test_vram.py`** after any engine changes. Verify VRAM hasn't regressed.
5. **Respect licenses.** CorridorKey allows commercial *use* but not resale. GVM and VideoMaMa are strictly non-commercial.
6. **Preserve output compatibility.** Don't break the EXR output format that downstream Nuke/Fusion/Resolve workflows depend on.

---

## 6. Code Artifacts You May Encounter

- **Commented-out logit clamping** in `model_transformer.py` — An earlier "Humility Clamp" ([-3, 3]) was removed to preserve backbone detail. The commented code can be ignored.
- **`_orig_mod.` prefix stripping** in checkpoint loading — Handles models saved after `torch.compile()`. This is intentional and must stay.
- **Training code** is not in this repository. It may be released separately if community demand is sufficient.
