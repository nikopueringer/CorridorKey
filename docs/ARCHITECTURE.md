# Architecture Deep Dive

Technical walkthrough of CorridorKey's neural network architecture and inference pipeline.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                                │
│  RGB Footage + (optional) Coarse Mask Hint                  │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌─────────────────────────┐
│   Alpha Generators   │          │  Manual Alpha Hint       │
│                      │          │  (Chroma key, AI roto,   │
│  ┌────────────────┐ │          │   hand-drawn mask)        │
│  │ GVM (Auto)     │ │          └────────────┬──────────────┘
│  │ ~80GB VRAM     │ │                       │
│  └────────────────┘ │                       │
│  ┌────────────────┐ │                       │
│  │ VideoMaMa      │ │                       │
│  │ ~24GB+ VRAM    │ │                       │
│  └────────────────┘ │                       │
└──────────┬──────────┘                       │
           │                                  │
           ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   CorridorKey Engine                          │
│                                                              │
│  Input: RGB [H,W,3] + AlphaHint [H,W,1]                    │
│  Processing: 2048x2048 native resolution                     │
│  Output: Straight FG (sRGB) + Linear Alpha + Premul RGBA    │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    VFX Pipeline Output                        │
│                                                              │
│  /FG/        Straight foreground color (EXR, sRGB gamut)    │
│  /Matte/     Linear alpha channel (EXR)                     │
│  /Processed/ Premultiplied RGBA (EXR, Linear)               │
│  /Comp/      Preview composite (PNG, sRGB)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## The GreenFormer Model

**File:** `CorridorKeyModule/core/model_transformer.py`

A hybrid Vision Transformer + CNN architecture. The transformer provides global context; the CNN refiner eliminates patch-boundary artifacts.

### Backbone: Hiera Base Plus

[Hiera](https://github.com/facebookresearch/hiera) (`hiera_base_plus_224`) pretrained on ImageNet via MAE + fine-tuning.

**Modifications:**
- Input patched from 3→4 channels (RGB + alpha hint), extra channel weights zero-initialized
- Image size 224→2048, positional embeddings bicubically interpolated
- Pretrained base weights NOT downloaded at init — the full checkpoint contains all weights

**Feature extraction:**
```
Stage 1: [B, 112, H/4,  W/4 ]   (finest)
Stage 2: [B, 224, H/8,  W/8 ]
Stage 3: [B, 448, H/16, W/16]
Stage 4: [B, 896, H/32, W/32]   (coarsest)
```

### Decoder Heads

Two independent `DecoderHead` instances (SegFormer-style MLP decoder):
1. **Alpha Decoder** → 1-channel logits
2. **Foreground Decoder** → 3-channel logits

Each head: project all 4 scales to 256 dims → upsample to H/4 × W/4 → concatenate → 1×1 conv fusion → BatchNorm → ReLU → Dropout(0.1) → classifier → bilinear upsample to full resolution.

### CNN Refiner

Eliminates macroblocking from the Hiera patch-based backbone.

```
Input: [B, 7, H, W]
  ├── 3ch: RGB from model input (ImageNet-normalized)
  └── 4ch: sigmoid(coarse logits) — alpha prob + FG probs

Stem:     Conv2d(7→64, 3×3) + GroupNorm(8) + ReLU
ResBlock1: Dilation=1  (local detail)
ResBlock2: Dilation=2  (medium context)
ResBlock3: Dilation=4  (wide context)
ResBlock4: Dilation=8  (global context)
Final:    Conv2d(64→4, 1×1) × 10.0

Output: [B, 4, H, W]  (1ch alpha delta + 3ch FG delta)
```

- **GroupNorm** instead of BatchNorm — safe for batch size 1
- **Dilated convolutions** — ~65px receptive field without downsampling
- **10× output scaling** — small stable predictions become meaningful corrections in logit space
- **Whisper initialization** — `Normal(0, 1e-3)` + zero bias, starts as near-identity

**Fusion:** Deltas are added in logit space (before sigmoid), enabling unbounded correction without gradient saturation.

---

## Inference Engine Pipeline

**File:** `CorridorKeyModule/inference_engine.py`

```
Step 1: INPUT NORMALIZATION
  uint8 → float32 / 255.0 (if needed)

Step 2: RESIZE TO 2048×2048 (bilinear interpolation)
  Linear input: resize in linear → convert to sRGB
  sRGB input:   resize directly in sRGB

Step 3: IMAGENET NORMALIZE
  (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

Step 4: CONCAT + INFERENCE
  [RGB, Mask] → [B, 4, 2048, 2048] → FP16 autocast → GreenFormer

Step 5: RESIZE BACK (Lanczos4)
  Lanczos4 to original resolution

Step 6: POST-PROCESS
  Auto-despeckle → Despill → sRGB→Linear → Premultiply → Pack RGBA
```

### Memory

| Resolution | Approximate VRAM |
|---|---|
| 2048×2048 (native) | ~22.7 GB |
| 1024×1024 | ~8 GB (estimated) |
| 512×512 | ~3 GB (estimated) |

Uses `torch.autocast(dtype=torch.float16)` during inference.

---

## Data Flow: Tensor Shapes

```
Input Image:  [H, W, 3]  NumPy float32
Input Mask:   [H, W, 1]  NumPy float32
                │
                ▼ cv2.resize (bilinear)
Resized:      [2048, 2048, 3+1]
                │
                ▼ ImageNet norm + concat + transpose
Tensor:       [1, 4, 2048, 2048]  PyTorch float32 → autocast float16
                │
                ▼ Hiera Encoder
Features:     [1, 112, 512, 512]  ... [1, 896, 64, 64]
                │
                ▼ Decoder Heads + Upsample
Logits:       [1, 1, 2048, 2048]  alpha
              [1, 3, 2048, 2048]  fg
                │
                ▼ Refiner + Sigmoid
Predictions:  [1, 1, 2048, 2048]  alpha (0-1)
              [1, 3, 2048, 2048]  fg sRGB (0-1)
                │
                ▼ .cpu().numpy() + cv2.resize(Lanczos4)
Output:       [H, W, 1]  alpha
              [H, W, 3]  fg sRGB
```
