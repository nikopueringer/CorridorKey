# Color Pipeline & Compositing Math

How CorridorKey handles color, and how to correctly use its output in your compositing pipeline.

> **For artists:** If you just want to know how to use the output files, skip to the [output specifications](#reference-output-pass-specifications) or the [Artist Quickstart](QUICKSTART_ARTISTS.md#step-7-use-your-output).

---

## What Makes CorridorKey Different

Traditional chroma keyers produce a binary mask and then try to remove green spill as a post-process. The foreground "color" is whatever was in the original plate with green subtracted.

CorridorKey predicts, for every pixel:
- **The true, physically unmixed foreground color** — what the subject would look like if the green screen never existed
- **The linear alpha transmission ratio** — how much light passes through vs. is blocked

This is a mathematically correct decomposition of captured light. Semi-transparent pixels (hair, motion blur, out-of-focus edges) have their true color reconstructed, not estimated.

---

## sRGB vs Linear

Human vision is nonlinear — sRGB encodes images with a curve that allocates more bits to dark values. Linear values are proportional to physical photon counts.

**CorridorKey uses the official piecewise sRGB transfer function**, not the common gamma 2.2 approximation:

```
sRGB → Linear:
  if sRGB <= 0.04045: Linear = sRGB / 12.92
  else:               Linear = ((sRGB + 0.055) / 1.055) ^ 2.4

Linear → sRGB:
  if Linear <= 0.0031308: sRGB = Linear * 12.92
  else:                    sRGB = 1.055 * Linear^(1/2.4) - 0.055
```

The pure power curve `x^2.2` produces visible errors near black. The true sRGB function has a linear segment near zero that prevents these artifacts.

**Implementation:** `CorridorKeyModule/core/color_utils.py` — `srgb_to_linear()` and `linear_to_srgb()`

---

## Straight vs Premultiplied Alpha

| Mode | Storage | Compositing Formula |
|---|---|---|
| **Straight** | Color stored independently from alpha | `FG * alpha + BG * (1-alpha)` |
| **Premultiplied** | Color already multiplied by alpha | `FG_premul + BG * (1-alpha)` |

---

## The CorridorKey Color Pipeline

Exact sequence of operations in `inference_engine.py`:

```
Step 1: MODEL OUTPUT
  fg_srgb = sigmoid(fg_logits)        # sRGB, straight, [0,1]
  alpha_linear = sigmoid(alpha_logits) # Linear, [0,1]

Step 2: AUTO-DESPECKLE (optional)
  Connected-components: threshold→find→filter→dilate→blur→multiply

Step 3: DESPILL
  Luminance-preserving green removal in sRGB space

Step 4: sRGB → LINEAR
  Piecewise sRGB transfer function (color_utils.py)

Step 5: PREMULTIPLY
  fg_premul = fg_linear * alpha

Step 6: PACK RGBA
  processed = concat(fg_premul, alpha)  # All channels linear float

Step 7: COMPOSITE PREVIEW
  bg_linear = srgb_to_linear(checkerboard)
  comp = fg_linear * alpha + bg_linear * (1 - alpha)
  comp_srgb = linear_to_srgb(comp)
```

**Key ordering:** Despill happens in **sRGB space** (Step 3), premultiplication happens in **Linear space** (Step 5). This is intentional.

---

## Common Mistakes

### 1. Double Gamma Correction
**Symptom:** Washed out or too bright.
**Cause:** Applying `linear_to_srgb()` to already-sRGB data.
**Fix:** Track color space through every operation.

### 2. Using Gamma 2.2 Instead of sRGB Transfer
**Symptom:** Subtle banding or incorrect dark values.
**Fix:** Always use `cu.srgb_to_linear()` and `cu.linear_to_srgb()`.

### 3. Compositing in sRGB Space
**Symptom:** Dark halos around semi-transparent edges.
**Fix:** Convert both FG and BG to linear before compositing, then convert result back.

### 4. Premultiplying in sRGB Space
**Symptom:** Bright edges or dark fringes.
**Fix:** Convert to linear first, then premultiply. The `Processed/` pass does this correctly.

### 5. Forgetting FG Is sRGB
**Symptom:** Dark, crushed shadows in the composite.
**Fix:** The `FG/` output is sRGB. Convert to linear before combining with the linear alpha.

---

## Reference: Output Pass Specifications

### FG Pass (`/Output/FG/*.exr`)

| Property | Value |
|---|---|
| Format | OpenEXR, Half-float (16-bit), PXR24 compression |
| Channels | RGB (3 channels) |
| Color Space | sRGB gamut, sRGB transfer function |
| Alpha Mode | Straight (un-premultiplied) |
| Value Range | [0.0, 1.0] |
| Usage | Must convert to Linear before combining with Matte |

### Matte Pass (`/Output/Matte/*.exr`)

| Property | Value |
|---|---|
| Format | OpenEXR, Half-float (16-bit), PXR24 compression |
| Channels | Single channel (grayscale) |
| Color Space | Linear |
| Value Range | [0.0, 1.0] |
| Usage | Direct use as alpha channel |

### Processed Pass (`/Output/Processed/*.exr`)

| Property | Value |
|---|---|
| Format | OpenEXR, Half-float (16-bit), PXR24 compression |
| Channels | RGBA (4 channels) |
| Color Space | Linear, premultiplied |
| Post-processing | Despilled + despecked |
| Value Range | [0.0, 1.0] |
| Usage | Ready for immediate use in VFX compositors |

### Comp Pass (`/Output/Comp/*.png`)

| Property | Value |
|---|---|
| Format | PNG, 8-bit |
| Channels | RGB (3 channels, no alpha) |
| Color Space | sRGB |
| Background | Dark/light gray checkerboard |
| Usage | Preview only — not for production compositing |
