# Process Overview

Every step CorridorKey performs on a single frame, from raw input to final output, explained for humans and AI agents alike. For canonical implementation details, see the inline comments in the source files referenced below.

**Source files:** Steps 1–8 live in `CorridorKeyModule/inference_engine.py` (`process_frame()`). Steps 5A–5E live in `CorridorKeyModule/core/model_transformer.py` (`GreenFormer.forward()`). Color math lives in `CorridorKeyModule/core/color_utils.py`.

---

## The Problem

You filmed a subject in front of a green screen. Every pixel in your footage is a physical mixture of the subject's true color and the green background, blended by the subject's transparency at that point. CorridorKey's job is to **unmix** that — predicting what the subject really looks like and exactly how transparent each pixel is.

---

## Phase 1: Pre-Processing

### Step 1 — Input Normalization
> *Convert pixel counts to pixel fractions*

Raw pixels are integers (0–255). Neural networks work in decimals (0.0–1.0). This divides every value by 255, turning "pixel brightness as a count" into "pixel brightness as a fraction."

`inference_engine.py` · Step 1

### Step 2 — Resize to Fixed Canvas (2048x2048)
> *Fit every image into the same picture frame*

The neural network has a fixed "field of view" — every image gets scaled to 2048x2048, like fitting any photo into the same frame. This is required because the transformer's internal position encoding is tied to a fixed grid size. If the input is linear-light (e.g., EXR), resizing happens in linear space first to preserve physical brightness, then converts to sRGB for the model.

`inference_engine.py` · Step 2

### Step 3 — ImageNet Normalization
> *Adjust the "exposure" to match what the model expects*

Shifts and scales pixel values to match the statistics the pretrained vision backbone saw during training. Like adjusting the exposure and white balance so the model recognizes features the same way it learned them. Without this, the model would see every image as too bright or oddly tinted.

`inference_engine.py` · Step 3

### Step 4 — Pack Into GPU Tensor
> *Stack image + mask into a single input for the network*

The 3-channel RGB image and 1-channel mask are stacked side-by-side into a single 4-channel input, then converted to a PyTorch tensor on the GPU. The mask acts as a "hint" — telling the network roughly where the subject is, so it can focus on refining edges rather than searching the whole frame.

`inference_engine.py` · Step 4

---

## Phase 2: Neural Network Inference

### Step 5A — Transformer Encoding
> *Break the image into patches, build understanding from local detail to global context*

The Vision Transformer (Hiera) breaks the image into small patches and processes them through hierarchical attention — patches attend locally first, then regions merge and widen at each stage, building understanding from fine detail up to scene-level context. It outputs 4 feature maps at progressively coarser scales (1/4, 1/8, 1/16, 1/32 of input size), capturing everything from fine edge detail to big-picture scene understanding.

`model_transformer.py` · Stage A

### Step 5B — Dual Decoding
> *Two independent heads read the same features, predict different things*

Two decoder heads process the encoder's multi-scale features:
- **Alpha decoder** — "How transparent is this pixel?" (1 channel)
- **Foreground decoder** — "What color is this pixel really, without the green screen?" (3 channels)

Each decoder takes features from all 4 scales, projects them to a common dimension, upscales them to the same size, and merges them — like combining a microscope view, a normal lens, a wide-angle, and a satellite view into one coherent prediction. Each decoder outputs at 1/4 resolution in "logit" space (raw scores, not yet constrained to 0–1). Bilinear interpolation then scales these logits back to the full 2048×2048 grid — upsampling before sigmoid rather than after keeps the math smoother and avoids clipping near 0 and 1.

`model_transformer.py` · Stages B–C

### Step 5C — Sigmoid Activation (Coarse Prediction)
> *Squash raw scores into probabilities*

Sigmoid converts the raw logit scores into the 0–1 range: large negative values become near-0 (transparent/black), large positive values become near-1 (opaque/bright). This produces the transformer's "coarse" prediction — accurate in broad strokes but with visible grid-line artifacts at patch boundaries.

`model_transformer.py` · Stage D

### Step 5D — CNN Refinement
> *Fine brush over broad strokes — clean up patch boundary artifacts*

A small CNN (the "Refiner") looks at the original image alongside the transformer's coarse prediction and outputs pixel-level corrections. It uses dilated convolutions with increasing dilation rates (1→2→4→8) to see across a ~63px receptive field — wide enough to look across patch boundaries and smooth the blocky artifacts. The corrections are added in **logit space** (before sigmoid), which means even very confident predictions can be adjusted without hitting mathematical ceilings. Initialized to near-zero so it starts as a no-op and learns what to fix during training.

`model_transformer.py` · Stage E, `CNNRefinerModule`

### Step 5E — Final Activation
> *Merge corrections and produce the final prediction*

The refiner's delta corrections are added to the original logits, then sigmoid is applied to produce the final 0–1 predictions. Alpha becomes the transparency matte; foreground becomes the "clean" subject color.

`model_transformer.py` · Stage F

---

## Phase 3: Post-Processing

### Step 6 — Resize Back to Original Resolution
> *Carefully enlarge back to the original image size*

Scales the 2048x2048 predictions back to the original image dimensions using Lanczos4 — a high-quality resampling algorithm that preserves sharp edges. Better than bilinear for final output because it maintains the crisp matte edges the network predicted.

`inference_engine.py` · Step 6

### Step 7A — Auto-Despeckle
> *Find and kill stray blobs*

The network sometimes keeps tiny floating islands of alpha — stray tracking markers, lens reflections, or noise specks. This finds every disconnected blob in the matte, throws away any blob smaller than a threshold (default 400px), then dilates and blurs the surviving regions to create a soft "safe zone." The original alpha is multiplied by this safe zone, killing the specks while preserving all detail within the real subject boundary.

`inference_engine.py` · Step 7A, `color_utils.clean_matte()`

### Step 7B — Despill
> *Neutralize green light bouncing onto the subject*

Green screens bounce green light onto the subject — tinting skin, hair, and clothing edges. For each pixel, if green exceeds the average of red and blue, that excess is "spill." The excess is subtracted from green and redistributed half each to red and blue, preserving the pixel's total RGB sum while neutralizing the green cast. Applied in sRGB space (before linear conversion) because the comparison operates on perceptually-encoded values, matching how the green cast appears to the eye.

`inference_engine.py` · Step 7B, `color_utils.despill()`

### Step 7C — Premultiply for EXR
> *Bake transparency into color to prevent fringing*

"Premultiply" means multiplying each color channel by its alpha (R×α, G×α, B×α). A 50% transparent red pixel goes from (1,0,0) to (0.5,0,0). This is the industry standard for compositing because it prevents bright color fringing at semi-transparent edges — without it, a 50% transparent pixel next to a green screen would bleed bright green into the final composite. Color is first converted from sRGB to linear light, because EXR files must store physically-correct linear values.

`inference_engine.py` · Step 7C, `color_utils.premultiply()`

### Step 7D — Pack RGBA
> *Combine color and alpha into the final deliverable*

The 3 processed color channels and 1 alpha channel are combined into a single 4-channel image. Everything is now in linear float space, ready for EXR export.

`inference_engine.py` · Step 7D

---

## Phase 4: Output

### Step 8 — Composite Preview
> *Overlay on a checkerboard to verify transparency*

The keyed subject is layered over a checkerboard pattern so you can visually verify the quality — like holding a cutout up against a patterned background to check for holes, halos, or green fringing. The checkerboard is generated in sRGB, converted to linear for physically-correct blending, then the result is converted back to sRGB for PNG output.

`inference_engine.py` · Step 8

### Written Outputs

| Pass | Path | Format | Contents |
|---|---|---|---|
| **FG** | `/Output/FG/*.exr` | EXR half-float, sRGB | Straight foreground color (convert to linear before compositing) |
| **Matte** | `/Output/Matte/*.exr` | EXR half-float, linear | Alpha channel for direct use |
| **Processed** | `/Output/Processed/*.exr` | EXR half-float, linear premul | Ready-to-use RGBA — despilled, despeckled, premultiplied |
| **Comp** | `/Output/Comp/*.png` | PNG 8-bit, sRGB | Preview only — composite on checkerboard |
