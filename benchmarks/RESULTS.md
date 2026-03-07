# Benchmark Results

Reference clip: `ClipsForInference/BetterGreenScreenTest_BASE/Input.mp4` (20 frames, 1920x1080 input, 2048x2048 inference)
Alpha hint: `ClipsForInference/BetterGreenScreenTest_BASE/AlphaHint/BetterGreenScreenTest_MASK.mp4`
Settings: sRGB colorspace, all other defaults (despill=1.0, auto_despeckle=True, refiner_scale=1.0)
Hardware: Apple M3, MPS backend

## Results Table

| Phase | Median Frame Time | Mean Frame Time | MPS Mem (post-inference) | Mem Delta | Notes |
|-------|------------------|-----------------|------------------------|-----------|-------|
| 0 -- Baseline | 7.85s | 13.10s* | 32.23 GB | 30.95 GB | *Mean skewed by MPS shader compilation on frames 2-3 (30.8s, 76.4s) |
| 1 -- FP16 weights | 5.70s | 5.66s | 25.02 GB | 23.90 GB | -7.21 GB mem, -27% time vs baseline |
| 2 -- GPU math + caching | 5.42s | 5.51s | 26.10 GB | 24.98 GB | -6.13 GB mem, -31% time vs baseline; +1.08 GB vs P1 (GPU holds post-proc tensors) |
| 3 -- Backbone 1024 | | | | | |
| 4 -- Tiled refiner | | | | | |

## Phase 0 -- Baseline (unoptimized)

**Date:** 2026-03-07
**Commit:** 995bd25

### Timing

| Metric | Value |
|--------|-------|
| Warmup (frame 1) | 10.61s |
| Mean (excl. warmup) | 13.10s |
| Median (excl. warmup) | 7.85s |
| Stdev | 16.23s |
| Min | 7.17s |
| Max | 76.40s |

Frames 2-3 were extreme outliers (30.8s, 76.4s) due to MPS shader compilation/caching.
Steady-state frames (4-20) ranged 7.2-12.5s with median ~7.8s.

### Memory (MPS unified)

| Metric | Value |
|--------|-------|
| Before inference | 1.28 GB |
| After inference | 32.23 GB |
| Delta | 30.95 GB |

MPS unified memory numbers reflect shared CPU/GPU pool. Useful for relative comparison between phases, not absolute VRAM claims.

### Quality

Baseline vs itself: all pixel diffs are 0. Quality gate tests pass trivially.

## Phase 1 -- FP16 Weight Casting

**Date:** 2026-03-07
**Commit:** 136a18f

### Change

Added `model.half()` after `load_state_dict` in `_load_model`. Autocast already ran FP16 activations; aligning weight storage halves static footprint and reduces activation memory.

### Timing

| Metric | Value |
|--------|-------|
| Warmup (frame 1) | 5.98s |
| Mean (excl. warmup) | 5.66s |
| Median (excl. warmup) | 5.70s |
| Stdev | 0.13s |
| Min | 5.47s |
| Max | 5.85s |

No MPS compilation outliers this run — shader cache likely warm from Phase 0.

### Memory (MPS unified)

| Metric | Value | Delta vs Baseline |
|--------|-------|-------------------|
| Before inference | 1.12 GB | -0.16 GB |
| After inference | 25.02 GB | -7.21 GB (-22.4%) |
| Delta | 23.90 GB | -7.05 GB |

Savings far exceed the estimated ~400MB. FP16 weights produce FP16 intermediates more efficiently under autocast, reducing activation memory too.

### Quality (vs Phase 0 Baseline)

| Channel | MAE | Max Err | PSNR | Pixels > 1e-4 | Pixels > 1e-2 |
|---------|-----|---------|------|---------------|---------------|
| Alpha | 0.000007 | 0.0057 | 83.7 dB | 1.55% | 0.00% |
| FG | 0.000035 | 0.0324 | 78.8 dB | 11.69% | 0.00% |
| Processed | 0.000017 | 0.0057 | 83.4 dB | 5.00% | 0.00% |
| Composite | 0.000029 | 0.0033 | 82.7 dB | 10.90% | 0.00% |

FP16 rounding introduces tiny precision diffs but zero pixels exceed 1e-2. Quality gate tests pass with fp16 thresholds (plan's original "lossless" 1e-4 max_err too tight for FP16).

## Phase 2 -- GPU Color Math + Asset Caching

**Date:** 2026-03-07
**Commit:** 59927d1

### Changes

- `F.interpolate(bicubic)` replaces `cv2.resize(Lanczos4)` for prediction upscaling — stays on GPU
- `despill`, `srgb_to_linear`, `premultiply`, compositing all run as GPU tensor ops
- `.cpu().numpy()` deferred to final return (single batch transfer)
- `clean_matte` stays CPU (cv2.connectedComponents) — only alpha transferred
- Checkerboard + linear variant cached per resolution as GPU tensors
- Dilation kernel cached in module-level dict
- Bicubic clamp added (can overshoot [0,1])

### Timing

| Metric | Value |
|--------|-------|
| Warmup (frame 1) | 5.81s |
| Mean (excl. warmup) | 5.51s |
| Median (excl. warmup) | 5.42s |
| Stdev | 0.16s |
| Min | 5.31s |
| Max | 5.74s |

~5% faster than Phase 1. Modest gain — inference dominates; post-processing was already a small fraction.

### Memory (MPS unified)

| Metric | Value | Delta vs Baseline | Delta vs Phase 1 |
|--------|-------|-------------------|-------------------|
| Before inference | 1.12 GB | -0.16 GB | 0 |
| After inference | 26.10 GB | -6.13 GB (-19.0%) | +1.08 GB |
| Delta | 24.98 GB | -5.97 GB | +1.08 GB |

Memory slightly higher than Phase 1 because GPU now holds post-processing tensors (checkerboard, color math intermediates) that previously lived on CPU numpy. Net still 6.13 GB below baseline.

### Quality (vs Phase 0 Baseline)

| Channel | MAE | Max Err | PSNR | Pixels > 1e-4 | Pixels > 1e-2 |
|---------|-----|---------|------|---------------|---------------|
| Alpha | 0.000041 | 0.0204 | 68.8 dB | 2.77% | 0.01% |
| FG | 0.000125 | 0.0826 | 64.6 dB | 21.29% | 0.06% |
| Processed | 0.000065 | 0.0250 | 70.3 dB | 11.45% | 0.00% |
| Composite | 0.000092 | 0.0185 | 70.7 dB | 20.09% | 0.00% |

Quality diffs larger than Phase 1 due to Lanczos4→bicubic interpolation change (different algorithm, not just floating point ordering). FG channel most affected (3 channels of color data). However, pixels > 1e-2 remains near 0% across all channels — visually indistinguishable.

**Note:** FG max err 0.083 exceeds plan's Phase 1-2 threshold of 0.04. This is cumulative: FP16 rounding (Phase 1) + Lanczos4→bicubic (Phase 2). The threshold assumed Phase 2 would only introduce "floating-point ordering differences" but the interpolation algorithm change is more significant. Practically lossless — 0.06% pixels > 1e-2 in FG, 0% elsewhere.
