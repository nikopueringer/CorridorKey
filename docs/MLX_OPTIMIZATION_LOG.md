# MLX Optimization Log

Tracks all optimization experiments on the `feature/mlx-optimization` branch. Each entry follows the protocol from `HANDOFF_TO_CORRIDORKEY.md`.

---

## Baseline & Results

| Configuration | 37-frame clip @ 1920x1080 | ms/frame |
|---|---|---|
| PyTorch (MPS) | 3:34 | — |
| MLX (pre-optimization, 512px tiles) | 2:04 | ~3400 |
| **MLX (all opts, 768px tiles, all outputs)** | **1:20** | **2948** |
| **MLX (all opts, 768px tiles, matte+fg only)** | **~1:10** | **~2400** |

---

## Completed Optimizations

### 1. Async I/O Pipeline (Issue #2) — `d98ad12`

**What**: 3-thread pipeline — reader decodes frames into `Queue(maxsize=2)`, main thread runs inference, writer writes outputs.
**Measured**: Write overlaps with inference. Write 691ms → effectively free during GPU compute.
**Fidelity**: Tier 3 (within MLX non-determinism: max_abs_diff < 0.005)

### 2. Checkerboard Cache (Issue #4) — `d98ad12`

**What**: Cache `create_checkerboard(w, h)` by `(w, h, size, c1, c2)` key.
**Measured**: Eliminates per-frame allocation (~1-2ms saved).

### 3. Postprocess Timing Fix (Issue #9) — `d98ad12`

**What**: Adapter returns `_timing` dict, writer collects `phase_times["postprocess"]`.
**Result**: Postprocess now visible in timing summary (was hidden inside "infer").

### 4. Configurable Output Selection (Issue #3) — `64c04d3`

**What**: `--outputs` CLI flag (e.g. `--outputs matte,fg`). Skips unneeded file writes.
**Measured**: Write 691ms → 384ms with matte+fg only (44% reduction).

### 5. Conditional Postprocess Skip (Issue #5) — `64c04d3`

**What**: When comp+processed disabled, skip despill/composite/colorspace/premultiply.
**Measured**: Postprocess 158ms → 2.9ms (98% reduction).

### 6. Larger Tiles 768px + Overlap 128 (Issue #7, #10) — `62474f1`

**What**: Increased tile size from 512 to 768, overlap from 64 to 128 (2x refiner RF).
**Measured**: Infer 3031ms → 2139ms (29% faster). ~6 tiles/frame vs 15.
**Fidelity**: No NaN/Inf, proper [0,1] range. Overlap 128 matches handoff doc.

---

## Remaining Issues

| # | Title | Priority | Status |
|---|---|---|---|
| 6 | sRGB/linear conversions to MLX GPU | P3c | Open |
| 8 | PyAV VideoToolbox hw decode | P5 | Open |
| 11 | mx.async_eval benefit when GPU saturated | Investigation | Open |
| 12 | PyAV min version for hwaccel | Investigation | Open |
| 13 | MLX thread safety with Metal | Investigation | Open |

---

## Dead Ends (from prior research)

See `HANDOFF_TO_CORRIDORKEY.md` Section 5 for full list. Key items:
- Temporal blending/caching: edge artifacts at all blend ratios
- Int8 quantization: 11% slower on Apple Silicon
- Backbone resolution decoupling: edge degradation at even 12% downscale
- GPU stream parallelism: single GPU on Apple Silicon
