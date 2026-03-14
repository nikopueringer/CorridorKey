# MLX Optimization Log

Tracks all optimization experiments on the `feature/mlx-optimization` branch. Each entry follows the protocol from `HANDOFF_TO_CORRIDORKEY.md`.

---

## Baseline & Results

| Configuration | 37-frame clip @ 1920x1080 | ms/frame |
|---|---|---|
| PyTorch (MPS) | 3:34 | — |
| MLX (pre-optimization, 512px tiles) | 2:04 | ~3400 |
| MLX (pipeline opts, 768px tiles) | 1:20 | 2948 |
| MLX (+ refiner fp16, compile, no buffer limits) | 1:15 | 2762 |
| MLX (+ remove gc.collect/mx.clear_cache) | 0:55 | 2208 |
| MLX (+ iterative dilation, vectorized CCL) | 0:54 | 2177 |
| **MLX (all opts, all outputs)** | **0:54** | **2177** |
| **MLX (all opts, matte+fg, fast-exr)** | **0:51** | **1417** |
| **Speedup vs pre-opt (all outputs)** | | **1.56x** |
| **Speedup vs pre-opt (matte+fg fast-exr)** | | **2.40x** |
| **Speedup vs PyTorch MPS** | | **~4.2x** |

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

### 7. CUDA Cache Clearing (#18) — `0ec9749`

**What**: `torch.cuda.empty_cache()` after model forward pass.
**Impact**: Reduces peak VRAM on constrained CUDA GPUs. No effect on MLX.

### 8. LUT-Accelerated sRGB Conversions — `1409938`

**What**: 65536-entry float32 LUT replaces `np.power` in srgb_to_linear/linear_to_srgb.
**Measured**: sRGB ops 2x faster (27ms → 13ms each). Postprocess 159ms → 117ms.
**Fidelity**: Max error 0.0002 (well within 5e-3 Tier 1 threshold).

---

## From CorridorKey-Engine Fork (99oblivius)

| # | What | Outcome |
|---|------|---------|
| #14 | FlashAttention for Hiera | Already in timm 1.0.25 |
| #15 | GPU postprocessing | Deferred — CUDA-only, significant refactor |
| #16 | Deferred DMA + triple buffering | Deferred — CUDA-only |
| #17 | Multi-worker write pool | Tested, no improvement (CPU contention) |
| #18 | CUDA cache clearing | Implemented |

---

## Investigated and Closed (no action needed)

| # | Title | Finding |
|---|---|---|
| 6 | sRGB/linear to MLX GPU | Diminishing returns; postprocess already 2.9ms with skip |
| 8 | PyAV VideoToolbox hw decode | Reader already faster than inference (4.6ms vs 2139ms) |
| 11 | mx.async_eval when GPU saturated | Not applicable; threading handles I/O overlap |
| 12 | PyAV min version for hwaccel | PyAV 16.1.0 supports it; not needed per #8 |
| 13 | MLX thread safety with Metal | NOT SAFE; concurrent mx calls crash Metal encoder |

---

## Dead Ends (from prior research + this round)

See `HANDOFF_TO_CORRIDORKEY.md` Section 5 for full list. Key items:
- Temporal blending/caching: edge artifacts at all blend ratios
- Int8 quantization: 11% slower on Apple Silicon
- Backbone resolution decoupling: edge degradation at even 12% downscale
- GPU stream parallelism: single GPU on Apple Silicon
- MLX concurrent threads: Metal command encoder crashes on concurrent calls
- GPU-side colorspace: requires cross-repo API change for marginal gain
