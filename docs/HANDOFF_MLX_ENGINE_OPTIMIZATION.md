# Handoff: corridorkey-mlx Engine — Further Optimization

This doc covers what's been done, what's left, and how to continue optimizing the MLX inference engine at `../corridorkey-mlx` (github.com/cmoyates/corridorkey-mlx).

---

## 1. Current performance

| Configuration | 37-frame clip @ 1920x1080 |
|---|---|
| **Raw engine.process_frame()** | **~1345ms/frame** |
| **Pipeline (all outputs)** | **0:54 (2177ms/frame)** |
| **Pipeline (matte+fg, fast-exr)** | **0:51 (1417ms/frame)** |

### Per-tile breakdown (~6 tiles/frame at 768px)

| Component | ms/tile | % |
|---|---|---|
| Backbone (24 Hiera blocks) | ~37 | 43% |
| Refiner (4 dilated ResBlocks, fp16) | ~20 | 23% |
| Decoders (alpha + fg, bf16) | ~5 | 6% |
| mx.eval + np.array conversion | ~10 | 12% |
| Blend accumulation (numpy) | ~10 | 12% |
| Preprocessing (ImageNet norm) | ~3 | 4% |

---

## 2. What's been optimized (don't revert)

| Change | File | Impact |
|---|---|---|
| Remove gc.collect + mx.clear_cache from tiling loop | `inference/tiling.py` | -530ms/frame |
| Remove gc.collect + mx.clear_cache from engine.process_frame | `engine.py` | (included above) |
| Refiner dtype = float16 | `engine.py` → `load_model(refiner_dtype=mx.float16)` | ~130ms/frame (register pressure relief) |
| Compile enabled for tiled mode | `engine.py` → `compile=compile` (was forced False) | ~130ms/frame (graph fusion) |
| Conditional buffer limits | `engine.py` → only set MLX_MAX_OPS/MB when compile=False | Enables compile benefit |
| stage_gc = False | `engine.py` → `load_model(stage_gc=False)` | ~20ms/frame |
| Alpha-hint tile skipping | `inference/tiling.py` | Variable (skips pure BG/FG tiles) |

---

## 3. Proven dead ends (don't re-attempt)

| Approach | Result | Why |
|---|---|---|
| backbone_bf16_stages123 | 20% SLOWER | Dtype conversion at stage boundaries costs more than bandwidth savings |
| refiner_skip_confidence + refiner_frozen_gn + refiner_tile_size=384 | 55% SLOWER | GN stats collection pass + sub-tile mx.eval loop overhead exceeds skip savings at 768px |
| 1024px tiles | 20% SLOWER | Attention scales quadratically, im2col 9x inflation at dilation=8 |
| Int8 quantization | 11% SLOWER | Dequant overhead on unified memory (from prior research) |
| Multi-threaded mx.eval | CRASH | Metal command encoder is not thread-safe |
| Multi-process MLX engines | 6x SLOWER | Single GPU contends across processes |
| FP8 quantization | BLOCKED | MLX has no float8 dtype |
| Numba fused postprocess | 80% SLOWER | Per-pixel loop can't beat vectorized numpy+LUT |
| Skip explicit mx.eval before np.array | No change | np.array implicitly forces evaluation |

---

## 4. What to try next (prioritized)

### Priority 1: mx.compile + buffer limit experiment matrix

**What**: The buffer limit removal + compile interaction hasn't been fully explored. Currently `MLX_MAX_OPS_PER_BUFFER` is unset when compile=True (defaults to system). Test explicit values: 4, 8, 16, 32, unlimited.

**Where**: `engine.py:95-100`

**Hypothesis**: There's an optimal buffer size that balances graph size (for compile fusion) vs memory pressure. The current "remove limits entirely" may not be optimal.

**Risk**: Low. Pure A/B benchmark.

### Priority 2: Custom Metal kernels for refiner bottlenecks

**What**: The refiner's dilated convolutions use im2col which inflates memory ~9x at dilation=8. A custom Metal kernel using `mx.fast.metal_kernel()` could implement dilated conv via direct strided indexing instead of im2col.

**Where**: `model/refiner.py` — `RefinerBlock.__call__`

**Why**: The refiner is 23% of per-tile time. The im2col inflation is documented as the reason per-tile mx.eval is needed (to free the temporary buffers). A custom kernel that avoids im2col would:
1. Reduce per-tile memory pressure
2. Allow larger graphs before materialization
3. Potentially run faster via optimized thread group sizing

**Reference**: `mx.fast.metal_kernel` docs, Apple Metal Shading Language spec

**Risk**: High effort. Need MSL expertise. Validate against im2col output.

### Priority 3: Per-component compilation tuning

**What**: Currently the model uses full-forward `mx.compile(__call__)`. An alternative is compiling backbone, decoders, and refiner separately. The model already supports this via `compile_backbone`, `compile_decoders`, `compile_refiner` flags.

**Where**: `model/corridorkey.py:637-659` (compilation setup), `inference/pipeline.py:103-105`

**Why**: Per-component compilation allows `forward_eager()` path which uses `mx.async_eval` between stages and multi-stream dispatch for decoders. Full-forward compilation prevents this.

**Test**: Compare `compile_forward=True` (current) vs `compile_forward=False, compile_backbone=True, compile_decoders=True, compile_refiner=True` with `forward_eager()` called from tiling loop.

**Risk**: Low. Both paths exist. Benchmark required.

### Priority 4: Reduce numpy accumulation in tiling

**What**: Tile blending currently uses numpy (CPU) for weighted accumulation: `alpha_accum[y:ye, x:xe] += tile * weights`. This is ~60ms/frame.

**Alternatives**:
- Keep accumulation in MLX arrays (avoid np.array conversion per tile)
- Use mx.scatter_add or direct indexing for GPU-side accumulation
- Skip blend weights for non-overlapping owned regions (only blend overlap strips)

**Where**: `inference/tiling.py:180-184`

**Risk**: Medium. Changing to GPU-side accumulation requires restructuring the tile loop.

### Priority 5: Preprocess on GPU

**What**: ImageNet normalization currently happens in numpy (`preprocess()` in `io/image.py:48-68`): numpy concat + divide + subtract. Could do this entirely in MLX.

**Where**: `io/image.py:preprocess()` vs `io/preprocess_mlx.py:preprocess_mlx()`

**Note**: `preprocess_mlx` already exists for the full-frame path but the tiled path uses the numpy version. Switch to `preprocess_mlx` for tiled mode too.

**Risk**: Low. ~3ms savings per frame.

### Priority 6: Metal System Trace profiling

**What**: Run the pipeline under Metal GPU Profiler to identify shader-level bottlenecks.

```bash
MTL_CAPTURE_ENABLED=1 MLX_METAL_DEBUG=1 python corridorkey_cli.py run-inference ...
```

Open the `.gputrace` capture in Xcode Metal Debugger. Look for:
- Thread occupancy (are SIMD groups underutilized?)
- Memory bandwidth utilization (are we near the roofline?)
- Specific shader bottlenecks (which kernels take the most time?)
- Register spilling (is the compiler spilling to main memory?)

**Risk**: None. Diagnostic only.

---

## 5. Architecture-level changes (need training pipeline)

These require fine-tuning or distillation — not possible with fixed weights.

| Approach | Potential | Requirement |
|---|---|---|
| Knowledge distillation: 24-block → 4-block student | 80% backbone reduction | Training harness + green screen dataset |
| Depthwise separable refiner | 60% refiner reduction | Replace dilated ResBlocks, fine-tune |
| Wavelet-domain selective attention | Up to 4x backbone | DWT/IDWT modules + fine-tuning |
| Optical flow keyframe warping | 6x throughput | Flow estimator + differentiable warping |
| Hiera-Small/Tiny backbone swap | 40-60% backbone | Fine-tune smaller backbone for matting |

---

## 6. Key files

| Path | Purpose |
|---|---|
| `src/corridorkey_mlx/engine.py` | Public API, buffer limits, compile flags, model loading |
| `src/corridorkey_mlx/inference/tiling.py` | Tiled inference loop, blend weights, tile skipping |
| `src/corridorkey_mlx/inference/pipeline.py` | load_model, compile_model |
| `src/corridorkey_mlx/model/corridorkey.py` | GreenFormer: __call__, forward_eager, _refiner_tiled |
| `src/corridorkey_mlx/model/refiner.py` | CNNRefinerModule, FrozenGroupNorm, dilated ResBlocks |
| `src/corridorkey_mlx/model/decoder.py` | DecoderHead (FPN-style) |
| `src/corridorkey_mlx/model/backbone.py` | HieraBackbone wrapper |
| `src/corridorkey_mlx/io/image.py` | Preprocessing (numpy path) |
| `src/corridorkey_mlx/io/preprocess_mlx.py` | Preprocessing (MLX GPU path) |

---

## 7. Critical lesson learned

**gc.collect() and mx.clear_cache() in hot loops are the #1 hidden performance killer on Apple Silicon.** Removing them saved 530ms/frame — more than any other single optimization. On unified memory with sufficient headroom, Python's reference counting handles cleanup. Forced GC is only needed on memory-constrained devices.

See: `docs/solutions/performance/gc-collect-mlx-unified-memory.md`
