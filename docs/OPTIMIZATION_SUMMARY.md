# CorridorKey MLX — Complete Optimization Summary

106 experiments across two repositories optimizing CorridorKey's MLX inference pipeline on Apple Silicon. This doc covers everything that was tried, what worked, what didn't, and why.

## Model architecture

CorridorKey is a video matting model that separates foreground from a green screen:

- **Backbone**: Hiera-Base-Plus (24 transformer blocks, 4 stages) — extracts multiscale features from a 4-channel input (RGB + coarse alpha hint)
- **Decoder**: Two SegFormer-style heads (alpha + foreground) — fuse multiscale features via linear projections + bilinear upsampling
- **Refiner**: CNN with 4 dilated residual blocks (dilation 1/2/4/8) + 9 GroupNorm layers — sharpens edges at full resolution using RGB + coarse predictions

Pretrained at 2048×2048. At inference: **tiled processing** (768px tiles, 128px overlap) handles arbitrary resolutions. A 1920×1080 frame = ~6 tiles.

## Results

| Configuration | 37 frames @ 1920×1080 | ms/frame |
|---|---|---|
| PyTorch (MPS) | 3:34 | ~5800 |
| MLX (pre-optimization, 512px tiles) | 2:04 | ~3400 |
| MLX (pipeline opts, 768px tiles) | 1:20 | 2948 |
| MLX (+ refiner fp16, compile, buffer tuning) | 1:15 | 2762 |
| MLX (+ remove gc.collect/mx.clear_cache) | 0:55 | 2208 |
| MLX (+ iterative dilation, vectorized CCL) | 0:54 | 2177 |
| **MLX (all opts, all outputs)** | **0:54** | **~1456** |
| **MLX (matte+fg only, fast-exr)** | **0:51** | **~1417** |
| **Speedup vs PyTorch MPS** | | **~4.0×** |

## Per-tile breakdown (768px)

| Component | ~ms/tile | % |
|---|---|---|
| Backbone S0 (2 blocks, windowed, dim=112) | ~12 | 5% |
| Backbone S1 (3 blocks, windowed, dim=224) | ~13 | 6% |
| **Backbone S2 (16 blocks, global, dim=448)** | **~60** | **27%** |
| Backbone S3 (3 blocks, global, dim=896) | ~10 | 5% |
| Decoders (alpha + fg, dual-stream) | ~13 | 6% |
| **Refiner (9 GN, 4 dilated ResBlocks)** | **~75** | **35%** |
| Per-tile overhead (eval, slice, pad) | ~35 | 16% |
| **Total** | **~218** | **100%** |

---

## What worked

### Model-level optimizations (corridorkey-mlx repo)

#### Operator fusion and dispatch reduction

| What | Why it helps |
|---|---|
| Folded BatchNorm (precompute scale+offset at load) | 2 ops vs 5 per BN layer |
| Conv1×1 → `mx.addmm` bypass | Avoids conv dispatch overhead for pointwise ops |
| Split-fuse decoder (process alpha/fg without intermediate concat) | Eliminates large temporary allocation |
| Einsum fused output projection | One op replaces reshape→transpose→matmul chain |
| Split QKV + pretranspose MLP weights | Contiguous memory access in hot loops |
| Precomputed unroll/reroll via `mx.take` | Single gather replaces 3-step reshape→transpose→reshape |

#### Precision management

| What | Why it helps |
|---|---|
| Decoder BF16 weights | Halves decoder memory bandwidth |
| BF16 coarse sigmoid path | Fewer dtype conversions |
| Deferred FP32 cast (only at output boundary) | Avoids redundant precision upcasts |
| Refiner FP16 weights + activations | Halves refiner bandwidth at full resolution |

**Critical**: Backbone stage 0 (dim=112) MUST stay FP32. BF16 activations in stage 0 cause fidelity regression.

#### Memory and scheduling

| What | Why it helps |
|---|---|
| Per-component `mx.compile` (backbone, decoders, refiner individually) | Eager async_eval between components enables better scheduling |
| `mx.async_eval` at stage boundaries | CPU builds next graph while GPU executes current |
| Dual-stream decoder dispatch (alpha on default, fg on secondary) | Overlaps two independent decoder forward passes |
| Per-tile `mx.eval` between tiles | Frees im2col buffers (9× inflation from dilated convs) before next tile |
| Cache limit 1536MB (`mx.set_cache_limit`) | Forces Metal buffer reuse, reduces peak memory |
| `MLX_MAX_MB_PER_BUFFER=2, MLX_MAX_OPS_PER_BUFFER=2` | Small buffers force frequent materialization, preventing graph buildup. **17% faster** for tiled workloads. |

#### Custom Metal kernels

| What | Why it helps |
|---|---|
| Metal GroupNorm (shared-mem stats + parallel normalize) | -67% vs nn.GroupNorm. Eliminates NHWC↔NCHW transposes. |
| Frozen GN stats mode (precomputed mean/var for tiled inference) | **Eliminates tiling artifacts completely** — 0.0 error vs full-image reference. Per-tile stats cause 68/255 max error at boundaries. |

### Pipeline-level optimizations (CorridorKey repo)

| What | Impact |
|---|---|
| 3-thread async I/O (reader → inference → writer) | Write 691ms → effectively free during GPU compute |
| Configurable output selection (`--outputs matte,fg`) | Write 691ms → 384ms (44% reduction) |
| Conditional postprocess skip (skip despill/composite when unused) | Postprocess 158ms → 2.9ms (98% reduction) |
| 768px tiles + 128px overlap (up from 512px/64px) | Infer 3031ms → 2139ms (29% faster). ~6 tiles vs 15. |
| LUT-accelerated sRGB conversions (65536-entry float32 LUT) | sRGB ops 2x faster (27ms → 13ms). Max error 0.0002. |
| `--fast-exr` uncompressed EXR writes | Write 10x faster for matte+fg workflows |
| Iterative small-kernel dilation for clean_matte | 4.8x faster morphological ops |
| Vectorized connected-component label filter | 5x faster despeckle |
| Checkerboard cache by (w, h, size, c1, c2) key | ~1-2ms/frame saved |
| Adaptive despeckle bypass + frame-level inference skip | Skips processing for trivial frames |
| Remove gc.collect/mx.clear_cache from hot path | **~530ms/frame saved** — these forced Metal buffer deallocation and Python garbage collection mid-inference |

---

## What didn't work (and why)

### Matting is edge-sensitive

Matting produces per-pixel alpha where edges (hair, fingers, silhouettes) matter most. Any optimization degrading edge fidelity fails, even if mean error is tiny.

| Technique | Result | Why |
|---|---|---|
| Backbone resolution decoupling (run backbone at lower res) | 91/255 max edge error at even 12% downscale | Backbone provides spatial features for edge localization. Downscaling destroys sub-pixel detail. |
| Feature caching S2-S3 across frames | 247/255 max error | S2 features (stride-16, 448ch) change significantly between real frames. |
| Output-space EMA blending | Fails at ALL blend values | Temporal lag on edges visible even at α=0.95 |
| Feature-space EMA | 70.5/255 max error at α=0.9 | Edge features in decoder outputs shift between frames. Blending smears them. |
| Backbone skip (every other frame) | Visible motion artifacts | Same as feature caching — too much changes between frames. |
| Skip + interpolation | Still artifacts | Interpolation can't recover missed edge detail. |

### Apple Silicon architecture constraints

Unified memory and single GPU change which optimizations are profitable.

| Technique | Result | Why |
|---|---|---|
| **Int8 quantization** | **11% SLOWER** | Dequantize-multiply overhead exceeds bandwidth savings on unified memory. |
| FP8 quantization | Not available | MLX doesn't support FP8 as of v0.31. |
| BF16 backbone stage 0 | Fidelity regression | dim=112 is precision-sensitive. |
| GPU stream parallelism | No GPU-GPU parallelism | Apple Silicon = one GPU. `mx.stream` is scheduling hints only. |
| Multi-process MLX | Crashes | Metal command encoder fails on concurrent process access. |
| Multi-threaded MLX | Crashes | Not thread-safe. Concurrent mx.eval crashes Metal encoder. |
| Wired memory limit | No benefit | Unified memory rarely pages at model size. |
| GEMM pad stage 0 K=112→128 | Regression | Padding overhead > alignment benefit at this scale. |
| Batch B>1 | Linear scaling, zero amortization | GPU already fully utilized at B=1. |

### Not a bottleneck at production scale

| Technique | Micro-bench | Pipeline impact | Why |
|---|---|---|---|
| Custom Metal GroupNorm | -67% | 0% total pipeline | GroupNorm is ~5ms/tile out of 218ms/tile. Already fused by mx.compile. |
| GELU fast approximation | 0ms | 0ms | GELU compute is negligible vs attention and convolutions. |
| Token dedup / RLT | — | N/A | Windowed attention stages have small token counts. Global stages only 1024/256 tokens. |
| HW video decode | — | 0% | Read is 0.2% of total time. |
| Zero-copy numpy views | — | 0% | Copy at GPU→CPU boundary is unavoidable. |

### Tiling and compile variations

| Technique | Result | Why |
|---|---|---|
| 1024px tiles | Regression | Memory pressure from larger feature maps. |
| Overlap 128→64 at 768px | No difference | Same tile count — last tile shifts back. |
| Overlap 0 | Same tile count | Overlap doesn't affect grid at 768px for 1920×1080. |
| Whole-forward `mx.compile` | No benefit | Per-component compile already optimal. |
| `mx.compile(shapeless=True)` | Unsafe | Hiera uses shape-dependent reshapes. Reductions return stale values. |
| GPU preprocessing for tiled path | +4s slower | CPU preprocessing is already fast enough. |
| `mx.scatter_add` blending | No gain | Copy-on-write, same cost as numpy blend. |
| Deferred eval across tiles | Memory explosion risk | Not tested; would require holding all tile feature maps simultaneously. |

### Final hail mary pass (27 experiments)

All ideas from deep research on MLX internals, Metal optimization, Apple Silicon microarchitecture.

**Tested and measured:**

| Technique | Result | Why |
|---|---|---|
| Space-to-Depth dilated conv (pixel_unshuffle → grouped conv → pixel_shuffle) | +17% regression (1682ms) | pixel_unshuffle/shuffle require physical memory copies (transpose creates non-contiguous view). MLX grouped conv with large group counts has poor GPU utilization. |
| SIMD-aligned attention padding (head_dim 56→64) | +23% regression (1756ms) | `mx.pad` allocates fresh memory. 24 blocks × 3 Q/K/V × 6 tiles = 432 pad ops per frame. Allocation cost dwarfs any SIMD kernel alignment benefit. |
| Granular per-component eval (backbone→eval→decoder→eval→refiner→eval) | +2.7% regression (1495ms) | Extra CPU-GPU sync barriers outweigh memory pool recycling benefit. |
| GPU power pinning (caffeinate -disu) | No effect | GPU reaches peak clocks fast enough during 218ms tile inference. DVFS ramp-up is not a factor. |
| Shader cache pre-warming (dummy tile on model load) | +1.6% regression (1455ms) | Warmup adds ~2-3s startup cost that exceeds JIT savings amortized over 37 frames. |
| MLX memory pool cache-limit removal | Inconsistent (1392-1660ms) | Wild variance. Current 2/2 buffer settings are optimal for consistency. |
| `MTL_SHADER_VALIDATION=0` | No effect | Not enabled by default in production builds. |
| `MLX_MAX_OPS_PER_BUFFER=1` | +24% regression (1783ms) | Too-frequent kernel dispatch overhead. |
| `MLX_MAX_OPS_PER_BUFFER=4, MB=4` | +18% regression (1712ms) | Buffers too large, graph buildup hurts tiled workloads. |
| NCHW transposition sandwiches | Impossible | MLX Conv2d only supports NHWC layout. |

**Triaged as impractical (not testable from Python/MLX):**

| Technique | Why |
|---|---|
| In-place memory mutation (`input_rw_status` flag) | Requires MLX C++ source patch. Not exposed via Python API. |
| Force implicit GEMM for refiner convs | Requires modifying MLX's C++ conv dispatch (`conv.cpp`). |
| SIMD-level fused GroupNorm (simdgroup_reduce_add) | Previous custom GroupNorm: -67% micro but 0% pipeline impact. Compile already fuses it. |
| Sub-tile graph interleaving (merge tile N refiner + tile N+1 backbone) | Granular eval proved sync barriers hurt. Register explosion risk. |
| Fused overlap blending (pre-multiply blend into final conv weights) | Weights frozen at `mx.compile` time. Coordinate-dependent weights require per-tile recompilation. |
| Pointwise shapeless compilation (shapeless=True for decoder/refiner only) | Fixed 768×768 shapes mean standard compile already caches optimally. |
| QoS thread escalation (NSQualityOfServiceUserInteractive) | GPU-bound, not CPU scheduling bound. MLX C++ threads manage Metal dispatch independently. |
| Async CPU-GPU postprocessing overlap | No CPU work left to overlap — async I/O pipeline already hides all write latency. |
| Safetensors mmap bypass | Startup latency is ~2s, not the bottleneck. |
| Texture lookup activations, fast inverse sqrt, ICB, TBDR imageblock abuse, ANE dispatch | Implementation effort: months. Success probability: <10%. Requires Obj-C/C++/Swift Metal extensions. |
| Polynomial softmax approximation | Requires modifying MLX flash attention kernel. High fidelity risk. |
| Fused GeLU/SiLU custom kernel | GELU fast approx already measured at 0ms difference. |

---

## Key lessons

1. **Matting demands bit-exact or near-exact fidelity.** Approximate techniques that work for classification/detection fail here because max error on edges determines visual quality.

2. **Don't assume discrete-GPU optimizations transfer to Apple Silicon.** Int8 quantization, multi-GPU parallelism, and memory pinning are counterproductive on unified memory.

3. **Any operation that adds memory allocation in the hot path hurts.** mx.pad, pixel_unshuffle, grouped conv weight tiling, extra tensors — they all trigger physical copies that cost more than they save.

4. **Extra CPU-GPU sync barriers hurt.** More mx.eval calls = more synchronization overhead, even if they enable better buffer recycling.

5. **Always benchmark at production resolution and tile size.** Micro-benchmarks overstate impact. The only number that matters is wall-clock time on the 37-frame clip.

6. **The biggest wins came from removing work, not optimizing work.** Removing gc.collect saved 530ms. Removing unnecessary outputs saved 300ms. Removing postprocessing saved 155ms. Larger tiles (fewer of them) saved 900ms.

7. **MLX's existing compilation and buffer management are already well-tuned.** The framework's defaults (with the 2/2 buffer settings for non-compiled mode) represent a near-optimal configuration. Attempts to override them consistently made things worse.

---

## What's left

The 0:54 baseline (1456ms/frame median) is the hardware limit for frozen Hiera-Base-Plus + CNN refiner on Apple Silicon via MLX. 106 experiments confirm this. The only remaining paths require changes outside the current project scope:

1. **Model retraining** — knowledge distillation to smaller backbone, depthwise separable refiner, wavelet-domain processing
2. **MLX framework patches** — in-place mutation, conv algorithm selection, improved grouped conv
3. **Native runtime** — bypass Python entirely (see [CorridorKey-Runtime](https://github.com/99oblivius/CorridorKey-Runtime))
4. **Optical flow feature warping** — warp cached features using motion vectors. Theoretically viable but high complexity and uncertain payoff.
