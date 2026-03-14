---
title: "gc.collect and mx.clear_cache are massive bottlenecks on Apple Silicon"
category: performance
tags:
  - mlx
  - apple-silicon
  - garbage-collection
  - unified-memory
  - counterintuitive
component: corridorkey-mlx
symptoms:
  - Per-tile overhead ~15ms seems small but accumulates to ~530ms/frame
  - Inference appears memory-bound but is actually CPU-bound on GC
root_cause: >
  gc.collect() forces a full Python garbage collection sweep every tile.
  mx.clear_cache() forces Metal to release buffer cache between tiles.
  On unified memory with ample headroom (36GB), both are pure overhead —
  the system manages memory fine without forced collection.
resolved_by: corridorkey-mlx 303001f, 9a852c2
date: 2026-03-13
---

# gc.collect + mx.clear_cache Are the #1 Hidden Bottleneck on Apple Silicon

## Problem

The MLX tiling loop called `gc.collect()` + `mx.clear_cache()` after every tile (6 tiles/frame). The engine's `process_frame` also called both after inference. Total overhead: **~530ms/frame** — larger than ANY other single optimization.

## Why This Is Counterintuitive

The original code was written defensively: "free memory between tiles to avoid OOM on constrained devices." On discrete GPUs this makes sense. On Apple Silicon unified memory with sufficient headroom, it's pure overhead because:

1. **gc.collect()** walks the entire Python object graph. With large numpy/MLX arrays, this is expensive (~10-30ms per call).
2. **mx.clear_cache()** forces Metal to release cached GPU buffers. Reallocation is then required for the next tile, adding latency.
3. Python's reference counting already frees most objects immediately on `del`. The cyclical GC is only needed for reference cycles, which don't occur in the inference pipeline.

## Solution

Remove ALL `gc.collect()` and `mx.clear_cache()` from the hot path. Keep `del` statements for reference cleanup.

## Key Metrics

| Config | Before | After |
|--------|--------|-------|
| Per-tile overhead | ~15ms gc + ~15ms clear_cache | ~0ms |
| Per-frame total | ~530ms wasted | ~0ms |
| Infer ms/frame | 2030 | 1474 |

## Prevention

- **Never add gc.collect() to hot loops** without benchmarking first
- On unified memory systems, memory pressure is visible via `mx.metal.get_active_memory()` — check before adding defensive GC
- If GC IS needed (low-memory devices), make it conditional on available memory, not unconditional per-iteration
