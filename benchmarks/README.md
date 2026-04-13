# benchmarks/

Shared measurement harness for CorridorKey inference. The goal is a single,
reproducible "ruler" that any perf PR can cite — so numbers from different
authors, machines, and branches are actually comparable.

This directory does **not** change pipeline behavior. It's a read-only
observer of the existing `CorridorKeyEngine` path.

## Setup

The harness reuses the main project env. If you haven't installed deps yet:

```bash
# CUDA GPU (Linux, Windows, RTX 30xx/40xx/50xx)
uv sync --extra cuda

# ROCm GPU
uv sync --extra rocm

# Apple Silicon
uv sync --extra mlx
```

Without an explicit `--extra`, `uv` resolves to the **CPU-only** torch wheel and
`torch.cuda.is_available()` returns False. The harness will then still run but
on CPU, which is not what you want.

> **RTX 50-series note.** Blackwell (`sm_120`) needs `torch 2.8.0+cu128` or
> newer. The `cuda` extra in `pyproject.toml` already points at the cu128
> wheel index, so `uv sync --extra cuda` is sufficient — you do not need to
> install torch by hand.

## Quickstart

```bash
# Smoke test with synthetic frames — no corpus required
uv run --extra cuda python benchmarks/bench_inference.py --synthetic --iters 5

# Real corpus (drop .mp4/.mov files into benchmarks/corpus/ first)
uv run --extra cuda python benchmarks/bench_inference.py \
    --corpus benchmarks/corpus \
    --resolution 2048 --warmup 3 --iters 10 \
    --output benchmarks/results/$(date +%Y%m%d-%H%M%S).json

# Batch-size sweep (exercises engine.process_frame with a batched ndarray)
uv run --extra cuda python benchmarks/bench_inference.py --synthetic --batch 1,2,4,8
```

`uv run` re-resolves the env each call. You **must** pass `--extra cuda`
(or `--extra rocm` / `--extra mlx`) every time, otherwise uv will silently
swap the active torch back to the CPU wheel. Pick whichever extra matches
the accelerator you installed above.

## Platform notes

**Windows + triton-windows (RTX 50xx).** At the time of writing,
`torch.compile` autotune on Blackwell hits `OutOfMemoryError: out of resource:
triton_mm Required: 131072 Hardware limit: 101376` during Inductor kernel
selection. The harness catches the compile failure and falls back to eager
mode — you will see `Model compilation failed. Falling back to eager mode.`
in the output. Numbers collected this way are still self-consistent (eager
vs eager is a fair comparison), but don't compare them against compiled runs
on other boxes. Tracked for a future fix.

**Checkpoint auto-download.** On first run the engine downloads
`CorridorKey_v1.0.pth` (~382 MB) from HuggingFace into
`CorridorKeyModule/checkpoints/`. If the auto-download 404s, grab it manually
from `https://huggingface.co/nikopueringer/CorridorKey_v1.0` and drop it in
that directory — `_discover_checkpoint` globs `*.pth` so any filename works.

**Large-batch upstream bug.** `--batch 8` at `--resolution 2048` currently
crashes inside `CorridorKeyModule/core/color_utils.py::connected_components`
with `RuntimeError: n cannot be greater than 2^24+1 for Float type` —
`torch.randperm` with `dtype=torch.float32` can't handle
`batch*W*H > 16,777,217`. Stick to batches ≤ 4 at 2048², or drop resolution
for larger sweeps, until that's patched upstream.

## Scope and next steps

**What it currently covers**

- `CorridorKeyEngine` forward pass (GreenFormer / Hiera), single or batched
- Synthetic frames or real clips from `benchmarks/corpus/`
- Per-run `inference_ms`, `decode_ms`, `total_ms_per_frame`, `vram_peak_mb`,
  each reported as `{median, mean, p95, min, max, n}`
- Structured JSON output with git SHA, GPU name, torch version, full run config

**Next steps**

Things this harness doesn't yet do. Each is its own follow-up bench file
or module that will share the same conventions (CUDA-synced timers,
warmup discard, JSON schema, stat helpers):

1. **Quality gate** — a `benchmarks/metrics/` module (SAD, MSE, gradient
   error, connectivity error) so perf PRs that change model output can be
   checked for regressions against a golden corpus. Needs to land before
   any optimization that touches the forward pass (quantization, token
   merging, fused attention) can ship safely.
2. **End-to-end timing** — a `bench_e2e.py` that drives
   `clip_manager.run_inference` instead of the engine directly, so the
   numbers include video decode, EXR writes, and the serial per-frame
   loop. That's what a user actually waits for; this harness is narrower
   than that by design.
3. **Isolated decode benchmark** — a `bench_decode.py` that compares
   OpenCV vs decord vs PyAV throughput with no inference involved, so
   decoder-swap experiments have their own ruler.
4. **BiRefNet path** — its own bench file; the alpha-hint generator has a
   completely different hot loop from GreenFormer.
5. **VideoMaMa / GVM path** — its own bench file; the SVD-based temporal
   matting stack (UNet + temporal VAE) is architecturally separate from
   GreenFormer.
6. **Profiler wrappers** — canonical `torch.profiler` / NSight / `rocprof`
   invocations so traces from different runs use the same format and are
   directly comparable.

## What it measures

Per clip, per batch size:

| Field | Meaning |
|---|---|
| `decode_ms` | wall time for `_decode_pair` (mostly a copy right now — real decode benchmarking is a future `bench_decode.py`) |
| `inference_ms` | `engine.process_frame` wall time (single frame or batched ndarray), CUDA-synced |
| `total_ms_per_frame` | `(decode_ms + inference_ms) / batch` |
| `fps_median` | `1000 / total_ms_per_frame.median` |
| `vram_peak_mb` | `torch.cuda.max_memory_allocated` after the run |

Each timing is reported as `{median, mean, p95, min, max, n}`. Always quote
the **median**, not the mean — means hide stalls.

## Reproducibility rules

The harness is only useful if runs on the same box agree to within ~3%.
That means:

1. **Warmup iterations are separate from measured iterations.** First calls
   pay compile, cudnn autotune, and cold-cache costs. Default `--warmup 3`.
2. **`torch.cuda.synchronize()` before every timer stop.** Otherwise you're
   timing kernel launches, not completion.
3. **`cudnn.benchmark=True` and TF32 enabled.** A perf-tuned production
   config, not the default. We're measuring the ceiling, not the floor.
4. **Fixed seeds on synthetic data.** `np.random.default_rng(seed=0)` so two
   runs produce the same frames.
5. **No background jobs.** Close your browser. Seriously.
6. **Report the GPU.** The harness writes `meta.gpu`, `meta.torch_version`,
   `meta.git_sha`, `meta.git_dirty` into every JSON output. A benchmark run
   with `git_dirty: true` is noted but not rejected — use your judgment.

If you can't reproduce a number within 3% across three runs on the same
box, **the benchmark is lying and you need to fix it before using it**. Do
not optimize against a noisy ruler.

## Output format

JSON, schema version 1:

```json
{
  "meta": {
    "schema": 1,
    "timestamp": "2026-04-12T...",
    "git_sha": "d836296",
    "git_dirty": false,
    "gpu": "NVIDIA GeForce RTX 4090",
    "torch_version": "2.8.0+cu121",
    "cuda_version": "12.1",
    "hip_version": null,
    "platform": "Linux-6.5...",
    "python": "3.11.9",
    "config": {
      "resolution": 2048, "batches": [1],
      "warmup": 3, "iters": 10,
      "device": "cuda", "precision": "fp16",
      "mixed_precision": true, "synthetic": false,
      "checkpoint": "CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth"
    }
  },
  "results": [
    {
      "clip": "hair_01", "source": "video:hair_01.mp4",
      "batch": 1, "resolution": 2048, "num_frames": 60,
      "inference_ms": {"median": 38.2, "p95": 41.0, "mean": 38.7, ...},
      "total_ms_per_frame": {"median": 39.4, ...},
      "fps_median": 25.4,
      "vram_peak_mb": 7430.0,
      ...
    }
  ]
}
```

Add fields freely in new perf PRs. Don't rename existing fields — downstream
comparison scripts depend on the schema. If a field must change, bump
`SCHEMA_VERSION` in `bench_inference.py`.

## Recommended PR workflow for perf changes

Once this harness is merged, the ask for future perf PRs becomes:

1. Run the harness on `main` with a pinned corpus and commit the JSON to
   `benchmarks/results/baseline-<sha>.json` (or just paste the table in
   the PR body).
2. Apply your change.
3. Run the harness again on the same machine + same corpus.
4. Paste the before/after summary table in the PR body.
5. If the change touches model output, run the (future) quality gate.
6. Note the GPU model and `torch.__version__` in the PR.

That's it. No omnibus refactors, no 100-experiment design docs. One ruler,
one number, one PR.
