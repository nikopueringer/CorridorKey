# Testing `feature/misc-optimizations` on CUDA

Quick guide for verifying this branch on an NVIDIA GPU machine.

## Prerequisites

- Python 3.10+
- NVIDIA GPU w/ CUDA (22.7 GB VRAM for full-res inference)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- A test clip: an RGB video + matching alpha hint video (BW mask)

## Setup

```bash
git clone git@github.com:cmoyates/CorridorKey.git
cd CorridorKey
git checkout feature/misc-optimizations
uv sync --group dev
```

Model weights go in `CorridorKeyModule/checkpoints/` (any `.pth` file). These are gitignored — grab from wherever we store them internally.

## Run the benchmark

The benchmark script processes up to 20 frames and reports timing, VRAM, and quality metrics.

### 1. Generate a baseline (unoptimized)

This runs with all optimizations **off** and saves reference outputs for quality comparison:

```bash
uv run python benchmarks/bench_phase.py \
  --clip path/to/input.mp4 \
  --alpha path/to/alpha_hint.mp4 \
  --generate-baseline \
  --no-fp16 \
  --gpu-postprocess \
  --refiner-tile-size 0
```

### 2. Benchmark with optimizations

Run with defaults (Quality preset: FP16 on, tiled refiner 512, 96px overlap):

```bash
uv run python benchmarks/bench_phase.py \
  --clip path/to/input.mp4 \
  --alpha path/to/alpha_hint.mp4
```

This compares against the baseline and prints per-channel pixel diffs + PSNR.

### 3. Full matrix (all presets)

Runs Quality, Fast Preview, Low VRAM, and Legacy presets back-to-back:

```bash
uv run python benchmarks/bench_matrix.py \
  --clip path/to/input.mp4 \
  --alpha path/to/alpha_hint.mp4
```

Outputs a summary table with median frame time and peak VRAM per preset.

## What to look for

- **CUDA peak memory** — main thing we're validating (MPS numbers from M3 aren't directly comparable)
- **Median frame time** — speed improvement per preset
- **Quality diffs** — PSNR should stay above ~40 dB for Quality preset; Fast Preview will be lower due to backbone downscale
- **No crashes/errors** — confirms CUDA codepath works end-to-end

## Quick smoke test (no benchmark clips)

If you just want to verify it loads and runs:

```bash
uv run pytest -m "not gpu"    # CPU-only unit tests
uv run pytest                  # all tests (needs GPU + weights)
```

## Benchmark flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--fp16` / `--no-fp16` | on | FP16 weight casting |
| `--gpu-postprocess` / `--no-gpu-postprocess` | on | Color math on GPU |
| `--backbone-size N` | full res | Backbone resolution (e.g. 1024) |
| `--refiner-tile-size N` | 512 | Refiner tile size (0 = no tiling) |
| `--refiner-tile-overlap N` | 96 | Tile overlap in pixels |
| `--max-frames N` | 20 | Frames to benchmark |
| `--device cuda\|mps\|cpu` | auto | Force device |
