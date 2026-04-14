# benchmarks/corpus/

Fixed clip set used by `bench_inference.py` to produce comparable numbers.
**Do not check video files into git.** This README is the source of truth
for what should be here; contributors download the clips locally.

## Target composition

The corpus is intentionally small and heterogeneous — 10 clips, ~30-60
frames each, covering the real difficulty distribution of CorridorKey's
workload. The idea is that a change that helps on "easy" clips but hurts
on "hair" clips is visible immediately in the per-clip summary.

| Slot | Category | What it stresses |
|---|---|---|
| `01_easy_static.mp4`     | clean chroma, locked-off camera | baseline / kernel launch overhead |
| `02_easy_motion.mp4`     | clean chroma, panning camera | decode + H2D |
| `03_hair_01.mp4`         | loose hair against green | matting quality, ambiguous edges |
| `04_hair_02.mp4`         | fine hair + fast head turn | hair + motion blur |
| `05_transparent.mp4`     | glass / fabric with partial alpha | refiner quality, edge cases |
| `06_motion_blur.mp4`     | fast subject motion | diffusion-only territory |
| `07_compressed.mp4`      | web-source H.264 artifacts | robustness to input noise |
| `08_4k_static.mp4`       | 4K UHD, locked-off | I/O + VRAM |
| `09_4k_motion.mp4`       | 4K UHD, handheld | pipeline at max res |
| `10_lowlight.mp4`        | underexposed + grain | edge / noise robustness |

Resolutions: clips are stored at their native resolution. The harness
resizes to `--resolution` on decode, so the same corpus exercises 1080p,
2K, or 4K modes just by changing the flag.

## Conventions

- Filenames are stable. Never rename. Benchmarks compare across runs by
  `clip` field in the JSON output, which comes from `Path.stem`.
- `max_frames` in `load_video_clip` is 60 by default. Longer clips are
  fine — only the first 60 frames get used.
- No alpha sidecars. The harness generates a naive green-screen mask
  heuristic inline; a real mask source (BiRefNet) can be added later as
  an optional per-clip `*.alpha.mp4`.

## Provenance

Each clip needs a `provenance` entry below. Licensing matters — we don't
want to bake anything into a benchmark suite that we can't redistribute
the measurement of.

| Slot | Source | License | sha256 |
|---|---|---|---|
| `01_easy_static.mp4`   | _TBD_ | _TBD_ | _TBD_ |
| `02_easy_motion.mp4`   | _TBD_ | _TBD_ | _TBD_ |
| `03_hair_01.mp4`       | _TBD_ | _TBD_ | _TBD_ |
| `04_hair_02.mp4`       | _TBD_ | _TBD_ | _TBD_ |
| `05_transparent.mp4`   | _TBD_ | _TBD_ | _TBD_ |
| `06_motion_blur.mp4`   | _TBD_ | _TBD_ | _TBD_ |
| `07_compressed.mp4`    | _TBD_ | _TBD_ | _TBD_ |
| `08_4k_static.mp4`     | _TBD_ | _TBD_ | _TBD_ |
| `09_4k_motion.mp4`     | _TBD_ | _TBD_ | _TBD_ |
| `10_lowlight.mp4`      | _TBD_ | _TBD_ | _TBD_ |

Once real clips are landed, replace `_TBD_` with a URL + SPDX license
identifier + `sha256sum` output. The sha256 is the part that makes the
corpus reproducible across machines.

## Don't have the corpus yet?

Run the harness with `--synthetic`. It generates a deterministic
gradient-noise clip entirely in RAM:

```bash
uv run python benchmarks/bench_inference.py --synthetic --resolution 2048
```

This exists specifically so the harness can be smoke-tested and the
reproducibility target (~3% across runs) can be validated before any real
clips are assembled. **Do not quote synthetic-mode numbers as benchmark
results** — they exercise the engine but they don't represent real video
workloads.
