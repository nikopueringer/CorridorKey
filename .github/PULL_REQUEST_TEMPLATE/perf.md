## Hypothesis

<!-- One sentence: what change, what you expect to move, by how much.
     Example: "Replace OpenCV despeckle with torch connected components —
     expect ~15% faster postprocess at 2048² batch 1." -->

## Results

**Hardware / build**
<!-- Paste from the `meta` block of the bench JSON, or fill by hand. -->
- GPU:
- Driver:
- torch:
- Platform:

**Command used**

```bash
uv run --extra cuda python benchmarks/bench_inference.py \
    --synthetic --resolution 2048 --warmup 3 --iters 10
```

**Before / after**

| clip | batch | resolution | infer median (ms) | fps | vram MB | Δ vs baseline |
|------|-------|------------|-------------------|-----|---------|---------------|
|      |       |            |                   |     |         |               |

Quote the **median**, not the mean. See `benchmarks/README.md` for why.

## Artifacts

- Baseline run: `benchmarks/results/<baseline-sha>.json`
- PR run:       `benchmarks/results/<pr-sha>.json`

<!-- Commit both JSONs alongside the code change, or link to existing
     baselines on main. Reviewers should be able to diff the two files. -->

## Reproducibility

- [ ] Ran 3× on the same box; medians agree within 3%
- [ ] `meta.git_dirty: false` in both JSONs
- [ ] No other GPU workloads active during the run
- [ ] Correctness unchanged — or, if output shifts, quality gate results attached

## Notes / caveats

<!-- Anything the reviewer should know: known regressions on other
     hardware, backend-specific behavior, experimental flags, etc.
     "none" is a valid answer. -->
