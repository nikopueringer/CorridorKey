---
title: "fix: Address PR #54 review concerns"
type: fix
date: 2026-03-07
pr: 54
---

# Fix PR #54 Review Concerns

Address code review feedback on the VRAM & performance optimizations PR before merge.

## Tasks

### 1. Extract shared CLI optimization args

Both `corridorkey_cli.py` and `clip_manager.py` define identical `--fp16`, `--gpu-postprocess`, `--backbone-size`, `--refiner-tile-size`, `--refiner-tile-overlap` args. Extract to shared function.

**Approach:** Add `add_optimization_args(parser)` to `clip_manager.py` (since `corridorkey_cli.py` already imports from it). Both files call this function instead of duplicating definitions.

**Files:**
- `clip_manager.py` — define `add_optimization_args(parser)`, refactor local usage
- `corridorkey_cli.py` — import and call `add_optimization_args(parser)`

### 2. Remove per-tile `empty_cache()` calls

`model_transformer.py:294` calls `torch.cuda.empty_cache()` per tile — expensive (synchronizes CUDA stream, fragments allocator). The `del` statements on line 292 already free tile tensors; PyTorch's allocator reuses that memory.

**Approach:** Remove the `empty_cache()` block entirely. The `del` + natural allocator reuse is sufficient. If OOM occurs on extreme tile counts, users can reduce tile count rather than paying the sync cost per tile.

**Files:**
- `CorridorKeyModule/core/model_transformer.py` — remove lines 293-296

### 3. Fix `_clamp` ignoring its `min` parameter

`color_utils.py:40` hardcodes `min=0.0` in `np.clip` regardless of the `min` parameter passed.

**Approach:** Use the `min` parameter in both branches.

```python
# Before
def _clamp(x, min: float):
    if _is_tensor(x):
        return x.clamp(min=0.0)      # ignores min param
    else:
        return np.clip(x, 0.0, None)  # ignores min param

# After
def _clamp(x, min: float):
    if _is_tensor(x):
        return x.clamp(min=min)
    else:
        return np.clip(x, min, None)
```

**Files:**
- `CorridorKeyModule/core/color_utils.py` — fix `_clamp` function

### 4. Fix lossy threshold inversion

`test_quality_gate.py:52` has `max_abs_err=0.02` for lossy — *stricter* than fp16's `0.04`. Lossy (backbone downsampling + tiling) should tolerate more error.

**Approach:** Swap to `max_abs_err=0.06` for lossy (higher than fp16's 0.04, since lossy includes resolution changes + tile seam artifacts on top of fp16 rounding).

**Files:**
- `tests/test_quality_gate.py` — update `LOSSY_THRESHOLDS["max_abs_err"]`

### 5. Fix hardcoded path in `.claude/settings.json`

The `Stop` hook hardcodes `/Users/cristopheryates/Documents/Projects/Python/CorridorKey`. This breaks for other contributors.

**Approach:** Use `$CLAUDE_PROJECT_DIR` or relative path. The hook runs in the project directory by default, so `cd` is unnecessary — just run `uv run ruff check --fix` directly.

```json
"command": "uv run ruff check --fix 2>&1 | tail -20"
```

**Files:**
- `.claude/settings.json` — remove hardcoded `cd` prefix

### 6. Document bicubic vs Lanczos4 difference

GPU path uses `F.interpolate(bicubic)`, CPU path uses `cv2.INTER_LANCZOS4`. These produce slightly different ringing. Add a comment explaining the tradeoff.

**Approach:** Add a brief comment in `_postprocess_gpu` noting the difference and why it's acceptable (avoids PCIe bottleneck, quality delta is within fp16 thresholds).

**Files:**
- `CorridorKeyModule/inference_engine.py` — add comment at line 232

### 7. Add justification comment for humility clamp removal

The commented-out clamp block in `model_transformer.py:326-330` says "User requested" but doesn't explain the quality/stability tradeoff.

**Approach:** Replace the vague comment with a brief rationale: clamping limited refiner correction range, causing visible banding in low-contrast regions. FP16 autocast handles numerical stability.

**Files:**
- `CorridorKeyModule/core/model_transformer.py` — rewrite comment block

## Acceptance Criteria

- [x] `--fp16` and other optimization flags defined in one place only
- [x] `uv run pytest -m "not gpu"` passes
- [x] `uv run ruff check` clean
- [x] No hardcoded absolute paths in committed files
- [x] `_clamp` uses its `min` parameter
- [x] Lossy thresholds are less strict than fp16 thresholds

## Unresolved Questions

- Lossy `max_abs_err` value: 0.06 is a guess. Should we run the benchmark matrix to calibrate?
- Should `.claude/settings.json` be gitignored entirely? Other contributors may have different hooks.
- The tiled refiner CPU accumulators (GPU→CPU per tile) — keep as-is or move accumulators to GPU? Keeping on CPU was the original VRAM-saving intent but costs N PCIe transfers. Separate concern from this fix PR?
