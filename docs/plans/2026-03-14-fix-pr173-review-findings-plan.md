---
title: "fix: address PR #173 code review findings"
type: fix
date: 2026-03-14
pr: 173
branch: feature/4k-mlx-test
---

# fix: address PR #173 code review findings

## Overview

Address 7 code review findings from PR #173. Spec-flow analysis refined several findings — notably Fix #2 (thread safety) has no actual cross-thread contention, and Fix #4 (runtime import) is a non-issue since `backend.frame_io` is already in `sys.modules`.

## Fixes

### Fix 1 — Eliminate engine type duck-typing (P1)

**Problem**: `clip_manager.py` uses `hasattr(engine, "_engine")` to detect MLX adapter, and `getattr(engine, "last_frame_timing", None)` for timing. Fragile, breaks if either class changes.

**Approach**: Add `enabled_outputs` param to `CorridorKeyEngine.process_frame()` (Torch). This eliminates the branch entirely — both engines accept the same kwargs. The Torch engine can skip comp/processed computation when not requested (free speedup).

For timing, add `last_frame_timing: dict = {}` to `CorridorKeyEngine` too (always empty, but `getattr` becomes a safe uniform access).

**Files**:
- `CorridorKeyModule/inference_engine.py` — add `enabled_outputs` param, skip comp/processed when not requested, add `last_frame_timing` attr
- `clip_manager.py` — remove `hasattr` branch, pass `enabled_outputs` unconditionally

### Fix 2 — Thread safety comment (P1 → P3, downgraded)

**Problem**: Comment says `list.append` needs GIL for thread safety. Spec-flow analysis shows each list in `phase_times` is only ever written by a single thread (read→main, infer/write/postprocess→writer). No actual cross-thread contention exists.

**Approach**: Fix the misleading comment. No lock needed.

**Files**:
- `clip_manager.py` — update comment to accurately describe the single-writer-per-list pattern

### Fix 3 — Reader thread silent frame drops (P2)

**Problem**: `_reader_worker` does `continue` on read failure, so the main thread receives fewer frames than `num_frames`. Progress callback never reaches 100%.

**Approach**: Enqueue a sentinel tuple `(frame_index, frame_stem, None, None, 0.0)` so the main thread can log the skip and still advance the progress counter. Main thread skips inference for that frame but calls `on_frame_complete`. This keeps progress accurate and output numbering correct (holes in sequence are acceptable — downstream tools handle missing frames).

**Files**:
- `clip_manager.py::_reader_worker` — enqueue skip sentinel instead of `continue`
- `clip_manager.py::run_inference` main loop — detect skip sentinel, log, advance progress

### Fix 4 — Runtime import in reader thread (P2 → skip)

**Analysis**: `backend.frame_io` is already imported at module scope (line 23-24). The deferred import at line 622 is just a `sys.modules` lookup. Not a real risk. Skip.

### Fix 5 — Cache linear checkerboard (P2)

**Problem**: `_wrap_mlx_output` calls `cu.srgb_to_linear()` on the cached sRGB checkerboard every frame.

**Approach**: Cache the linear version directly. Rename `_get_checkerboard` → `_get_checkerboard_linear` and have it return the already-converted linear array. Verify `composite_straight` allocates a new array (it does — numpy broadcasting `A * alpha + B * (1-alpha)` creates new array). Set `arr.flags.writeable = False` on cached arrays as a safety net.

**Files**:
- `CorridorKeyModule/backend.py` — modify `_get_checkerboard` to cache+return linear, mark read-only

### Fix 6 — Writer hardcodes EXR format (P3 → defer)

**Analysis**: `clip_manager.py` is the CLI path; `backend/service.py` is the web service path. They are independent. Adopting `OutputConfig` in clip_manager would couple them unnecessarily. The CLI currently only needs EXR+PNG. Defer until format flexibility is actually needed.

### Fix 7 — Magic number MAX_SINGLE_KERNEL (P3)

**Problem**: `MAX_SINGLE_KERNEL = 11` is a local constant inside `clean_matte()`.

**Approach**: Move to module level in `color_utils.py` with a comment explaining the 11px choice (CV morphology kernel area grows quadratically — 11px balances per-pixel cost vs iteration count, benchmarked at ~4-5x faster than single large kernel).

**Files**:
- `CorridorKeyModule/core/color_utils.py` — move constant to module level

## Implementation Order

No hard dependencies. Suggested order by impact:

1. Fix 1 (eliminates duck-typing + Torch speedup)
2. Fix 5 (perf win, small change)
3. Fix 3 (correctness)
4. Fix 7 (trivial)
5. Fix 2 (comment fix)

## Acceptance Criteria

- [x] No `hasattr(engine, "_engine")` in codebase
- [x] Both engines accept `enabled_outputs` param
- [x] Torch engine skips comp/processed when not in `enabled_outputs`
- [x] Reader thread frame drops produce accurate progress (100% reached)
- [x] Linear checkerboard cached (verify with `time.perf_counter` in debug log)
- [x] `MAX_SINGLE_KERNEL` at module level in color_utils.py
- [x] Thread safety comment accurately describes single-writer pattern
- [x] All existing tests pass

## Unresolved Questions

- Should Torch engine's `enabled_outputs` skip despill too when only matte requested? (probably not — despill feeds into processed which could be toggled on later in same session via service.py)
- Is `_get_checkerboard` LRU cache with `maxsize=4` the right size? Only one resolution per clip, so `maxsize=1` would suffice for CLI path
