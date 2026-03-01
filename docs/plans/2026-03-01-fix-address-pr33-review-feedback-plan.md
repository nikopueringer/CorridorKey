---
title: Address PR #33 Review Feedback
type: fix
date: 2026-03-01
source: https://github.com/nikopueringer/CorridorKey/pull/33#issuecomment-3979207190
---

# Address PR #33 Review Feedback

PR #33 (centralized device selection) received review from `@taylorOntologize`. Four items to address, one question to answer.

## Items

### 1. `[major]` Add tests for `device_utils.py`

Zero test coverage on the core new module. Need tests for all code paths:

- `detect_best_device()`: CUDA available, MPS available, CPU-only
- `resolve_device()`: auto-detect, explicit valid device, explicit unavailable backend (`RuntimeError`), invalid device string (`RuntimeError`), env var override, CLI arg priority over env var
- `clear_device_cache()`: CUDA path, MPS path (after fix), CPU no-op

**File:** `tests/test_device_utils.py`

Pattern: follow `tests/test_color_utils.py` — class-per-function, `monkeypatch` to mock `torch.cuda.is_available()` / `torch.backends.mps.is_available()`. Use `conftest.py` fixtures where applicable.

### 2. `[minor]` Fix `clear_device_cache()` — dead code + missing MPS support

Two sub-issues:

**a)** Function defined at `device_utils.py:64-68` but never called. Third-party code (`pipeline_gvm.py:226`, `inference.py:155`) uses inline `torch.cuda.empty_cache()` guards instead — reasonable since third-party shouldn't import project utils. Wire it into CorridorKey-owned code where cache clearing happens, or remove if no owned call sites exist.

**b)** Docstring says "no-op for CPU/MPS" but `torch.mps.empty_cache()` exists in PyTorch 2.0+. Add MPS cache clearing.

**File:** `device_utils.py`

```python
def clear_device_cache(device: torch.device | str) -> None:
    device_type = device if isinstance(device, str) else device.type
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()
```

### 3. `[minor]` Restore `docs/LLM_HANDOVER.md`

Commit `aa123a5` ("Remove temp design docs") deleted both `DEVICE_SELECTION.md` (intended) and `LLM_HANDOVER.md` (accidental). `README.md` still references it at lines 19 and 137. Restore from git history.

```bash
git show 06260e1:docs/LLM_HANDOVER.md > docs/LLM_HANDOVER.md
```

### 4. `[minor]` Resolve device once in `interactive_wizard`

Currently `device=None` causes auto-detection to run independently in each sub-function (`generate_alphas`, `run_videomama`, `run_inference`). Add early resolution at wizard entry:

**File:** `clip_manager.py`, top of `interactive_wizard()`

```python
def interactive_wizard(win_path, device=None):
    device = device or resolve_device()
    # ... rest of function
```

### 5. `[question]` Apple Silicon end-to-end testing

Reviewer asks: has `torch.autocast(device_type="mps", dtype=torch.float16)` at `inference_engine.py:160` been tested with actual model weights on Apple Silicon?

**Action:** Answer in PR comment. ~~If not tested, note it as known limitation.~~ **Tested successfully on M3 Max.** `PYTORCH_ENABLE_MPS_FALLBACK=1` env var (already documented in README) handles unsupported ops. Two bugs found and fixed during testing (float16 resize crash, alpha video-in-directory not detected).

## Acceptance Criteria

- [x] `tests/test_device_utils.py` covers all `resolve_device()` paths (6+ cases)
- [x] `tests/test_device_utils.py` covers `detect_best_device()` (3 cases)
- [x] `tests/test_device_utils.py` covers `clear_device_cache()` (3 device types)
- [x] `clear_device_cache()` calls `torch.mps.empty_cache()` on MPS
- [x] `clear_device_cache()` kept as public API util (no owned call sites, not dead — it's part of the module interface)
- [x] `docs/LLM_HANDOVER.md` restored from pre-deletion commit
- [x] `interactive_wizard()` resolves device once at entry
- [x] All existing tests still pass (99/99)
- [ ] PR comment replies posted for each item + question answer

## Unresolved Questions

- ~~Wire `clear_device_cache()` into owned code or remove as dead code?~~ **Resolved:** No owned code calls `empty_cache()`. Only third-party (`VideoMaMaInferenceModule/inference.py:156`, `gvm_core/pipelines/pipeline_gvm.py:227`). Kept as public API util.
- ~~Has anyone tested on Apple Silicon with real weights? If not, flag as known gap.~~ **Resolved:** Tested successfully on M3 Max MacBook. Required two fixes: `float16->float32` cast before `cv2.resize` in `inference_engine.py`, and alpha video-in-directory fallback in `clip_manager.py`.
