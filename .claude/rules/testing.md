---
paths:
  - "tests/**"
  - "conftest.py"
---

# Testing Conventions

## Running tests

```bash
uv run pytest              # all tests (~3s, no GPU needed for most)
uv run pytest -v           # verbose
uv run pytest -m "not gpu" # skip GPU tests explicitly
```

## Markers

- `@pytest.mark.gpu` — requires CUDA or MPS. Auto-skipped when no GPU is detected.
- `@pytest.mark.slow` — long-running tests.
- `strict-markers = true` — all custom markers must be registered in `conftest.py`.

## Fixtures (in `tests/conftest.py`)

- `sample_frame_rgb` — 64x64 float32 RGB frame in [0, 1]
- `sample_mask` — matching 64x64 binary float32 mask
- `tmp_clip_dir` — temp directory with expected clip structure (Input/, AlphaHint/, Output/)

## Patterns

- Test both numpy and torch paths for dual-backend functions in `color_utils.py`.
- Mock the model for `inference_engine.py` tests — no GPU or weights needed.
- `test_gamma_consistency.py` documents the known gamma 2.2 vs piecewise sRGB inconsistency. It is documentation-as-code, not a regression test. Do not "fix" it without also fixing the source inconsistency.
- No `print()` in tests — use `logging` or `pytest.fail()` for diagnostics.
