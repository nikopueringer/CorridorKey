# Contributing to CorridorKey

Thanks for your interest in improving CorridorKey! Whether you're a VFX artist, a pipeline TD, or a machine learning researcher, contributions of all kinds are welcome — bug reports, feature ideas, documentation fixes, and code.

## Legal Agreement

By contributing to this project, you agree that your contributions will be licensed under the project's **[CorridorKey Licence](LICENSE)**.

By submitting a Pull Request, you specifically acknowledge and agree to the terms set forth in **Section 6 (CONTRIBUTIONS)** of the license. This ensures that Corridor Digital maintains the full right to use, distribute, and sublicense this codebase, including PR contributions. This is a project for the community, and will always remain freely available here.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

### Dev Setup

```bash
git clone https://github.com/nikopueringer/CorridorKey.git
cd CorridorKey
uv sync --group dev    # installs all dependencies + dev tools (pytest, ruff)
```

That's it. No manual virtualenv creation, no `pip install` — uv handles everything.

### Running Tests

```bash
uv run pytest              # run all tests
uv run pytest -v           # verbose (shows each test name)
uv run pytest -m "not gpu" # skip tests that need a CUDA GPU
uv run pytest --cov        # show test coverage (sources and branch mode configured in pyproject.toml)
```

Most tests run in a few seconds and don't need a GPU or model weights. Tests that require CUDA are marked with `@pytest.mark.gpu` and will be skipped automatically if no GPU is available.

### Apple Silicon (Mac) Notes

If you are contributing on an Apple Silicon Mac, there are a few extra things to be aware of.

**`uv.lock` drift:** Running `uv run pytest` on macOS regenerates `uv.lock` with macOS-specific dependency markers. **Do not commit this file.** Before staging your changes, always run:

```bash
git restore uv.lock
```

**Selecting the compute backend:** CorridorKey auto-detects MPS on Apple Silicon. To test with the MLX backend or force CPU, set the environment variable before running:

```bash
export CORRIDORKEY_BACKEND=mlx   # use native MLX on Apple Silicon
export CORRIDORKEY_DEVICE=cpu    # force CPU (useful for isolating device bugs)
```

**MPS operator fallback:** If PyTorch raises an error about an unsupported MPS operator, enable CPU fallback for those ops:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Linting and Formatting

```bash
uv run ruff check          # check for lint errors
uv run ruff format --check # check formatting (no changes)
uv run ruff format         # auto-format your code
```

CI runs both checks on every pull request. Running them locally before pushing saves a round-trip.

## Making Changes

### Pull Requests

1. Fork the repo and create a branch for your change
2. Make your changes
3. Run `uv run pytest` and `uv run ruff check` to make sure everything passes
4. Open a pull request against `main`

In your PR description, focus on **why** you made the change, not just what changed. If you're fixing a bug, describe the symptoms. If you're adding a feature, explain the use case. A couple of sentences is plenty.

### What Makes a Good Contribution

- **Bug fixes** — especially for edge cases in EXR/linear workflows, color space handling, or platform-specific issues
- **Tests** — more test coverage is always welcome, particularly for `clip_manager.py` and `inference_engine.py`
- **Documentation** — better explanations, usage examples, or clarifying comments in tricky code
- **Performance** — reducing GPU memory usage, speeding up frame processing, or optimizing I/O

### Code Style

- The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting
- Lint rules: `E, F, W, I, B` (basic style, unused imports, import sorting, common bug patterns)
- Line length: 120 characters
- Third-party code in `gvm_core/` and `VideoMaMaInferenceModule/` is excluded from lint enforcement — those are derived from research repos and we try to keep them close to upstream

### Model Weights

The model checkpoint (`CorridorKey.safetensors`, or the legacy `CorridorKey.pth`) and optional GVM/VideoMaMa weights are **not** in the git repo. Most tests don't need them. If you're working on inference code and need the weights, follow the download instructions in the [README](README.md).

## Questions?

Join the [Discord](https://discord.gg/zvwUrdWXJm) — it's the fastest way to get help or discuss ideas before opening a PR.
