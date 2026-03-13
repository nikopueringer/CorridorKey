# CorridorKey — AGENTS.md

> Agent-facing project guide following the [AGENTS.md open format](https://agents.md).
> For deeper architectural detail, see [`docs/LLM_HANDOVER.md`](docs/LLM_HANDOVER.md).

## Project Overview

**CorridorKey** is a neural-network-based green screen removal tool built for professional VFX pipelines. Unlike traditional keyers that produce binary masks, CorridorKey physically unmixes the foreground from the green screen at every pixel — including semi-transparent regions like motion blur, hair, and out-of-focus edges.

**Core inputs:**

- **RGB image** — the green screen plate (sRGB color gamut).
- **Coarse Alpha Hint** — a rough black-and-white mask isolating the subject (does not need to be precise).

**Core outputs:**

- **Alpha** — a clean, linear alpha channel.
- **Foreground Straight** — the un-multiplied straight color of the foreground element (sRGB), with the green screen contribution removed.

**License:** [CC-BY-NC-SA-4.0](LICENSE)

## Architecture & Dataflow

### GreenFormer Architecture

The core model is called the **GreenFormer**:

- **Backbone:** A `timm` **Hiera** vision-transformer (`hiera_base_plus_224.mae_in1k_ft_in1k`), patched to accept 4 input channels (RGB + Coarse Alpha Hint).
- **Decoders:** Multiscale feature fusion heads predicting coarse Alpha (1 ch) and Foreground (3 ch) logits.
- **Refiner (`CNNRefinerModule`):** A custom CNN head with dilated residual blocks. It takes the original RGB input and the coarse predictions, outputting purely additive delta logits applied to the backbone outputs before final Sigmoid activation.

### Dataflow Rules

1. **Tensor range:** Model input and output are strictly `[0.0, 1.0]` float tensors. The foreground is sRGB; the alpha is linear.
2. **EXR pipeline:** To build the `Processed` EXR output, the sRGB foreground is converted via the piecewise `srgb_to_linear()` function, then premultiplied by the linear alpha, and saved as half-float EXR (`cv2.IMWRITE_EXR_TYPE_HALF`).
3. **Inference resizing:** The engine is trained on **2048×2048** crops. `inference_engine.py` uses OpenCV **Lanczos4** to resize arbitrary input to 2048×2048, runs inference, then resizes predictions back to the original resolution.
4. **Despill:** A luminance-preserving `despill()` function removes residual green contamination from the foreground.

> ⚠️ **Gamma 2.2 warning:** Never apply a pure mathematical gamma 2.2 curve. Always use the piecewise sRGB transfer functions defined in `color_utils.py`. A naive power-law curve will produce incorrect results in the toe region and break compositing math.

## Key File Map

| Path | Responsibility |
|---|---|
| `CorridorKeyModule/core/model_transformer.py` | GreenFormer PyTorch architecture (Hiera backbone, decoders, CNNRefinerModule) |
| `CorridorKeyModule/inference_engine.py` | `CorridorKeyEngine` class — loads weights, handles 2048×2048 resize and frame processing API |
| `CorridorKeyModule/core/color_utils.py` | Pure math for compositing: `srgb_to_linear()`, `linear_to_srgb()`, `premultiply()`, `despill()` |
| `clip_manager.py` | User-facing CLI wizard — directory scanning, inference settings, piping data to the engine |
| `device_utils.py` | Compute device detection and selection (CUDA / MPS / CPU), backend resolution |
| `backend/` | FastAPI-based backend service: job queue, project management, FFmpeg tools, frame I/O |

## Dev Environment Setup

**Prerequisites:** Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

uv handles Python installation, virtual environment creation, and package management — no manual `pip install` or virtualenv setup required.

```bash
git clone https://github.com/nikopueringer/CorridorKey.git
cd CorridorKey
uv sync --group dev    # installs all dependencies + dev tools (pytest, ruff, hypothesis)
```

## Build & Test Commands

```bash
uv run pytest              # run all tests
uv run pytest -v           # verbose output
uv run pytest -m "not gpu" # skip GPU-dependent tests
uv run ruff check          # lint check
uv run ruff format --check # formatting check (no changes)
uv run ruff format         # auto-format
```

Tests that require a CUDA GPU are marked with `@pytest.mark.gpu` and are automatically skipped when no GPU is available. CI runs `pytest -m "not gpu"` to exclude them.

## Code Style

The project uses **[Ruff](https://docs.astral.sh/ruff/)** for both linting and formatting.

| Setting | Value |
|---|---|
| Lint rules | `E`, `F`, `W`, `I`, `B` |
| Line length | 120 |
| Target version | `py311` |
| Excluded dirs | `gvm_core/`, `VideoMaMaInferenceModule/` |

`gvm_core/` and `VideoMaMaInferenceModule/` are third-party research code kept close to upstream — they are excluded from lint enforcement.

## Platform-Specific Caveats

### Apple Silicon (macOS)

- **MPS operator fallback:** Some PyTorch operations are not yet implemented for MPS. Enable CPU fallback:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```

### Windows

- **CUDA 12.8:** GPU acceleration on Windows requires NVIDIA drivers supporting **CUDA 12.8** or higher. Older drivers will cause a silent fallback to CPU.

## Prohibited Actions

1. **Do not apply a pure gamma 2.2 curve.** Always use the piecewise sRGB transfer functions in `color_utils.py`. A naive `pow(x, 2.2)` breaks the toe region and produces incorrect compositing results.
2. **Do not modify files inside `gvm_core/` or `VideoMaMaInferenceModule/`.** These are third-party research modules kept close to upstream. Changes should be made in wrapper code or upstream PRs.

## PR Workflow & GitHub Templates

### Workflow

1. Fork the repo and create a branch for your change.
2. Make your changes.
3. Run `uv run pytest` and `uv run ruff check` to verify everything passes.
4. Open a pull request against `main`.

PR descriptions should focus on **why** the change was made, not just what changed. If fixing a bug, describe the symptoms. If adding a feature, explain the use case.

Before preparing any pull request, check `.github/` for PR templates, issue templates, and CI workflows.

### PR Template

The repository includes a PR template (`.github/pull_request_template.md`) with the following structure:

- **"What does this change?"** — Explain the motivation and scope of the change.
- **"How was it tested?"** — Describe specific test steps or commands run to verify correctness.
- **Checklist:**
  - `uv run pytest` passes
  - `uv run ruff check` passes
  - `uv run ruff format --check` passes

Fill in all sections thoroughly. The "What does this change?" section should explain motivation, and "How was it tested?" should describe specific test steps or commands run.

### CI Workflow (`ci.yml`)

Runs on every push and pull request to `main`:

- **Lint job:** `ruff format --check` + `ruff check`.
- **Test job:** `pytest -v --tb=short -m "not gpu"` on Python **3.10** and **3.13**. GPU tests are excluded via the `-m "not gpu"` marker filter.

### Docs Workflow (`docs.yml`)

Triggers on pushes to `main` that change files matching `docs/**` or `zensical.toml`. Builds and deploys the documentation site to GitHub Pages via **Zensical**.

## Documentation Accuracy

When making code changes, evaluate whether the change affects the accuracy of existing documentation. If a code change alters behavior, CLI flags, file paths, or configuration described in the docs, flag or update the outdated documentation.

Documentation files to check:

- `README.md`
- `CONTRIBUTING.md`
- `AGENTS.md`
- `docs/LLM_HANDOVER.md`
- All pages under `docs/`

## AI Directives

- **Skip basic tutorials.** The user is a VFX professional and coder. Dive straight into advanced implementation guidance, but document math thoroughly.
- **Prioritize performance.** This is video processing — every `.numpy()` transfer or `cv2.resize` matters in a loop running on 4K footage.
- **Check sRGB-to-linear conversion order.** If the user reports "crushed shadows" or "dark fringes", the problem is almost certainly an sRGB-to-linear conversion step happening in the wrong order inside `color_utils.py`.

## Further Reading

- [`docs/LLM_HANDOVER.md`](docs/LLM_HANDOVER.md) — Detailed architecture walkthrough, dataflow properties, inference pipeline, and AI directives for the CorridorKey codebase.
