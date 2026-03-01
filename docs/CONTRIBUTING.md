# Contributing to CorridorKey

Thank you for your interest in contributing to CorridorKey! This guide covers how to set up a development environment, the project's conventions, and how to submit changes.

---

**You don't need to be a programmer to contribute.** Bug reports, workflow guides, test footage feedback, and documentation improvements are all enormously valuable.

---

## Table of Contents

1. [Getting Involved](#getting-involved)
2. [Contributing Without Code](#contributing-without-code)
3. [Development Setup](#development-setup)
4. [Project Conventions](#project-conventions)
5. [Submitting Changes](#submitting-changes)
6. [Code Review Checklist](#code-review-checklist)
7. [Licensing](#licensing)

---

## Getting Involved

- **Discord:** Join the [Corridor Creates Discord](https://discord.gg/zvwUrdWXJm) to discuss ideas, share results, and coordinate work
- **Issues:** File bugs and feature requests on the [GitHub Issues](https://github.com/nikopueringer/CorridorKey/issues) page
- **Pull Requests:** Fork the repo, make changes, and submit a PR

---

## Contributing Without Code

Some of the most valuable contributions don't involve writing a single line of Python:

- **Bug reports:** Ran into a problem? File a [GitHub Issue](https://github.com/nikopueringer/CorridorKey/issues) with details about what happened, what you expected, and screenshots if possible.
- **Share results:** Post your before/after comparisons on the Discord. Real-world examples help everyone understand what CorridorKey handles well and where it struggles.
- **Workflow guides:** Created a Nuke template, After Effects preset, or Resolve workflow for using CorridorKey output? Share it!
- **Documentation:** Found something confusing in the docs? Typos? Missing steps? Open an issue or PR.
- **Test edge cases:** Try CorridorKey on unusual footage — heavy motion blur, extreme backlighting, translucent fabrics, rain/smoke, colored screens — and report what works and what doesn't.

---

## Development Setup

1. **Fork and clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/CorridorKey.git
   cd CorridorKey
   ```

2. **Install dependencies with uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh   # Install uv (if not already installed)
   uv sync                                             # Install Python + all dependencies
   ```

3. **Download the model checkpoint** (see [Getting Started](GETTING_STARTED.md#verifying-your-installation))

4. **Verify your setup:**
   ```bash
   uv run python -c "from CorridorKeyModule import CorridorKeyEngine; print('OK')"
   uv run python test_vram.py  # If you have a GPU
   ```

---

## Project Conventions

### Code Style

- **Python 3.11+** — use modern syntax (type hints, f-strings, walrus operator where clear)
- **No strict formatter enforced** — match the style of surrounding code
- **Imports:** stdlib → third-party → local, separated by blank lines
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
- **Constants:** `UPPER_SNAKE_CASE` (see `clip_manager.py` for examples)

### Color Math

This is the most critical area. All color operations must:
- Use the **piecewise sRGB transfer function** from `color_utils.py` — never `x ** 2.2`
- Clearly document whether values are in **sRGB** or **Linear** space
- Clearly document whether foreground is **straight** or **premultiplied**
- Operate in the correct color space for the operation (compositing in linear, despill in sRGB)

### File Organization

```
CorridorKey/
├── clip_manager.py              # CLI wizard — user-facing orchestration
├── CorridorKeyModule/           # Core engine — inference only
│   ├── inference_engine.py      # Public API
│   └── core/                    # Architecture + math
├── gvm_core/                    # GVM module — self-contained
├── VideoMaMaInferenceModule/    # VideoMaMa module — self-contained
└── docs/                        # Documentation
```

Each module should remain self-contained with its own `__init__.py` and `README.md`. Dependencies are centrally managed in the root `pyproject.toml`.

### Performance Considerations

This processes video frame-by-frame at high resolution. Avoid:
- Unnecessary `.cpu()` / `.numpy()` transfers in the inference loop
- Extra memory copies or allocations per frame
- Python-level loops over pixels (use vectorized NumPy/PyTorch operations)

---

## Submitting Changes

### For Bug Fixes

1. Create a branch: `git checkout -b fix/description-of-fix`
2. Make your changes
3. Verify with `test_vram.py` if engine changes were made
4. Test manually with a green screen clip
5. Submit a PR with:
   - What the bug was
   - How you fixed it
   - How you tested it

### For Features

1. **Discuss first** — open an issue or Discord thread to align on approach
2. Create a branch: `git checkout -b feature/description`
3. Implement the feature
4. Update relevant documentation in `docs/`
5. Submit a PR with:
   - What the feature does
   - Design decisions made
   - Testing performed

### For Documentation

Documentation improvements are always welcome! No discussion needed — just submit a PR.

---

## Code Review Checklist

When reviewing PRs (or self-reviewing before submitting), check:

- [ ] **Color spaces are correct** — all operations happen in the documented color space
- [ ] **sRGB functions are piecewise** — no `** 2.2` approximations
- [ ] **Alpha is treated as linear** — no gamma applied to alpha channels
- [ ] **EXR output is linear premultiplied** — the VFX standard
- [ ] **No hardcoded paths** — use relative paths or `BASE_DIR`
- [ ] **No model weights committed** — `.pth`, `.pt`, `.safetensors` are in `.gitignore`
- [ ] **VRAM usage hasn't regressed** — run `test_vram.py` if engine changed
- [ ] **Existing API not broken** — `process_frame()` signature and return dict unchanged
- [ ] **Module independence preserved** — gvm_core and VideoMaMa remain self-contained

---

## Licensing

By contributing to CorridorKey, you agree that your contributions will be licensed under the same terms as the project:

- **CorridorKey core:** Custom CC BY-NC-SA 4.0 variant (see [README.md](../README.md))
- **gvm_core:** CC BY-NC-SA 4.0 ([LICENSE.md](../gvm_core/LICENSE.md))
- **VideoMaMaInferenceModule:** CC BY-NC 4.0 + Stability AI Community License ([LICENSE.md](../VideoMaMaInferenceModule/LICENSE.md))

Key constraints:
- Commercial *use* of CorridorKey for processing images is allowed
- Repackaging and selling CorridorKey is **not** allowed
- Forks and improvements must remain free and open source
- The "Corridor Key" name must be retained in forks
- Paid API inference services using this model are **not** allowed without agreement from Corridor Digital (contact@corridordigital.com)

The GVM and VideoMaMa modules are strictly non-commercial. Using those modules for commercial purposes is prohibited by their respective licenses.
