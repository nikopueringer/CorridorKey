# CorridorKey Project Context

Read @/AGENTS.md for full project instructions covering architecture, critical dataflow rules, dev workflow, code conventions, landmines, and hardware requirements.

Key points:

- CorridorKey is an AI green screen keyer for professional VFX pipelines
- Model input is sRGB float [0, 1], output FG is sRGB straight, alpha is linear
- EXR output is linear float premultiplied — always use piecewise sRGB transfer, never gamma 2.2 approximation
- Use `uv sync --group dev`, `uv run pytest`, `uv run ruff check` for development
- Never modify `gvm_core/` or `VideoMaMaInferenceModule/` (upstream-derived research code)
