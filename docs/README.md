# CorridorKey Documentation

Welcome! This documentation is organized into two paths depending on your background. Pick the one that fits you best — both are complete and will get you where you need to go.

---

## I'm a VFX Artist / Editor

You use tools like After Effects, DaVinci Resolve, Nuke, or Premiere and want to use CorridorKey to pull keys from your green screen footage.

| Document | What It Covers |
|---|---|
| **[Artist Quickstart](QUICKSTART_ARTISTS.md)** | Install, prepare footage, run your first key — no coding required |
| [Color Pipeline](COLOR_PIPELINE.md) | How CorridorKey's output files work and how to use them in your compositor |
| [Troubleshooting](TROUBLESHOOTING.md) | Common problems and fixes |

**Start here:** [Artist Quickstart](QUICKSTART_ARTISTS.md)

---

## I'm a Developer / ML Engineer

You're comfortable with Python, virtual environments, and want to integrate CorridorKey into scripts, pipelines, or contribute code.

| Document | What It Covers |
|---|---|
| **[Getting Started (Developer)](GETTING_STARTED.md)** | Installation, environment setup, model downloads, CLI usage |
| [Python API Reference](API_REFERENCE.md) | Full API docs for CorridorKeyEngine, GVMProcessor, VideoMaMa |
| [Architecture Deep Dive](ARCHITECTURE.md) | GreenFormer model, Hiera backbone, CNN Refiner, data flow |
| [Color Pipeline](COLOR_PIPELINE.md) | sRGB/Linear math, premultiplication, output specs |
| [Troubleshooting](TROUBLESHOOTING.md) | Common problems and fixes |
| [Contributing](CONTRIBUTING.md) | How to contribute code, optimizations, or features |

**Start here:** [Getting Started (Developer)](GETTING_STARTED.md)

---

## For AI Assistants

| Document | What It Covers |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Project-level AI assistant guide (loaded automatically) |
| [LLM Handover Guide](LLM_HANDOVER.md) | Detailed technical reference for AI coding assistants |

---

## All Documents

- [Artist Quickstart](QUICKSTART_ARTISTS.md) — zero to first key, no coding
- [Getting Started (Developer)](GETTING_STARTED.md) — full setup for developers
- [Python API Reference](API_REFERENCE.md) — CorridorKeyEngine, GVMProcessor, VideoMaMa APIs
- [Architecture Deep Dive](ARCHITECTURE.md) — model architecture, inference pipeline, module design
- [Color Pipeline & Compositing Math](COLOR_PIPELINE.md) — color science, gamma, premultiplication
- [Troubleshooting](TROUBLESHOOTING.md) — common issues and solutions
- [Contributing](CONTRIBUTING.md) — how to help improve CorridorKey
- [LLM Handover Guide](LLM_HANDOVER.md) — technical reference for AI assistants
