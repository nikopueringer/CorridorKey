# Getting Started (Developer Guide)

> **Are you a VFX artist** who primarily uses After Effects, Resolve, Nuke, or similar tools? Check out the [Artist Quickstart](QUICKSTART_ARTISTS.md) instead — it covers the same setup without requiring coding experience.

This guide supplements the [main README](../README.md) with developer-specific details: prerequisites, supported formats, folder structure, and compositor integration.

---

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA with 24GB VRAM (RTX 3090, 4090, 5090) | 24GB+ dedicated GPU not driving displays |
| RAM | 16 GB | 32 GB+ |
| Storage | 2 GB (code + weights) | 100 GB+ (with GVM weights) |
| OS | Windows 10/11, Linux (Ubuntu 20.04+), macOS | Linux workstation |

**VRAM notes:**
- CorridorKey core: **~22.7 GB** at 2048x2048
- GVM (optional): **~80 GB** — typically cloud-only
- VideoMaMa (optional): **~24 GB+**
- Running on a GPU that also drives your displays increases OOM risk

### Software

- **[uv](https://docs.astral.sh/uv/)** — handles Python installation, virtual environments, and all dependencies
- **CUDA** compatible with your GPU and PyTorch version
- **Git** for cloning the repository

> **Note:** You do not need to install Python separately. uv automatically downloads and manages the correct Python version (3.11+).

---

## Verifying Your Installation

After following the installation steps in the [README](../README.md#1-installation):

```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
uv run python -c "from CorridorKeyModule import CorridorKeyEngine; print('CorridorKey module OK')"
```

You should see PyTorch version, CUDA availability, and no import errors.

---

## Supported Input Formats

| Format | Type | Notes |
|---|---|---|
| PNG | Image sequence | 8-bit sRGB (most common) |
| EXR | Image sequence | 16/32-bit Linear float (VFX standard) |
| TIFF | Image sequence | 8/16-bit |
| JPG | Image sequence | 8-bit sRGB (lossy, not recommended for VFX) |
| MP4 | Video file | H.264/H.265 |
| MOV | Video file | ProRes, H.264 |
| AVI / MKV | Video file | Various codecs |

---

## Folder Structure

CorridorKey expects footage organized into shot folders. The wizard can organize things for you, or you can set it up manually:

```
MyProject/
├── Shot_001/
│   ├── Input/                  # Your green screen frames
│   │   ├── frame_0001.exr
│   │   ├── frame_0002.exr
│   │   └── ...
│   ├── AlphaHint/              # Coarse alpha masks (generated or manual)
│   │   ├── frame_0001.png
│   │   ├── frame_0002.png
│   │   └── ...
│   └── VideoMamaMaskHint/      # Binary mask for VideoMaMa (optional)
│       ├── frame_0001.png
│       └── ...
├── Shot_002/
│   ├── Input.mp4               # Video file also works
│   └── AlphaHint/
│       └── ...
└── ...
```

You can also drag a loose video file onto the launcher script and the wizard will organize it for you.

**Important:** Input and AlphaHint must have the **same number of frames**.

---

## Using Output in Your Compositing Software

After inference, you'll find four output passes. See the [Color Pipeline](COLOR_PIPELINE.md#reference-output-pass-specifications) for full specs.

### DaVinci Resolve / Fusion
1. Import `Processed/*.exr` for a quick drop-in
2. Or import `FG/*.exr` + `Matte/*.exr` separately for more control
3. When using separate passes: apply sRGB → Linear conversion to the FG pass before combining with the matte

### Nuke
1. Read the `FG/*.exr` and `Matte/*.exr` sequences
2. Apply a Colorspace node (sRGB → Linear) to the FG
3. Use a Copy node to set the alpha channel from the Matte
4. Premultiply before compositing over your background plate

### After Effects
1. Import the `Processed/*.exr` sequence
2. Set the interpretation to Linear Light (32bpc project)
3. The premultiplied alpha will work automatically with AE's compositing

---

## Next Steps

- [Python API Reference](API_REFERENCE.md) — integrate CorridorKey into custom scripts
- [Color Pipeline](COLOR_PIPELINE.md) — understand the math behind the output
- [Architecture Deep Dive](ARCHITECTURE.md) — model architecture and inference pipeline
- [Troubleshooting](TROUBLESHOOTING.md) — common issues and solutions
