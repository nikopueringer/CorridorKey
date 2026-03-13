# Hardware Requirements

CorridorKey was designed and built on a Linux workstation equipped with an
NVIDIA RTX Pro 6000 (96 GB VRAM). The community is actively optimising it for
consumer GPUs — the most recent build should work on cards with **6–8 GB of
VRAM**, and it can run on most Mac systems with unified memory.

## Core Engine (CorridorKey)

| Spec | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 8 GB+ |
| Compute | CUDA, MPS, or CPU | CUDA (NVIDIA) |
| System RAM | 8 GB | 16 GB+ |

The engine dynamically scales inference to its native 2048×2048 backbone, so
more VRAM allows larger plates to be processed without tiling.

!!! warning "Windows CUDA driver requirement"
    To run GPU acceleration natively on Windows, your system **must** have
    NVIDIA drivers that support **CUDA 12.8 or higher**. If your drivers only
    support older CUDA versions, the installer will likely fall back to the CPU.

## Optional Modules

GVM and VideoMaMa are optional Alpha Hint generators with significantly higher
hardware requirements. You do **not** need them — you can always provide your
own Alpha Hints from other software.

--8<-- "docs/_snippets/optional-weights.md"

| Module | VRAM Required | Notes |
|---|---|---|
| **GVM** | ~80 GB | Uses massive Stable Video Diffusion models. |
| **VideoMaMa** | 80 GB+ (native) / <24 GB (community optimised) | Community tweaks reduce VRAM, but extreme optimisations are not yet fully integrated in this repo. |

## Apple Silicon

--8<-- "docs/_snippets/apple-silicon-note.md"
