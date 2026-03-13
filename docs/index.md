# CorridorKey

> 📖 **[GitHub Repository](https://github.com/nikopueringer/CorridorKey)** — Source code, issues, and releases.

When you film something against a green screen, the edges of your subject
inevitably blend with the green background — creating pixels that mix your
subject's true color with the screen. Traditional keyers struggle to untangle
these colors, and even modern "AI Roto" solutions typically output a harsh
binary mask, destroying the delicate semi-transparent pixels needed for a
realistic composite.

CorridorKey solves this *unmixing* problem. You input a raw green screen frame,
and the neural network completely separates the foreground object from the green
screen. For every single pixel — even highly transparent ones like motion blur
or out-of-focus edges — the model predicts the true, un-multiplied straight
color of the foreground element alongside a clean, linear alpha channel.

No more fighting with garbage mattes or agonizing over "core" vs "edge" keys.
Give CorridorKey a hint of what you want, and it separates the light for you.

## Features

- **Physically Accurate Unmixing** — Clean extraction of straight color
  foreground and linear alpha channels, preserving hair, motion blur, and
  translucency.
- **Resolution Independent** — The engine dynamically scales inference to
  handle 4K plates while predicting using its native 2048×2048 high-fidelity
  backbone.
- **VFX Standard Outputs** — Natively reads and writes 16-bit and 32-bit
  Linear float EXR files, preserving true color math for integration in Nuke,
  Fusion, or Resolve.
- **Auto-Cleanup** — Includes a morphological cleanup system to automatically
  prune any tracking markers or tiny background features that slip through
  detection.

## Get Started

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Installation, hardware requirements, usage instructions, and device
    configuration.

    [:octicons-arrow-right-24: Installation](installation.md)
    · [Hardware Requirements](hardware-requirements.md)
    · [Usage](usage.md)
    · [Device & Backend Selection](device-and-backend-selection.md)

-   :material-code-braces:{ .lg .middle } **Developer Guide**

    ---

    Architecture overview, contribution guidelines, and the LLM handover
    document for AI-assisted development.

    [:octicons-arrow-right-24: Architecture](architecture.md)
    · [Contributing](contributing.md)
    · [LLM Handover](LLM_HANDOVER.md)

</div>
