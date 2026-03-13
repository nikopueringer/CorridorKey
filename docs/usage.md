# Usage

## How It Works

CorridorKey requires two inputs to process a frame:

1. **The Original RGB Image** — your green screen footage. This requires the
   sRGB colour gamut (interchangeable with Rec. 709), and the engine can ingest
   either an sRGB gamma or Linear gamma curve.
2. **A Coarse Alpha Hint** — a rough black-and-white mask that generally
   isolates the subject. It does *not* need to be precise; a rough chroma key
   or AI roto is enough.

!!! tip "Better hints → better results"
    The model was trained on coarse, blurry, eroded masks and is exceptional at
    filling in details from the hint. However, it is generally less effective at
    *subtracting* unwanted mask details if your Alpha Hint is expanded too far.
    Experiment with different amounts of mask erosion or feathering.

### Alpha Hint Generators

Two optional modules are bundled for automatic hint generation inside
`clip_manager.py`:

| Generator | Input Required | Strengths |
|---|---|---|
| **GVM** | None — fully automatic | Excellent for people; can struggle with inanimate objects. |
| **VideoMaMa** | A rough `VideoMamaMaskHint` (hand-drawn or AI-generated) | Spectacular results with finer control via the mask hint. |

If you choose VideoMaMa, place your mask hint in the `VideoMamaMaskHint/`
folder that the wizard creates for your shot.

!!! note "Show these projects some love"
    GVM and VideoMaMa are open-source research projects. Please star their
    repos: [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa) and
    [GVM](https://github.com/aim-uofa/GVM).

--8<-- "docs/_snippets/model-download.md"

## The Command Line Wizard

For the easiest experience, use the provided launcher scripts. They start a
prompt-based configuration wizard in your terminal.

=== "Windows"

    Drag-and-drop a video file or folder onto
    `CorridorKey_DRAG_CLIPS_HERE_local.bat`.

    !!! note
        Only launch via drag-and-drop or from CMD. Double-clicking the `.bat`
        directly will throw an error.

=== "Linux / Mac"

    Run or drag-and-drop a video file or folder onto
    `./CorridorKey_DRAG_CLIPS_HERE_local.sh`.


### Workflow Steps

1. **Launch** — Drag-and-drop a single loose video file (e.g. `.mp4`), a shot
   folder containing image sequences, or a master "batch" folder with multiple
   shots onto the launcher script.

2. **Organisation** — The wizard detects what you dragged in. If you dropped
   loose video files or unorganised folders, it will ask whether to organise
   them into the proper structure:

    ```
    YourShot/
    ├── Input/                  # original green screen frames
    ├── AlphaHint/              # coarse alpha masks
    └── VideoMamaMaskHint/      # (optional) mask hints for VideoMaMa
    ```

    This structure is required for the engine to pair your hints and footage
    correctly.

3. **Generate Hints (Optional)** — If the wizard detects missing `AlphaHint`
   frames, it offers to generate them automatically using GVM or VideoMaMa.

4. **Configure** — Once clips have both Inputs and AlphaHints, select
   *Process Ready Clips*. The wizard prompts you to configure the run:

    | Option | Description |
    |---|---|
    | **Gamma Space** | Tell the engine whether your sequence uses a **Linear** or **sRGB** gamma curve. |
    | **Despill Strength** | Traditional despill filter (0–10). Set to 0 to handle despill in your comp later. |
    | **Auto-Despeckle** | Toggle automatic cleanup and define the size threshold. Removes tracking dots and any small disconnected pixel islands. |
    | **Refiner Strength** | Use the default (`1.0`) unless experimenting with extreme detail pushing. |

5. **Result** — The engine generates output folders inside your shot directory
   (see below).

## Folder Structure

### Input Folders

| Folder | Purpose |
|---|---|
| `Input/` | Original green screen frames (sRGB gamut, sRGB or Linear gamma). |
| `AlphaHint/` | Coarse alpha masks — one per frame, matching filenames. |
| `VideoMamaMaskHint/` | *(Optional)* Rough mask hints for the VideoMaMa generator. |

### Output Folders

| Folder | Format | Colour Space | Description |
|---|---|---|---|
| `Matte/` | 32-bit EXR | Linear | Raw linear alpha channel. |
| `FG/` | 32-bit EXR | sRGB gamut, sRGB gamma | Raw straight foreground colour. Convert to linear gamma before combining with the alpha in your compositing program. |
| `Processed/` | 32-bit EXR | Linear (premultiplied) | RGBA image — linear foreground premultiplied against linear alpha. Drop straight into Premiere / Resolve for a quick preview without complex premultiplication routing. |
| `Comp/` | PNG | sRGB | Preview of the key composited over a checkerboard. |

!!! info "Working with raw outputs"
    The `Processed/` pass is a convenience for quick previews. For maximum
    control in Nuke, Fusion, or Resolve, work with the separate `FG/` and
    `Matte/` outputs and handle premultiplication yourself.
