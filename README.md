# CorridorKey


https://github.com/user-attachments/assets/1fb27ea8-bc91-4ebc-818f-5a3b5585af08


When you film something against a green screen, the edges of your subject inevitably blend with the green background. This creates pixels that are a mix of your subject's color and the green screen's color. Traditional keyers struggle to untangle these colors, forcing you to spend hours building complex edge mattes or manually rotoscoping. Even modern "AI Roto" solutions typically output a harsh binary mask, completely destroying the delicate, semi-transparent pixels needed for a realistic composite.

I built CorridorKey to solve this *unmixing* problem. 

You input a raw green screen frame, and the neural network completely separates the foreground object from the green screen. For every single pixel, even the highly transparent ones like motion blur or out-of-focus edges, the model predicts the true, un-multiplied straight color of the foreground element, alongside a clean, linear alpha channel. It doesn't just guess what is opaque and what is transparent; it actively reconstructs the color of the foreground object as if the green screen was never there.

No more fighting with garbage mattes or agonizing over "core" vs "edge" keys. Give CorridorKey a hint of what you want, and it separates the light for you.

## Alert!

This is a brand new release, I'm sure you will discover many ways it can be improved! I invite everyone to help. Join us on the "Corridor Creates" Discord to share ideas, work, forks, etc! https://discord.gg/zvwUrdWXJm

Also, if you are a novice at using python scripts much like I was, consider downloading a smart IDE like Antigravity (from google, it's free), downloading this repository, and then asking Antigravity to help you get up and running. I even made a LLM Handover doc in the docs/ directory. This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies — it handles Python installation, virtual environments, and packages all in one step, so you don't need to worry about any of that.

Naturally, I have not tested everything. If you encounter errors, please consider patching the code as needed and submitting a pull request.

## Features

*   **Physically Accurate Unmixing:** Clean extraction of straight color foreground and linear alpha channels, preserving hair, motion blur, and translucency.
*   **Resolution Independent:** The engine dynamically scales inference to handle 4K plates while predicting using its native 2048x2048 high-fidelity backbone.
*   **VFX Standard Outputs:** Natively reads and writes 16-bit and 32-bit Linear float EXR files, preserving true color math for integration in Nuke, Fusion, or Resolve.
*   **Auto-Cleanup:** Includes a morphological cleanup system to automatically prune any tracking markers or tiny background features that slip through CorridorKey's detection.

## Hardware Requirements

This project was designed and built on a Linux workstation (Puget Systems PC) equipped with an NVIDIA RTX Pro 6000 with 96GB of VRAM. This project is not yet optimized for sub 24 gig VRAM systems, but with the help of the community, maybe we can make that happen.

*   **CorridorKey:** Running inference natively at 2048x2048 requires approximately **22.7 GB of VRAM**. You will need at least a 24GB GPU (such as a 3090, 4090, 5090, etc). It is highly recommended to run this on a secondary GPU that is not driving your OS/displays, or on a rented cloud instance (like Runpod or Google Colab) to avoid Out-Of-Memory errors.
    *   **Windows Users:** To run GPU acceleration natively on Windows, your system MUST have NVIDIA drivers that support **CUDA 12.6 or higher** installed. If your drivers only support older CUDA versions, the installer will likely fallback to the CPU.
*   **GVM (Optional):** Requires approximately **80 GB of VRAM** and utilizes massive Stable Video Diffusion models.
*   **VideoMaMa (Optional):** Natively requires a massive chunk of VRAM as well (originally 80GB+). While the community has tweaked the architecture to run at less than 24GB, those extreme memory optimizations have not yet been fully implemented in this repository.

Because GVM and VideoMaMa have huge model file sizes and extreme hardware requirements, installing their modules is completely optional. You can always provide your own Alpha Hints generated from other, lighter software.

## Getting Started

### 1. Installation

This project uses **[uv](https://docs.astral.sh/uv/)** to manage Python and all dependencies. uv is a fast, modern replacement for pip that automatically handles Python versions, virtual environments, and package installation in a single step. You do **not** need to install Python yourself — uv does it for you.

**For Windows Users (Automated):**
1.  Clone or download this repository to your local machine.
2.  Double-click `Install_CorridorKey_Windows.bat`. This will automatically install uv (if needed), set up your Python environment, install all dependencies, and download the CorridorKey model.
    > **Note:** If this is the first time installing uv, any terminal windows you already had open won't see it. The installer script handles the current window automatically, but if you open a new terminal and get "'uv' is not recognized", just close and reopen that terminal.
3.  (Optional) Double-click `Install_GVM_Windows.bat` and `Install_VideoMaMa_Windows.bat` to download the heavy optional Alpha Hint generator weights.

**For Linux / Mac Users:**
1.  Clone or download this repository to your local machine.
2.  Install uv if you don't have it:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3.  Install all dependencies (uv will download Python 3.10+ automatically if needed):
    ```bash
    uv sync
    ```
4.  **Download the Models:** You must manually download these open-source foundational models and place them in their exact respective folders:
    *   **CorridorKey v1.0 Model (~300MB):** [Download CorridorKey_v1.0.pth](https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth) 
        *   Place inside: `CorridorKeyModule/checkpoints/` and ensure it is named exactly `CorridorKey.pth`.
    *   **GVM Weights (Optional):** [HuggingFace: geyongtao/gvm](https://huggingface.co/geyongtao/gvm)
        *   Download using the CLI: `uv run hf download geyongtao/gvm --local-dir gvm_core/weights`
    *   **VideoMaMa Weights (Optional):** [HuggingFace: SammyLim/VideoMaMa](https://huggingface.co/SammyLim/VideoMaMa)
        *   Download the VideoMaMa fine-tuned weights:
            ```
            uv run hf download SammyLim/VideoMaMa --local-dir VideoMaMaInferenceModule/checkpoints/VideoMaMa
            ```
        *   VideoMaMa also requires the Stable Video Diffusion base model (VAE + image encoder only, ~2.5GB). Accept the license at [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt), then:
            ```
            uv run hf download stabilityai/stable-video-diffusion-img2vid-xt \
              --local-dir VideoMaMaInferenceModule/checkpoints/stable-video-diffusion-img2vid-xt \
              --include "feature_extractor/*" "image_encoder/*" "vae/*" "model_index.json"
            ```

### 2. How it Works

CorridorKey requires two inputs to process a frame:
1.  **The Original RGB Image:** The to-be-processed green screen footage. This requires the sRGB color gamut (interchangeable with REC709 gamut), and the engine can ingest either an sRGB gamma or Linear gamma curve. 
2.  **A Coarse Alpha Hint:** A rough black-and-white mask that generally isolates the subject. This does *not* need to be precise. It can be generated by you with a rough chroma key or AI roto.

I've had the best results using GVM or VideoMaMa to create the AlphaHint, so I've repackaged those projects and integrated them here as optional modules inside `clip_manager.py`. Here is how they compare:

*   **GVM:** Completely automatic and requires no additional input. It works exceptionally well for people, but can struggle with inanimate objects.
*   **VideoMaMa:** Requires you to provide a rough VideoMamaMaskHint (often drawn by hand or AI) telling it what you want to key. If you choose to use this, place your mask hint in the `VideoMamaMaskHint/` folder that the wizard creates for your shot. VideoMaMa results are spectacular and can be controlled more easily than GVM due to this mask hint. 

Perhaps in the future, I will implement other generators for the AlphaHint! In the meantime, the better your Alpha Hint, the better CorridorKey's final result will be. Experiment with different amounts of mask erosion or feathering. The model was trained on coarse, blurry, eroded masks, and is exceptional at filling in details from the hint. However, it is generally less effective at subtracting unwanted mask details if your Alpha Hint is expanded too far. 

Please give feedback and share your results!

### 3. Usage: The Command Line Wizard

For the easiest experience, use the provided launcher scripts. These scripts launch a prompt-based configuration wizard in your terminal.

*   **Windows:** Drag-and-drop a video file or folder onto `CorridorKey_DRAG_CLIPS_HERE_local.bat` (Note: Only launch via Drag-and-Drop or CMD. Double-clicking the `.bat` directly will throw an error).
*   **Linux / Mac:** Run or drag-and-drop a video file or folder onto `./CorridorKey_DRAG_CLIPS_HERE_local.sh`

**Workflow Steps:**
1.  **Launch:** You can drag-and-drop a single loose video file (like an `.mp4`), a shot folder containing image sequences, or even a master "batch" folder containing multiple different shots all at once onto the launcher script.
2.  **Organization:** The wizard will detect what you dragged in. If you dropped loose video files or unorganized folders, the first prompt will ask if you want it to organize your clips into the proper structure. 
    *   If you say Yes, the script will automatically create a shot folder, move your footage into an `Input/` sub-folder, and generate empty `AlphaHint/` and `VideoMamaMaskHint/` folders for you. This structure is required for the engine to pair your hints and footage correctly!
3.  **Generate Hints (Optional):** If the wizard detects your shots are missing an `AlphaHint`, it will ask if you want to generate them automatically using the repackaged GVM or VideoMaMa modules.
4.  **Configure:** Once your clips have both Inputs and AlphaHints, select "Process Ready Clips". The wizard will prompt you to configure the run:
    *   **Gamma Space:** Tell the engine if your sequence uses a Linear or sRGB gamma curve.
    *   **Despill Strength:** This is a traditional despill filter (0-10), if you wish to have it baked into the output now as opposed to applying it in your comp later.
    *   **Auto-Despeckle:** Toggle automatic cleanup and define the size threshold. This isn't just for tracking dots, it removes any small, disconnected islands of pixels.
    *   **Refiner Strength:** Use the default (1.0) unless you are experimenting with extreme detail pushing.
5.  **Result:** The engine will generate several folders inside your shot directory:
    *   `/Matte`: The raw Linear Alpha channel (EXR).
    *   `/FG`: The raw Straight Foreground Color Object. (Note: The engine natively computes this in the sRGB gamut. You must manually convert this pass to linear gamma before being combined with the alpha in your compositing program).
    *   `/Processed`: An RGBA image containing the Linear Foreground premultiplied against the Linear Alpha (EXR). This pass exists so you can immediately drop the footage into Premiere/Resolve for a quick preview without dealing with complex premultiplication routing. However, if you want more control over your image, working with the raw FG and Matte outputs will give you that.
    *   `/Comp`: A simple preview of the key composited over a checkerboard (PNG).

## But What About Training and Datasets?

If enough people find this project interesting I'll get the training program and datasets uploaded so we can all really go to town making the absolute best keyer fine tunes! Just hit me with some messages on the Corridor Creates discord or here. If enough people lock in, I'll get this stuff packaged up. Hardware requirements are beefy and the gigabytes are plentiful so I don't want to commit the time unless there's demand.

## Device Selection

By default, CorridorKey auto-detects the best available compute device: **CUDA > MPS > CPU**.

**Override via CLI flag:**
```bash
uv run python clip_manager.py --action wizard --path "/mnt/ssd-storage/..." --device mps
uv run python clip_manager.py --action run_inference --device cpu
```

**Override via environment variable:**
```bash
export CORRIDORKEY_DEVICE=cpu
uv run python clip_manager.py --action wizard --path "/mnt/ssd-storage/..."
```

Priority: `--device` flag > `CORRIDORKEY_DEVICE` env var > auto-detect.

## VRAM Optimization

If your GPU is running out of memory, lower the inference resolution and keep mixed precision enabled:

```bash
uv run python clip_manager.py --action run_inference --img-size 1536 --precision auto
```

Tips:
- `--img-size 2048` = highest quality, highest VRAM use (default)
- `--img-size 1536` = significantly lower VRAM, strong quality
- `--img-size 1024` = lowest VRAM, lower edge fidelity
- `--precision auto` uses fp16 on CUDA (saves VRAM) and fp32 elsewhere

**Mac users (Apple Silicon):** MPS support is experimental in PyTorch. If you encounter operator errors, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU for unsupported ops:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```



### Linux Path Compatibility

`clip_manager.py` now accepts a cross-platform `--path` flag (with `--win_path` kept as a deprecated alias), and path mapping can be customized on Linux with environment variables:

```bash
export CORRIDORKEY_WIN_DRIVE_ROOT="V:\\"
export CORRIDORKEY_LINUX_MOUNT_ROOT="/mnt/ssd-storage"
```

This lets you adapt studio mount points without editing source code.

## Backend Selection

CorridorKey supports two inference backends:
- **Torch** (default on Linux/Windows) — CUDA, MPS, or CPU
- **MLX** (Apple Silicon) — native Metal acceleration, no Torch overhead

Resolution: `--backend` flag > `CORRIDORKEY_BACKEND` env var > auto-detect.
Auto mode prefers MLX on Apple Silicon when available.

### MLX Setup (Apple Silicon)

1. Install the MLX backend:
   ```bash
   uv pip install corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git
   ```
2. Place converted weights in `CorridorKeyModule/checkpoints/`:
   ```
   CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors
   ```
3. Run with auto-detection or explicit backend:
   ```bash
   CORRIDORKEY_BACKEND=mlx uv run python clip_manager.py --action run_inference
   ```

MLX uses img_size=2048 by default (same as Torch).

### Troubleshooting
- **"No .safetensors checkpoint found"** — place MLX weights in `CorridorKeyModule/checkpoints/`
- **"corridorkey_mlx not installed"** — run `uv pip install corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git`
- **"MLX requires Apple Silicon"** — MLX only works on M1+ Macs
- **Auto picked Torch unexpectedly** — set `CORRIDORKEY_BACKEND=mlx` explicitly

## Advanced Usage

For developers looking for more details on the specifics of what is happening in the CorridorKey engine, check out the README in the `/CorridorKeyModule` folder. We also have a dedicated handover document outlining the pipeline architecture for AI assistants in `/docs/LLM_HANDOVER.md`.

### Running Tests

The project includes unit tests for the color math and compositing pipeline. No GPU or model weights required — tests run in a few seconds on any machine.

```bash
uv sync --group dev   # install test dependencies (pytest)
uv run pytest          # run all tests
uv run pytest -v       # verbose output (shows each test name)
```

## CorridorKey Licensing and Permissions

Use this tool for whatever you'd like, including for processing images as part of a commercial project! You MAY NOT repackage this tool and sell it, and any variations or improvements of this tool that are released must remain under the same license, and must include the name Corridor Key.

You MAY NOT offer inference with this model as a paid API service. If you run a commercial software package or inference service and wish to incoporate this tool into your software, shoot us an email to work out an agreement! I promise we're easy to work with. contact@corridordigital.com. Outside of the stipulations listed above, this license is effectively a variation of [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Please keep the Corridor Key name in any future forks or releases!

## Acknowledgements and Licensing

CorridorKey integrates several open-source modules for Alpha Hint generation. We would like to explicitly credit and thank the following research teams:

*   **Generative Video Matting (GVM):** Developed by the Advanced Intelligent Machines (AIM) research team at Zhejiang University. The GVM code and models are heavily utilized in the `gvm_core` module. Their work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). You can find their source repository here: [aim-uofa/GVM](https://github.com/aim-uofa/GVM).
*   **VideoMaMa:** Developed by the CVLAB at KAIST. The VideoMaMa architecture is utilized within the `VideoMaMaInferenceModule`. Their code is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/), and their specific foundation model checkpoints (`dino_projection_mlp.pth`, `unet/*`) are subject to the [Stability AI Community License](https://stability.ai/license). You can find their source repository here: [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa).

By using these optional modules, you agree to abide by their respective Non-Commercial licenses. Please review their repositories for full terms.
