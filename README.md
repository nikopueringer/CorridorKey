# CorridorKey


https://github.com/user-attachments/assets/1fb27ea8-bc91-4ebc-818f-5a3b5585af08


When you film something against a green screen, the edges of your subject inevitably blend with the green background. This creates pixels that are a mix of your subject's color and the green screen's color. Traditional keyers struggle to untangle these colors, forcing you to spend hours building complex edge mattes or manually rotoscoping. Even modern "AI Roto" solutions typically output a harsh binary mask, completely destroying the delicate, semi-transparent pixels needed for a realistic composite.

I built CorridorKey to solve this *unmixing* problem. 

You input a raw green screen frame, and the neural network completely separates the foreground object from the green screen. For every single pixel, even the highly transparent ones like motion blur or out-of-focus edges, the model predicts the true, un-multiplied straight color of the foreground element, alongside a clean, linear alpha channel. It doesn't just guess what is opaque and what is transparent; it actively reconstructs the color of the foreground object as if the green screen was never there.

No more fighting with garbage mattes or agonizing over "core" vs "edge" keys. Give CorridorKey a hint of what you want, and it separates the light for you.

## Alert!

This is a brand new release, I'm sure you will discover many ways it can be improved! I invite everyone to help. Join us on the "Corridor Creates" Discord to share ideas, work, forks, etc! https://discord.gg/zvwUrdWXJm

If you want an easy-install, artist-friendly user interface version of CorridorKey, check out [EZ-CorridorKey](https://github.com/edenaion/EZ-CorridorKey)

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies — it handles Python installation, virtual environments, and packages all in one step, so you don't need to worry about any of that. just run the appropriate install script for your OS.

Naturally, I have not tested everything. If you encounter errors, please consider patching the code as needed and submitting a pull request.

## Features

*   **Physically Accurate Unmixing:** Clean extraction of straight color foreground and linear alpha channels, preserving hair, motion blur, and translucency.
*   **Resolution Independent:** The engine dynamically scales inference to handle 4K plates while predicting using its native 2048x2048 high-fidelity backbone.
*   **VFX Standard Outputs:** Natively reads and writes 16-bit and 32-bit Linear float EXR files, preserving true color math for integration in Nuke, Fusion, or Resolve.
*   **Auto-Cleanup:** Includes a morphological cleanup system to automatically prune any tracking markers or tiny background features that slip through CorridorKey's detection.
*   **WebUI:** Browser-based interface with drag-and-drop upload, one-click full pipeline, real-time progress, video playback, A/B comparison, and project management. Run via `docker compose --profile web up -d` and open `localhost:3000`.

## Hardware Requirements

This project was designed and built on a Linux workstation (Puget Systems PC) equipped with an NVIDIA RTX Pro 6000 with 96GB of VRAM. The community is ACTIVELY optimizing it for consumer GPUS.

The most recent build should work on computers with 6-8 gig of VRAM, and it can run on most M1+ Mac systems with unified memory. Yes, it might even work on your old Macbook pro. Let us know on the Discord!

*   **Windows Users:** To run GPU acceleration natively on Windows, your system MUST have NVIDIA drivers that support **CUDA 12.8 or higher** installed. If your drivers only support older CUDA versions, the installer will likely fallback to the CPU.
*   **GVM (Optional):** Requires approximately **80 GB of VRAM** and utilizes massive Stable Video Diffusion models.
*   **VideoMaMa (Optional):** Natively requires a massive chunk of VRAM as well (originally 80GB+). While the community has tweaked the architecture to run at less than 24GB, those extreme memory optimizations have not yet been fully implemented in this repository.
*   **BiRefNet (Optional):** Lightweight AlphaHint generator option.

Because GVM and VideoMaMa have huge model file sizes and extreme hardware requirements, installing their modules is completely optional. You can always provide your own Alpha Hints generated from your editing program, BiRefNet, or any other method. The better the AlphaHint, the better the result.

## Getting Started

### 1. Installation

This project uses **[uv](https://docs.astral.sh/uv/)** to manage Python and all dependencies. uv is a fast, modern replacement for pip that automatically handles Python versions, virtual environments, and package installation in a single step. You do **not** need to install Python yourself — uv does it for you.

**For Windows Users (Automated):**
1.  Clone or download this repository to your local machine.
2.  Double-click `Install_CorridorKey_Windows.bat`. This will automatically install uv (if needed), set up your Python environment, install all dependencies, and download the CorridorKey model.
    > **Note:** If this is the first time installing uv, any terminal windows you already had open won't see it. The installer script handles the current window automatically, but if you open a new terminal and get "'uv' is not recognized", just close and reopen that terminal.
3.  (Optional) Double-click `Install_GVM_Windows.bat` and `Install_VideoMaMa_Windows.bat` to download the heavy optional Alpha Hint generator weights.

**For Linux / Mac Users (Automated):**
1.  Clone or download this repository to your local machine.
2.  Open terminal and write `bash`. Put a space after writing `bash`.
3.  Drag and drop `Install_CorridorKey_Linux_Mac.sh` into the terminal. Then press enter.
4.  (Optional) Do the 2. step again. But now drag and drop `Install_GVM_Linux_Mac.sh` and `Install_VideoMaMa_Linux_Mac.sh` to download the heavy optional Alpha Hint generator weights.

**For Linux / Mac Users (Manual):**
1.  Clone or download this repository to your local machine.
2.  Install uv if you don't have it:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3.  Install all dependencies (uv will download Python 3.10+ automatically if needed):
    ```bash
    uv sync                  # CPU/MPS (default — works everywhere)
    uv sync --extra cuda     # CUDA GPU acceleration (Linux/Windows)
    uv sync --extra mlx      # Apple Silicon MLX acceleration
    ```
4.  **Download the Models:**
    *   **CorridorKey v1.0 Model (~300MB):** Downloads automatically on first run. If no `.pth` file is found in `CorridorKeyModule/checkpoints/`, the engine fetches it from [CorridorKey's HuggingFace](https://huggingface.co/nikopueringer/CorridorKey_v1.0) and saves it as `CorridorKey.pth`. No manual download needed.
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
        *   VideoMaMa is an amazing project, please go star their [repo](https://github.com/cvlab-kaist/VideoMaMa) and show them some support! 
### 2. How it Works

CorridorKey requires two inputs to process a frame:
1.  **The Original RGB Image:** The to-be-processed green screen footage. This requires the sRGB color gamut (interchangeable with REC709 gamut), and the engine can ingest either an sRGB gamma or Linear gamma curve. 
2.  **A Coarse Alpha Hint:** A rough black-and-white mask that generally isolates the subject. This does *not* need to be precise. It can be generated by you with a rough chroma key or AI roto.

I've had the best results using GVM or VideoMaMa to create the AlphaHint, so I've repackaged those projects and integrated them here as optional modules inside `clip_manager.py`. Here is how they compare:

*   **GVM:** Completely automatic and requires no additional input. It works exceptionally well for people, but can struggle with inanimate objects.
*   **VideoMaMa:** Requires you to provide a rough VideoMamaMaskHint (often drawn by hand or AI) telling it what you want to key. If you choose to use this, place your mask hint in the `VideoMamaMaskHint/` folder that the wizard creates for your shot. VideoMaMa results are spectacular and can be controlled more easily than GVM due to this mask hint.
*   **Please** go show the creators of these projects some love and star their repos. [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa) and [GVM](https://github.com/aim-uofa/GVM)

Perhaps in the future, I will implement other generators for the AlphaHint! In the meantime, the better your Alpha Hint, the better CorridorKey's final result will be. Experiment with different amounts of mask erosion or feathering. The model was trained on coarse, blurry, eroded masks, and is exceptional at filling in details from the hint. However, it is generally less effective at subtracting unwanted mask details if your Alpha Hint is expanded too far. 

Please give feedback and share your results!

### WebUI (Browser-based)

CorridorKey includes a full web interface for managing clips, running inference, and previewing results — no terminal required.

**Quick start with Docker Compose:**
```bash
docker compose --profile web up -d --build   # first run builds the image (~5 min)
# Open http://localhost:3000

# Subsequent runs (no rebuild needed unless code changes):
docker compose --profile web up -d
```

**Quick start without Docker:**
```bash
uv sync --group dev --extra web          # install web dependencies
uv sync --group dev --extra web --extra cuda  # with CUDA GPU support
uv run uvicorn web.api.app:create_app --factory --port 3000
# Open http://localhost:3000
```

**WebUI Features:**
- **Upload & organize** — drag-and-drop videos or zipped frame sequences, organize into projects
- **Full pipeline** — one-click processing: extract frames → generate alpha hints (GVM/VideoMaMa) → run inference
- **Real-time progress** — WebSocket-driven progress bars with ETA and fps counter
- **Frame viewer** — scrub through frames, play as video (ffmpeg-stitched MP4), A/B comparison mode
- **Download outputs** — download any pass (FG, Matte, Comp, Processed) as ZIP
- **Job queue** — parallel CPU jobs (extraction) + GPU jobs with configurable VRAM limits
- **Weight management** — download CorridorKey, GVM, and VideoMaMa weights from HuggingFace directly in Settings
- **VRAM monitoring** — system-wide GPU memory usage (via nvidia-smi)
- **Right-click context menus** — rename projects, move clips, batch process, delete
- **Keyboard shortcuts** — press `?` to see all shortcuts

**Important notes:**
- **Clip storage:** The WebUI manages clips under `Projects/`, while the CLI wizard uses `ClipsForInference/`. These directories are independent — clips created in the WebUI won't appear in the CLI and vice versa. Set `CK_CLIPS_DIR` to point at `ClipsForInference/` if you want both to use the same directory.
- **Mac / MLX:** The WebUI has not been validated on Mac with MLX inference. The server will start and the UI will work, but the VRAM meter will show N/A (nvidia-smi is not available on Mac) and the VRAM concurrency limit for parallel jobs is not enforced on non-CUDA systems.
- Model weights are volume-mounted and persist across Docker rebuilds.
- The web service uses the `web` Docker Compose profile.

### Docker CLI (Linux + NVIDIA GPU)

If you prefer the command-line interface in Docker:

Prerequisites:
- Docker Engine + Docker Compose plugin installed.
- NVIDIA driver installed on the host (Linux), with CUDA compatibility for the PyTorch CUDA 12.8 wheels used by this project.
- NVIDIA Container Toolkit installed and configured for Docker (`nvidia-smi` should work on host, and `docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi` should succeed).

1. Build the image:
   ```bash
   docker build -t corridorkey:latest .
   ```
2. Run an action directly (example: inference):
   ```bash
   docker run --rm -it --gpus all \
     -e OPENCV_IO_ENABLE_OPENEXR=1 \
     -v "$(pwd)/ClipsForInference:/app/ClipsForInference" \
     -v "$(pwd)/Output:/app/Output" \
     -v "$(pwd)/CorridorKeyModule/checkpoints:/app/CorridorKeyModule/checkpoints" \
     -v "$(pwd)/gvm_core/weights:/app/gvm_core/weights" \
     -v "$(pwd)/VideoMaMaInferenceModule/checkpoints:/app/VideoMaMaInferenceModule/checkpoints" \
     corridorkey:latest run_inference --device cuda
   ```
3. Docker Compose (recommended for repeat runs):
   ```bash
   docker compose build
   docker compose --profile gpu run --rm corridorkey run_inference --device cuda
   docker compose --profile gpu run --rm corridorkey list
   docker compose --profile cpu run --rm corridorkey-cpu run_inference --device cpu
   ```
4. Optional: pin to specific GPU(s) for multi-GPU workstations:
   ```bash
   NVIDIA_VISIBLE_DEVICES=0 docker compose --profile gpu run --rm corridorkey list
   NVIDIA_VISIBLE_DEVICES=1,2 docker compose --profile gpu run --rm corridorkey run_inference --device cuda
   ```

Notes:
- You still need to place model weights in the same folders used by native runs (mounted above).
- The container does not include kernel GPU drivers; those always come from the host. The image provides user-space dependencies and relies on Docker's NVIDIA runtime to pass through driver libraries/devices.
- The wizard works too, but use a path inside the container, for example:
  ```bash
  docker run --rm -it --gpus all \
    -e OPENCV_IO_ENABLE_OPENEXR=1 \
    -v "$(pwd)/ClipsForInference:/app/ClipsForInference" \
    -v "$(pwd)/Output:/app/Output" \
    -v "$(pwd)/CorridorKeyModule/checkpoints:/app/CorridorKeyModule/checkpoints" \
    -v "$(pwd)/gvm_core/weights:/app/gvm_core/weights" \
    -v "$(pwd)/VideoMaMaInferenceModule/checkpoints:/app/VideoMaMaInferenceModule/checkpoints" \
    corridorkey:latest wizard --win_path /app/ClipsForInference
  docker compose --profile gpu run --rm corridorkey wizard --win_path /app/ClipsForInference
  ```

### 3. Usage: The Command Line Wizard

For the easiest experience, use the provided launcher scripts. These scripts launch a prompt-based configuration wizard in your terminal.

*   **Windows:** Drag-and-drop a video file or folder onto `CorridorKey_DRAG_CLIPS_HERE_local.bat` (Note: Only launch via Drag-and-Drop or CMD. Double-clicking the `.bat` directly will throw an error).
*   **Linux / Mac:** Run or drag-and-drop a video file or folder onto `./CorridorKey_DRAG_CLIPS_HERE_local.sh`.
* - Or write `bash` again in terminal. Put a space after and then drag-and-drop `CorridorKey_DRAG_CLIPS_HERE_local.sh` and your clip folder together into terminal, respectively. Then press enter.

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
uv run python clip_manager.py --action wizard --win_path "V:\..." --device mps
uv run python clip_manager.py --action run_inference --device cpu
```

**Override via environment variable:**
```bash
export CORRIDORKEY_DEVICE=cpu
uv run python clip_manager.py --action wizard --win_path "V:\..."
```

Priority: `--device` flag > `CORRIDORKEY_DEVICE` env var > auto-detect.

### Apple Silicon / MPS Troubleshooting

**Confirm MPS is active:** Run with verbose logging to see which device was selected:
```bash
uv run python clip_manager.py --action list 2>&1 | grep -i "device\|backend\|mps"
```

**MPS operator errors** (`NotImplementedError: ... not implemented for 'MPS'`): Some PyTorch operations are not yet supported on MPS. Enable CPU fallback for those ops:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python corridorkey_cli.py wizard --win_path "/path/to/clips"
```

**Silent CPU fallback**: If MPS silently falls back to CPU without this variable, the run will be much slower. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` in your shell profile (`~/.zshrc`) ensures it is always active.

**Use native MLX instead of PyTorch MPS:** MLX avoids PyTorch's MPS layer entirely and typically runs faster on Apple Silicon. See the [Backend Selection](#backend-selection) section below for setup steps.

## Backend Selection

CorridorKey supports two inference backends:
- **Torch** (default on Linux/Windows) — CUDA, MPS, or CPU
- **MLX** (Apple Silicon) — native Metal acceleration, no Torch overhead

Resolution: `--backend` flag > `CORRIDORKEY_BACKEND` env var > auto-detect.
Auto mode prefers MLX on Apple Silicon when available.

**Override via CLI flag (corridorkey_cli.py):**
```bash
uv run python corridorkey_cli.py wizard --win_path "/path/to/clips" --backend mlx
uv run python corridorkey_cli.py run_inference --backend torch
```

### MLX Setup (Apple Silicon)

1. Install the MLX backend:
   ```bash
   uv sync --extra mlx
   ```
2. Obtain the MLX weights (`.safetensors`) — pick **one** option:

   **Option A — Download pre-converted weights (simplest):**
   ```bash
   # Download weights from GitHub Releases into a local cache directory
   uv run python -m corridorkey_mlx weights download

   # Print the cached path, then copy to the checkpoints folder
   WEIGHTS=$(uv run python -m corridorkey_mlx weights download --print-path)
   cp "$WEIGHTS" CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors
   ```

   **Option B — Convert from an existing `.pth` checkpoint:**
   ```bash
   # Clone the MLX repo (contains the conversion script)
   git clone https://github.com/nikopueringer/corridorkey-mlx.git
   cd corridorkey-mlx
   uv sync

   # Convert (point --checkpoint at your CorridorKey.pth)
   uv run python scripts/convert_weights.py \
       --checkpoint ../CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth \
       --output ../CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors
   cd ..
   ```

   Either way the final file must be at:
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
- **"corridorkey_mlx not installed"** — run `uv sync --extra mlx`
- **"MLX requires Apple Silicon"** — MLX only works on M1+ Macs
- **Auto picked Torch unexpectedly** — set `CORRIDORKEY_BACKEND=mlx` explicitly

## Advanced Usage

For developers looking for more details on the specifics of what is happening in the CorridorKey engine, check out the README in the `/CorridorKeyModule` folder. We also have a dedicated handover document outlining the pipeline architecture for AI assistants in `/docs/LLM_HANDOVER.md`.

You can also explore the full, auto-generated codebase documentation on [DeepWiki](https://deepwiki.com/nikopueringer/CorridorKey).

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

*   **Generative Video Matting (GVM):** Developed by the Advanced Intelligent Machines (AIM) research team at Zhejiang University. The GVM code and models are heavily utilized in the `gvm_core` module. Their work is licensed under the [2-clause BSD License (BSD-2-Clause)](https://opensource.org/license/bsd-2-clause). You can find their source repository here: [aim-uofa/GVM](https://github.com/aim-uofa/GVM). Give them a star!
*   **VideoMaMa:** Developed by the CVLAB at KAIST. The VideoMaMa architecture is utilized within the `VideoMaMaInferenceModule`. Their code is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/), and their specific foundation model checkpoints (`dino_projection_mlp.pth`, `unet/*`) are subject to the [Stability AI Community License](https://stability.ai/license). You can find their source repository here: [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa). Give them a star!

By using these optional modules, you agree to abide by their respective Non-Commercial licenses. Please review their repositories for full terms.
