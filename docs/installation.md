# Installation

--8<-- "docs/_snippets/uv-install.md"

## Platform Setup

=== "Windows"

    1. Clone or download this repository to your local machine.
    2. Double-click `Install_CorridorKey_Windows.bat`. This will automatically
       install uv (if needed), set up your Python environment, install all
       dependencies, and download the CorridorKey model.

        !!! note "CUDA driver requirement"
            To run GPU acceleration natively on Windows, your system **must**
            have NVIDIA drivers that support **CUDA 12.8 or higher**. If your
            drivers only support older CUDA versions, the installer will likely
            fall back to the CPU.

    3. *(Optional)* Double-click `Install_GVM_Windows.bat` and
       `Install_VideoMaMa_Windows.bat` to download the heavy optional Alpha
       Hint generator weights.

=== "Linux / Mac"

    1. Clone or download this repository to your local machine.
    2. Install uv if you don't have it:

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    3. Install all dependencies (uv will download Python 3.10+ automatically
       if needed):

        ```bash
        uv sync                  # CPU/MPS (default — works everywhere)
        uv sync --extra cuda     # CUDA GPU acceleration (Linux/Windows)
        uv sync --extra mlx      # Apple Silicon MLX acceleration
        ```

## Download the Model Checkpoint

--8<-- "docs/_snippets/model-download.md"

## Optional Weights

--8<-- "docs/_snippets/optional-weights.md"

## Docker (Linux + NVIDIA GPU)

If you prefer not to install dependencies locally, you can run CorridorKey in
Docker.

### Prerequisites

- Docker Engine + Docker Compose plugin installed.
- NVIDIA driver installed on the host (Linux), with CUDA compatibility for the
  PyTorch CUDA 12.6 wheels used by this project.
- NVIDIA Container Toolkit installed and configured for Docker (`nvidia-smi`
  should work on the host, and
  `docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi`
  should succeed).

### Build and Run

1. Build the image:

    ```bash
    docker build -t corridorkey:latest .
    ```

2. Run an action directly (example — inference):

    ```bash
    docker run --rm -it --gpus all \
      -e OPENCV_IO_ENABLE_OPENEXR=1 \
      -v "$(pwd)/ClipsForInference:/app/ClipsForInference" \
      -v "$(pwd)/Output:/app/Output" \
      -v "$(pwd)/CorridorKeyModule/checkpoints:/app/CorridorKeyModule/checkpoints" \
      -v "$(pwd)/gvm_core/weights:/app/gvm_core/weights" \
      -v "$(pwd)/VideoMaMaInferenceModule/checkpoints:/app/VideoMaMaInferenceModule/checkpoints" \
      corridorkey:latest --action run_inference --device cuda
    ```

3. Docker Compose (recommended for repeat runs):

    ```bash
    docker compose build
    docker compose --profile gpu run --rm corridorkey --action run_inference --device cuda
    docker compose --profile gpu run --rm corridorkey --action list
    docker compose --profile cpu run --rm corridorkey-cpu --action run_inference --device cpu
    ```

4. *(Optional)* Pin to specific GPU(s) for multi-GPU workstations:

    ```bash
    NVIDIA_VISIBLE_DEVICES=0 docker compose --profile gpu run --rm corridorkey --action list
    NVIDIA_VISIBLE_DEVICES=1,2 docker compose --profile gpu run --rm corridorkey --action run_inference --device cuda
    ```

!!! info "Notes"
    - You still need to place model weights in the same folders used by native
      runs (mounted above).
    - The container does not include kernel GPU drivers; those always come from
      the host. The image provides user-space dependencies and relies on
      Docker's NVIDIA runtime to pass through driver libraries/devices.
    - The wizard works too — use a path inside the container, for example:

        ```bash
        docker run --rm -it --gpus all \
          -e OPENCV_IO_ENABLE_OPENEXR=1 \
          -v "$(pwd)/ClipsForInference:/app/ClipsForInference" \
          -v "$(pwd)/Output:/app/Output" \
          -v "$(pwd)/CorridorKeyModule/checkpoints:/app/CorridorKeyModule/checkpoints" \
          -v "$(pwd)/gvm_core/weights:/app/gvm_core/weights" \
          -v "$(pwd)/VideoMaMaInferenceModule/checkpoints:/app/VideoMaMaInferenceModule/checkpoints" \
          corridorkey:latest --action wizard --win_path /app/ClipsForInference
        ```
