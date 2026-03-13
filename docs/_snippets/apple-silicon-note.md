!!! note "Apple Silicon (MPS / MLX)"
    CorridorKey runs on Apple Silicon Macs using unified memory. Two backend
    options are available:

    - **MPS** — PyTorch's Metal Performance Shaders backend. Works out of the
      box but some operators may require the CPU fallback flag:

        ```bash
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        ```

    - **MLX** — Native Apple Silicon acceleration via the
      [MLX framework](https://github.com/ml-explore/mlx). Avoids PyTorch's MPS
      layer entirely and typically runs faster. Requires installing the MLX
      extras (`uv sync --extra mlx`) and obtaining `.safetensors` weights.

    Because Apple Silicon shares memory between the CPU and GPU, the full
    system RAM is available to the model — no separate VRAM budget applies.
