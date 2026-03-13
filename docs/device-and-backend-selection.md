# Device and Backend Selection

## Device Selection

By default, CorridorKey auto-detects the best available compute device in this
priority order:

**CUDA → MPS → CPU**

### Override via CLI Flag

```bash
uv run python clip_manager.py --action wizard --win_path "V:\..." --device mps
uv run python clip_manager.py --action run_inference --device cpu
```

### Override via Environment Variable

```bash
export CORRIDORKEY_DEVICE=cpu
uv run python clip_manager.py --action wizard --win_path "V:\..."
```

!!! info "Resolution order"
    `--device` flag  >  `CORRIDORKEY_DEVICE` env var  >  auto-detect.

--8<-- "docs/_snippets/apple-silicon-note.md"

## Backend Selection

CorridorKey supports two inference backends:

| Backend | Platforms | Notes |
|---|---|---|
| **Torch** (default on Linux / Windows) | CUDA, MPS, CPU | Standard PyTorch inference. |
| **MLX** (Apple Silicon) | Metal | Native Apple Silicon acceleration — avoids PyTorch's MPS layer entirely and typically runs faster. |

### Override via CLI Flag

```bash
uv run python corridorkey_cli.py --action wizard --win_path "/path/to/clips" --backend mlx
uv run python corridorkey_cli.py --action run_inference --backend torch
```

### Override via Environment Variable

```bash
export CORRIDORKEY_BACKEND=mlx
uv run python corridorkey_cli.py --action run_inference
```

!!! info "Resolution order"
    `--backend` flag  >  `CORRIDORKEY_BACKEND` env var  >  auto-detect.

    Auto mode prefers MLX on Apple Silicon when the package is installed.


## MLX Setup (Apple Silicon)

Follow these steps to use the native MLX backend on an M1+ Mac.

### 1. Install the MLX Extras

```bash
uv sync --extra mlx
```

### 2. Obtain MLX Weights (`.safetensors`)

=== "Option A — Download Pre-Converted Weights (simplest)"

    ```bash
    # Download weights from GitHub Releases into a local cache directory
    uv run python -m corridorkey_mlx weights download

    # Print the cached path, then copy to the checkpoints folder
    WEIGHTS=$(uv run python -m corridorkey_mlx weights download --print-path)
    cp "$WEIGHTS" CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors
    ```

=== "Option B — Convert from an Existing `.pth` Checkpoint"

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

Either way, the final file must be at:

```
CorridorKeyModule/checkpoints/corridorkey_mlx.safetensors
```

### 3. Run

```bash
CORRIDORKEY_BACKEND=mlx uv run python clip_manager.py --action run_inference
```

MLX uses `img_size=2048` by default (same as Torch).

## Troubleshooting

### MPS (PyTorch Metal)

**Confirm MPS is active** — run with verbose logging to see which device was
selected:

```bash
uv run python clip_manager.py --action list 2>&1 | grep -i "device\|backend\|mps"
```

**MPS operator errors** (`NotImplementedError: ... not implemented for 'MPS'`):
Some PyTorch operations are not yet supported on MPS. Enable CPU fallback:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python corridorkey_cli.py --action wizard --win_path "/path/to/clips"
```

!!! tip "Make the fallback permanent"
    Add `export PYTORCH_ENABLE_MPS_FALLBACK=1` to your shell profile
    (`~/.zshrc`) so it is always active. Without it, MPS may silently fall back
    to CPU, making runs much slower.

**Use native MLX instead of PyTorch MPS** — MLX avoids PyTorch's MPS layer
entirely and typically runs faster on Apple Silicon. See the
[MLX Setup](#mlx-setup-apple-silicon) section above.

### MLX

| Symptom | Fix |
|---|---|
| `No .safetensors checkpoint found` | Place MLX weights in `CorridorKeyModule/checkpoints/`. |
| `corridorkey_mlx not installed` | Run `uv sync --extra mlx`. |
| `MLX requires Apple Silicon` | MLX only works on M1+ Macs. |
| Auto picked Torch unexpectedly | Set `CORRIDORKEY_BACKEND=mlx` explicitly. |
