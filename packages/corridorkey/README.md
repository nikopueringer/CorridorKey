# corridorkey

<div align="center">
  <a href="https://pypi.org/project/corridorkey/">
    <img src="https://img.shields.io/pypi/v/corridorkey?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
</div>

<div align="center">Application layer for the CorridorKey AI chroma keying pipeline.</div>

## What's in this package

- `CorridorKeyService` - pipeline orchestration: scan clips, run inference, write outputs, resume support
- `ClipEntry` / `ClipState` - clip data model and state machine (RAW -> READY -> COMPLETE)
- `GPUJobQueue` - thread-safe GPU job scheduling with cancellation and progress callbacks
- `create_engine` - engine factory supporting Torch (CUDA/CPU) and MLX (Apple Silicon)
- Project management utilities - create/scan projects, read/write metadata
- FFmpeg wrappers - frame extraction and video stitching

## Installation

```bash
uv add corridorkey
```

For CUDA support:

```bash
uv add corridorkey --extra cuda
```

For Apple Silicon (MLX):

```bash
uv add corridorkey --extra mlx
```

## Usage

```python
from corridorkey import CorridorKeyService, InferenceParams, ClipState

service = CorridorKeyService()
service.detect_device()

clips = service.scan_clips("/path/to/ClipsForInference")
ready = service.get_clips_by_state(clips, ClipState.READY)

for clip in ready:
    params = InferenceParams(despill_strength=0.8)
    service.run_inference(clip, params, on_progress=lambda name, cur, tot: print(f"{name}: {cur}/{tot}"))
```

## Architecture

This package is the Application Layer of the CorridorKey architecture. It sits between
the UI (CLI/GUI) and `corridorkey-core` (model inference), handling all pipeline
orchestration, state management, and I/O.

```
CLI / GUI
    |
corridorkey          (this package - pipeline, state, I/O)
    |
corridorkey-core     (model inference, compositing)
```

## License

See [LICENSE](https://github.com/nikopueringer/CorridorKey/blob/main/LICENSE).
