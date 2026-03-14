# corridorkey

`corridorkey` is the Application Layer of CorridorKey. It owns the processing pipeline, clip state machine, job queue, project management, and frame I/O. It has no UI dependencies and can be consumed by any frontend - CLI, GUI, or web.

## Installation

```shell
uv add corridorkey
```

For CUDA (Windows/Linux):

```shell
uv add "corridorkey[cuda]"
```

For MLX (Apple Silicon):

```shell
uv add "corridorkey[mlx]"
```

## How It Works

`CorridorKeyService` is the single entry point for all processing. It holds the loaded inference engine and GPU lock. Everything else - project management, frame I/O, validation - is handled by pure functions in their respective modules.

Clip state flows through a strict state machine:

```text
EXTRACTING -> RAW -> READY -> COMPLETE -> MASKED -> READY
```

Any state can transition to `ERROR` and back on retry.

## Quick Start

```python
from corridorkey import CorridorKeyService, InferenceParams, ClipState

service = CorridorKeyService()
service.detect_device()

clips = service.scan_clips("/path/to/clips")
ready = service.get_clips_by_state(clips, ClipState.READY)

for clip in ready:
    results = service.run_inference(clip, InferenceParams())
    print(f"{clip.name}: {sum(r.success for r in results)} frames processed")

service.unload_engine()
```

## Batch Processing

For CLI and batch workflows, `process_directory` wraps the full pipeline into a single call:

```python
from corridorkey import process_directory, InferenceParams

result = process_directory(
    "/path/to/clips",
    params=InferenceParams(despill_strength=0.8),
    device="cuda",
)

print(f"Done: {len(result.succeeded)} succeeded, {len(result.failed)} failed")
```

## Alpha Generation

Alpha generators are pluggable via the `AlphaGenerator` protocol. Any object with a `name` property and a `generate` method can be passed to `run_alpha_generator`:

```python
from corridorkey import CorridorKeyService, ClipState

service = CorridorKeyService()
service.detect_device()

clips = service.scan_clips("/path/to/clips")
raw_clips = service.get_clips_by_state(clips, ClipState.RAW)

for clip in raw_clips:
    service.run_alpha_generator(clip, my_generator)
```

## Job Queue

For GUI and async workflows, `GPUJobQueue` ensures only one GPU job runs at a time:

```python
from corridorkey import GPUJob, GPUJobQueue, JobType

queue = GPUJobQueue()
queue.on_completion = lambda clip_name: print(f"{clip_name} done")

queue.submit(GPUJob(JobType.INFERENCE, "shot1"))
queue.submit(GPUJob(JobType.ALPHA_GEN, "shot2"))
```

## Related

- [corridorkey API Overview](../api/corridorkey/index.md)
- [service reference](../api/corridorkey/service.md)
- [pipeline reference](../api/corridorkey/pipeline.md)
- [clip-state reference](../api/corridorkey/clip-state.md)
- [job-queue reference](../api/corridorkey/job-queue.md)
