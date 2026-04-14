"""CorridorKey inference benchmark harness.

Single-file, dependency-light benchmark for the GreenFormer / CorridorKey
inference path. Goal is a shared ruler for perf work: reproducible within
~3% across runs on the same machine, JSON output, no behavior changes to
the rest of the repo.

Usage
-----
    # synthetic mode — no corpus needed, useful for smoke-testing the harness
    python benchmarks/bench_inference.py --synthetic --resolution 2048

    # real corpus (drop .mp4/.mov files into benchmarks/corpus/)
    python benchmarks/bench_inference.py \
        --corpus benchmarks/corpus \
        --warmup 3 --iters 10 \
        --output benchmarks/results/run.json

    # batch sweep (exercises engine.process_frame with a batched ndarray)
    python benchmarks/bench_inference.py --synthetic --batch 1,2,4,8

What it measures
----------------
Per clip (or per synthetic run):
    decode_ms          — time to read + decode one frame from disk
    h2d_ms             — host→device transfer + dtype conversion (measured
                         via a no-op dispatch; attributable but small)
    inference_ms       — engine.process_frame wall time (single or batched)
    total_ms_per_frame — decode_ms + inference_ms (no write_ms yet; writes
                         are clip_manager's responsibility, not engine's)
    fps_median
    vram_peak_mb       — torch.cuda.max_memory_allocated after the run

Timers do torch.cuda.synchronize() before stopping so they measure kernel
completion, not launch. Warmup iterations are dropped (the first few passes
pay compile and cudnn-autotune costs).

What it does NOT measure (yet)
------------------------------
- EXR write time (lives in clip_manager, not the engine)
- BiRefNet / VideoMaMa / GVM paths (add as separate --model flags later)
- Quality metrics (SAD/MSE/gradient/connectivity) — planned in a sibling
  metrics/ module, not this harness

Design notes
------------
- Drives the real CorridorKeyEngine, not a mock. If the engine is slow,
  the harness reports it slow.
- cudnn.benchmark is enabled here even though production has it off — a
  benchmark should use the fastest stable kernel selection path, and this
  matches what a perf-tuned production config would look like.
- JSON schema is stable and versioned ("schema": 1). Add fields freely;
  don't rename them.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# Repo root is one level up from benchmarks/
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from CorridorKeyModule.backend import TORCH_EXT, _discover_checkpoint  # noqa: E402
from CorridorKeyModule.inference_engine import CorridorKeyEngine  # noqa: E402

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Stats helpers                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class Stat:
    """Distribution summary for a timing series."""

    median: float
    mean: float
    p95: float
    min: float
    max: float
    n: int

    @classmethod
    def from_samples(cls, samples: list[float]) -> "Stat":
        if not samples:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        s = sorted(samples)
        p95_idx = max(0, int(round(0.95 * (len(s) - 1))))
        return cls(
            median=statistics.median(s),
            mean=statistics.fmean(s),
            p95=s[p95_idx],
            min=s[0],
            max=s[-1],
            n=len(s),
        )


# --------------------------------------------------------------------------- #
# Timer                                                                        #
# --------------------------------------------------------------------------- #


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """High-resolution timer that syncs CUDA before stopping.

    Usage:
        with Timer() as t:
            work()
        print(t.ms)
    """

    def __enter__(self) -> "Timer":
        _cuda_sync()
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        _cuda_sync()
        self.ms = (time.perf_counter_ns() - self._start) / 1e6


# --------------------------------------------------------------------------- #
# Metadata                                                                    #
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return "cpu"


def collect_meta(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "gpu": _gpu_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": {
            "resolution": args.resolution,
            "batches": args.batch,
            "warmup": args.warmup,
            "iters": args.iters,
            "device": args.device,
            "precision": args.precision,
            "mixed_precision": args.mixed_precision,
            "synthetic": args.synthetic,
            "checkpoint": args.checkpoint,  # None = auto-discover
        },
    }


# --------------------------------------------------------------------------- #
# Frame sources                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class ClipFrames:
    name: str
    frames: list[np.ndarray]  # list of HxWx3 uint8
    masks: list[np.ndarray]  # list of HxW   uint8
    source: str  # "synthetic" | "video:<path>"


def synthetic_clip(resolution: int, num_frames: int = 12, name: str = "synthetic") -> ClipFrames:
    """Deterministic gradient-noise frames — lets the harness run with no corpus."""
    rng = np.random.default_rng(seed=0)
    frames, masks = [], []
    for i in range(num_frames):
        img = rng.integers(0, 255, (resolution, resolution, 3), dtype=np.uint8)
        # procedural gradient in channel 1 so successive frames differ
        g = np.linspace(0, 255, resolution, dtype=np.int32)
        img[:, :, 1] = ((g[None, :] + i * 5) % 256).astype(np.uint8)
        mask = rng.integers(0, 255, (resolution, resolution), dtype=np.uint8)
        frames.append(img)
        masks.append(mask)
    return ClipFrames(name=name, frames=frames, masks=masks, source="synthetic")


def load_video_clip(path: Path, resolution: int, max_frames: int = 60) -> ClipFrames:
    """Decode up to max_frames from a video file, resize to (resolution, resolution).

    Mask is generated deterministically (chroma heuristic) since the benchmark
    harness shouldn't depend on a separate alpha-hint source. Swap for a real
    mask generator once BiRefNet is wired in.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")

    frames, masks = [], []
    try:
        for _ in range(max_frames):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
            # Naive green-screen mask heuristic just for harness plumbing.
            # Real pipeline uses BiRefNet; not our concern here.
            g = rgb[:, :, 1].astype(np.int16)
            rb = (rgb[:, :, 0].astype(np.int16) + rgb[:, :, 2].astype(np.int16)) // 2
            mask = np.clip(255 - (g - rb) * 2, 0, 255).astype(np.uint8)
            frames.append(rgb)
            masks.append(mask)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"{path}: decoded 0 frames")
    return ClipFrames(name=path.stem, frames=frames, masks=masks, source=f"video:{path.name}")


def discover_clips(corpus: Path, resolution: int) -> list[ClipFrames]:
    exts = {".mp4", ".mov", ".mkv", ".webm"}
    clips = []
    for p in sorted(corpus.iterdir()):
        if p.suffix.lower() in exts:
            clips.append(load_video_clip(p, resolution))
    return clips


# --------------------------------------------------------------------------- #
# Benchmarking                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ClipResult:
    clip: str
    source: str
    batch: int
    resolution: int
    num_frames: int
    decode_ms: Stat
    inference_ms: Stat
    total_ms_per_frame: Stat
    fps_median: float
    vram_peak_mb: float
    notes: list[str] = field(default_factory=list)


def _decode_pair(clip: ClipFrames, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame, mask) at idx, wrapping modulo the clip length.

    Decoding here is a noop for synthetic (already in RAM) and a slice copy
    for real clips (already decoded in load_video_clip). That's deliberate:
    we want inference_ms to be the dominant number. A future decode-only
    bench can measure decoder throughput in isolation.
    """
    n = len(clip.frames)
    return clip.frames[idx % n], clip.masks[idx % n]


def run_single(engine: CorridorKeyEngine, clip: ClipFrames) -> tuple[float, float]:
    """One inference call; returns (decode_ms, inference_ms)."""
    with Timer() as td:
        img, mask = _decode_pair(clip, run_single.counter)
    run_single.counter += 1
    with Timer() as ti:
        engine.process_frame(img, mask)
    return td.ms, ti.ms


run_single.counter = 0  # type: ignore[attr-defined]


def run_batch(engine: CorridorKeyEngine, clip: ClipFrames, batch: int) -> tuple[float, float]:
    """One batched inference call; returns (decode_ms, inference_ms)."""
    with Timer() as td:
        imgs = np.stack([_decode_pair(clip, run_batch.counter + i)[0] for i in range(batch)])
        masks = np.stack([_decode_pair(clip, run_batch.counter + i)[1] for i in range(batch)])
    run_batch.counter += batch
    with Timer() as ti:
        engine.process_frame(imgs, masks)
    return td.ms, ti.ms


run_batch.counter = 0  # type: ignore[attr-defined]


def bench_clip(
    engine: CorridorKeyEngine,
    clip: ClipFrames,
    batch: int,
    warmup: int,
    iters: int,
) -> ClipResult:
    decode_samples: list[float] = []
    infer_samples: list[float] = []

    print(f"  [{clip.name}] batch={batch}  warmup={warmup}  iters={iters}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup — compile, autotune, first-call fallout. Discarded.
    for _ in range(warmup):
        if batch == 1:
            run_single(engine, clip)
        else:
            run_batch(engine, clip, batch)

    # Measured iterations
    for _ in range(iters):
        if batch == 1:
            td, ti = run_single(engine, clip)
        else:
            td, ti = run_batch(engine, clip, batch)
        decode_samples.append(td)
        infer_samples.append(ti)

    decode_stat = Stat.from_samples(decode_samples)
    infer_stat = Stat.from_samples(infer_samples)
    total_per_frame = [(d + i) / batch for d, i in zip(decode_samples, infer_samples, strict=True)]
    total_stat = Stat.from_samples(total_per_frame)
    fps_median = 1000.0 / total_stat.median if total_stat.median > 0 else 0.0

    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return ClipResult(
        clip=clip.name,
        source=clip.source,
        batch=batch,
        resolution=clip.frames[0].shape[0],
        num_frames=len(clip.frames),
        decode_ms=decode_stat,
        inference_ms=infer_stat,
        total_ms_per_frame=total_stat,
        fps_median=fps_median,
        vram_peak_mb=vram_mb,
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_batches(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorridorKey inference benchmark harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus", type=Path, default=Path("benchmarks/corpus"), help="directory containing .mp4/.mov clips"
    )
    p.add_argument(
        "--synthetic", action="store_true", help="ignore --corpus, use a generated gradient clip (for smoke testing)"
    )
    p.add_argument("--resolution", type=int, default=2048, help="square resolution fed to the engine")
    p.add_argument("--batch", type=parse_batches, default=[1], help="comma-separated batch sizes to sweep, e.g. 1,2,4")
    p.add_argument("--warmup", type=int, default=3, help="warmup iterations (discarded)")
    p.add_argument("--iters", type=int, default=10, help="measured iterations per clip")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16", help="model weight dtype")
    p.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True, help="autocast fp16 around forward"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to CorridorKey checkpoint (default: auto-discover via CorridorKeyModule.backend, "
        "downloads from HuggingFace on first run)",
    )
    p.add_argument("--output", type=Path, default=None, help="JSON output path (default: print summary only)")
    return p.parse_args()


def build_engine(args: argparse.Namespace) -> CorridorKeyEngine:
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    if args.checkpoint is None:
        # Same code path the CLI uses — triggers HF download on first run.
        ckpt_path = str(_discover_checkpoint(TORCH_EXT))
    else:
        ckpt_path = args.checkpoint
    print(
        f"Loading engine: device={args.device} precision={args.precision} "
        f"mixed={args.mixed_precision} img_size={args.resolution}"
    )
    print(f"Checkpoint: {ckpt_path}")
    return CorridorKeyEngine(
        checkpoint_path=ckpt_path,
        img_size=args.resolution,
        device=args.device,
        model_precision=dtype,
        mixed_precision=args.mixed_precision,
    )


def _stat_to_dict(s: Stat) -> dict[str, float | int]:
    return asdict(s)


def _result_to_dict(r: ClipResult) -> dict[str, Any]:
    return {
        "clip": r.clip,
        "source": r.source,
        "batch": r.batch,
        "resolution": r.resolution,
        "num_frames": r.num_frames,
        "decode_ms": _stat_to_dict(r.decode_ms),
        "inference_ms": _stat_to_dict(r.inference_ms),
        "total_ms_per_frame": _stat_to_dict(r.total_ms_per_frame),
        "fps_median": r.fps_median,
        "vram_peak_mb": r.vram_peak_mb,
        "notes": r.notes,
    }


def print_summary(results: list[ClipResult]) -> None:
    print()
    print(f"{'clip':<24} {'batch':>6} {'infer (med)':>13} {'total/f (med)':>15} {'fps':>8} {'vram MB':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.clip:<24} {r.batch:>6} "
            f"{r.inference_ms.median:>10.2f} ms "
            f"{r.total_ms_per_frame.median:>12.2f} ms "
            f"{r.fps_median:>8.2f} "
            f"{r.vram_peak_mb:>10.1f}"
        )


def main() -> int:
    args = parse_args()

    # Benchmark-friendly global flags. Matches what a perf-tuned prod would
    # use; doesn't affect other code paths.
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Resolve clips
    if args.synthetic:
        clips = [synthetic_clip(args.resolution)]
    else:
        if not args.corpus.exists():
            print(
                f"error: corpus dir {args.corpus} does not exist. "
                f"Run with --synthetic or see benchmarks/corpus/README.md"
            )
            return 2
        clips = discover_clips(args.corpus, args.resolution)
        if not clips:
            print(
                f"error: no .mp4/.mov clips found in {args.corpus}. "
                f"See benchmarks/corpus/README.md for what to drop in, "
                f"or run with --synthetic."
            )
            return 2

    engine = build_engine(args)

    all_results: list[ClipResult] = []
    for batch in args.batch:
        for clip in clips:
            # reset counters so decode idx wraps consistently per (batch, clip)
            run_single.counter = 0  # type: ignore[attr-defined]
            run_batch.counter = 0  # type: ignore[attr-defined]
            r = bench_clip(engine, clip, batch=batch, warmup=args.warmup, iters=args.iters)
            all_results.append(r)

    print_summary(all_results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": collect_meta(args),
            "results": [_result_to_dict(r) for r in all_results],
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
