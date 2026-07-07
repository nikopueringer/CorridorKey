"""run_videomama output handling and chunked streaming.

VideoMaMa's inference generator yields uint8 RGB frames in [0, 255]
(PIL round-trip). The write path must not clip them to [0, 1] before
rescaling — that binarizes the soft alpha the model produces.

Frames are decoded lazily, one chunk at a time: each inference call must
receive at most chunk_size frames, and resume must not re-run inference
for chunks that already have alpha frames on disk.
"""

import os
import sys
import types

import cv2
import numpy as np
import pytest

from backend.clip_state import ClipAsset, ClipEntry, ClipState
from backend.service import CorridorKeyService

FRAME_COUNT = 4
SIZE = 16


def _make_clip(root, input_asset) -> ClipEntry:
    mask_dir = os.path.join(root, "VideoMamaMaskHint")
    os.makedirs(mask_dir, exist_ok=True)
    for i in range(FRAME_COUNT):
        cv2.imwrite(
            os.path.join(mask_dir, f"frame_{i:06d}.png"),
            np.full((SIZE, SIZE), 255, dtype=np.uint8),
        )
    return ClipEntry(
        name=os.path.basename(root),
        root_path=str(root),
        state=ClipState.MASKED,
        input_asset=input_asset,
        mask_asset=ClipAsset(mask_dir, "sequence"),
    )


@pytest.fixture
def masked_clip(tmp_path):
    """A tiny MASKED clip with an input sequence and a mask-hint sequence."""
    root = tmp_path / "Shot"
    input_dir = root / "Input"
    input_dir.mkdir(parents=True)
    for i in range(FRAME_COUNT):
        frame = np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"frame_{i:06d}.png"), frame)
    return _make_clip(str(root), ClipAsset(str(input_dir), "sequence"))


@pytest.fixture
def masked_video_clip(tmp_path):
    """A MASKED clip whose input asset is a video file."""
    root = tmp_path / "VidShot"
    root.mkdir()
    video_path = str(root / "input.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (SIZE, SIZE))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter unavailable in this environment")
    for _ in range(FRAME_COUNT):
        writer.write(np.full((SIZE, SIZE, 3), 40, dtype=np.uint8))
    writer.release()
    return _make_clip(str(root), ClipAsset(video_path, "video"))


def _gradient_frame() -> np.ndarray:
    """Soft alpha ramp 0..255 across the width — destroyed by [0, 1] clipping."""
    ramp = np.linspace(0, 255, SIZE, dtype=np.uint8)
    return np.repeat(np.tile(ramp, (SIZE, 1))[:, :, None], 3, axis=2)


@pytest.fixture
def fake_videomama(monkeypatch):
    """Replace the VideoMaMa pipeline and inference generator with fakes.

    Returns a list that records the frame count of every inference call.
    """
    calls: list[int] = []

    def fake_run_inference(pipeline, input_frames, mask_frames, chunk_size=24):
        assert len(input_frames) == len(mask_frames)
        for i in range(0, len(input_frames), chunk_size):
            chunk = input_frames[i : i + chunk_size]
            calls.append(len(chunk))
            yield [_gradient_frame() for _ in chunk]

    fake_module = types.ModuleType("VideoMaMaInferenceModule.inference")
    fake_module.run_inference = fake_run_inference
    monkeypatch.setitem(sys.modules, "VideoMaMaInferenceModule.inference", fake_module)
    monkeypatch.setattr(CorridorKeyService, "_get_videomama_pipeline", lambda self: object())
    return calls


def _alpha_files(clip: ClipEntry) -> list[str]:
    return sorted(os.listdir(os.path.join(clip.root_path, "AlphaHint")))


def test_soft_alpha_survives_write(masked_clip, fake_videomama):
    service = CorridorKeyService()
    service.run_videomama(masked_clip, chunk_size=2)

    written = _alpha_files(masked_clip)
    assert written == [f"frame_{i:06d}.png" for i in range(FRAME_COUNT)]

    alpha_path = os.path.join(masked_clip.root_path, "AlphaHint", written[0])
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    midtones = np.count_nonzero((alpha > 32) & (alpha < 224))
    assert midtones > SIZE * SIZE * 0.5, (
        f"soft alpha gradient was binarized: only {midtones} midtone pixels, unique values {np.unique(alpha)[:8]}..."
    )


def test_inference_runs_one_chunk_at_a_time(masked_clip, fake_videomama):
    service = CorridorKeyService()
    service.run_videomama(masked_clip, chunk_size=2)
    assert fake_videomama == [2, 2]


def test_resume_skips_completed_chunks(masked_clip, fake_videomama):
    # 3 of 4 alpha frames already on disk with chunk_size=1 → rollback keeps
    # chunks 0-1 and re-runs from frame 2.
    alpha_dir = os.path.join(masked_clip.root_path, "AlphaHint")
    os.makedirs(alpha_dir)
    for i in range(3):
        cv2.imwrite(
            os.path.join(alpha_dir, f"frame_{i:06d}.png"),
            np.zeros((SIZE, SIZE), dtype=np.uint8),
        )

    service = CorridorKeyService()
    service.run_videomama(masked_clip, chunk_size=1)

    assert fake_videomama == [1, 1], "resume must not re-run inference for kept chunks"
    assert _alpha_files(masked_clip) == [f"frame_{i:06d}.png" for i in range(FRAME_COUNT)]


def test_video_input_streams_chunks(masked_video_clip, fake_videomama):
    service = CorridorKeyService()
    service.run_videomama(masked_video_clip, chunk_size=3)

    assert fake_videomama == [3, 1]
    assert _alpha_files(masked_video_clip) == [f"frame_{i:06d}.png" for i in range(FRAME_COUNT)]


def test_clip_transitions_to_ready(masked_clip, fake_videomama):
    service = CorridorKeyService()
    service.run_videomama(masked_clip, chunk_size=2)
    assert masked_clip.state == ClipState.READY
