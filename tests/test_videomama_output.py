"""run_videomama output handling.

VideoMaMa's inference generator yields uint8 RGB frames in [0, 255]
(PIL round-trip). The write path must not clip them to [0, 1] before
rescaling — that binarizes the soft alpha the model produces.
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


@pytest.fixture
def masked_clip(tmp_path):
    """A tiny MASKED clip with an input sequence and a mask-hint sequence."""
    root = tmp_path / "Shot"
    input_dir = root / "Input"
    mask_dir = root / "VideoMamaMaskHint"
    input_dir.mkdir(parents=True)
    mask_dir.mkdir()
    for i in range(FRAME_COUNT):
        frame = np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"frame_{i:06d}.png"), frame)
        cv2.imwrite(str(mask_dir / f"frame_{i:06d}.png"), np.full((SIZE, SIZE), 255, dtype=np.uint8))

    return ClipEntry(
        name="Shot",
        root_path=str(root),
        state=ClipState.MASKED,
        input_asset=ClipAsset(str(input_dir), "sequence"),
        mask_asset=ClipAsset(str(mask_dir), "sequence"),
    )


def _gradient_frame() -> np.ndarray:
    """Soft alpha ramp 0..255 across the width — destroyed by [0, 1] clipping."""
    ramp = np.linspace(0, 255, SIZE, dtype=np.uint8)
    return np.repeat(np.tile(ramp, (SIZE, 1))[:, :, None], 3, axis=2)


@pytest.fixture
def fake_videomama(monkeypatch):
    """Replace the VideoMaMa pipeline and inference generator with fakes."""
    calls: list[dict] = []

    def fake_run_inference(pipeline, input_frames, mask_frames, chunk_size=24):
        for i in range(0, len(input_frames), chunk_size):
            chunk = input_frames[i : i + chunk_size]
            calls.append({"chunk_start": i, "chunk_len": len(chunk)})
            yield [_gradient_frame() for _ in chunk]

    fake_module = types.ModuleType("VideoMaMaInferenceModule.inference")
    fake_module.run_inference = fake_run_inference
    monkeypatch.setitem(sys.modules, "VideoMaMaInferenceModule.inference", fake_module)
    monkeypatch.setattr(CorridorKeyService, "_get_videomama_pipeline", lambda self: object())
    return calls


def test_soft_alpha_survives_write(masked_clip, fake_videomama):
    service = CorridorKeyService()
    service.run_videomama(masked_clip, chunk_size=2)

    alpha_dir = os.path.join(masked_clip.root_path, "AlphaHint")
    written = sorted(os.listdir(alpha_dir))
    assert written == [f"frame_{i:06d}.png" for i in range(FRAME_COUNT)]

    alpha = cv2.imread(os.path.join(alpha_dir, written[0]), cv2.IMREAD_GRAYSCALE)
    midtones = np.count_nonzero((alpha > 32) & (alpha < 224))
    assert midtones > SIZE * SIZE * 0.5, (
        f"soft alpha gradient was binarized: only {midtones} midtone pixels, "
        f"unique values {np.unique(alpha)[:8]}..."
    )


def test_clip_transitions_to_ready(masked_clip, fake_videomama):
    service = CorridorKeyService()
    service.run_videomama(masked_clip, chunk_size=2)
    assert masked_clip.state == ClipState.READY
