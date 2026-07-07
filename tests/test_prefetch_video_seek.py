"""Video assets must be seeked to the start of a partial frame range.

VideoCapture decodes sequentially from frame 0, so when run_inference is
given in/out markers the prefetch thread has to seek before reading —
otherwise frame 0's content is processed under the range-start stem (and
video assets desync from image-sequence assets, which index directly).
"""

from queue import Queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.service import CorridorKeyService

FRAME_COUNT = 24
STEP = 8  # gray value per frame index; large enough to survive lossy encoding
TOLERANCE = 3.0


@pytest.fixture(scope="module")
def index_encoded_video(tmp_path_factory):
    """A tiny video whose frame N is solid gray with value N * STEP."""
    path = str(tmp_path_factory.mktemp("video") / "encoded.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter unavailable in this environment")
    for i in range(FRAME_COUNT):
        writer.write(np.full((32, 32, 3), i * STEP, dtype=np.uint8))
    writer.release()

    probe = cv2.VideoCapture(path)
    ok, _ = probe.read()
    probe.release()
    if not ok:
        pytest.skip("OpenCV cannot read back mp4v video in this environment")
    return path


def _decoded_index(mean_value: float) -> float:
    """Map a frame's mean pixel value back to the frame index it encodes."""
    return mean_value * 255.0 / STEP


def _run_prefetch(video_path: str, frame_indices) -> list[tuple]:
    service = CorridorKeyService()
    clip = SimpleNamespace(name="seek-test")
    input_cap = cv2.VideoCapture(video_path)
    alpha_cap = cv2.VideoCapture(video_path)
    q: Queue = Queue()
    try:
        service._prefetch_frames(clip, frame_indices, [], [], input_cap, alpha_cap, False, q, None)
    finally:
        input_cap.release()
        alpha_cap.release()

    items = []
    while True:
        item = q.get_nowait()
        if item is None:
            break
        items.append(item)
    return items


def test_partial_range_reads_matching_frames(index_encoded_video):
    start, end = 10, 15
    items = _run_prefetch(index_encoded_video, range(start, end + 1))

    assert [item[0] for item in items] == list(range(start, end + 1))
    for i, img, mask, stem, _is_linear, err in items:
        assert err is None
        assert stem == f"{i:05d}"
        assert img is not None and mask is not None
        assert _decoded_index(img.mean()) == pytest.approx(i, abs=TOLERANCE), (
            f"frame labeled {i} contains content of frame {_decoded_index(img.mean()):.1f}"
        )
        assert _decoded_index(mask.mean()) == pytest.approx(i, abs=TOLERANCE)


def test_full_range_is_unaffected(index_encoded_video):
    items = _run_prefetch(index_encoded_video, range(0, 5))

    for i, img, _mask, _stem, _is_linear, err in items:
        assert err is None
        assert _decoded_index(img.mean()) == pytest.approx(i, abs=TOLERANCE)
