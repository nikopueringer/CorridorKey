import timeit

import numpy as np
import torch

from CorridorKeyModule.inference_engine import CorridorKeyEngine


def process_frame(engine):
    img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (2160, 3840), dtype=np.uint8)

    engine.process_frame(img, mask)


def test_vram():
    print("Loading engine...")
    engine = CorridorKeyEngine(
        checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth",
        img_size=2048,
        device="cuda",
        model_precision=torch.float16,
    )

    # Reset stats
    torch.cuda.reset_peak_memory_stats()

    iterations = 24
    print(f"Running {iterations} inference passes...")
    time = timeit.timeit(lambda: process_frame(engine), number=iterations)
    print(f"Seconds per frame: {time / iterations}")

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak VRAM used: {peak_vram:.2f} GB")


if __name__ == "__main__":
    test_vram()
