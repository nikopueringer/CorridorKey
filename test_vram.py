import timeit

import numpy as np
import torch

from CorridorKeyModule.inference_engine import CorridorKeyEngine


def process_frame(engine: CorridorKeyEngine):
    img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (2160, 3840), dtype=np.uint8)

    engine.process_frame(img, mask)


def batch_process_frame(engine: CorridorKeyEngine, batch_size: int):
    imgs = np.random.randint(0, 255, (batch_size, 2160, 3840, 3), dtype=np.uint8)
    masks = np.random.randint(0, 255, (batch_size, 2160, 3840), dtype=np.uint8)

    engine.batch_process_frames(imgs, masks)


def test_vram():
    torch.backends.cudnn.benchmark = True

    print("Loading engine...")
    engine = CorridorKeyEngine(
        checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth",
        img_size=2048,
        device="cuda",
        model_precision=torch.float16,
        mixed_precision=True,
    )

    # Reset stats
    torch.cuda.reset_peak_memory_stats()

    total_seconds = 6
    batch_size = 2  # works with a 16GB GPU
    iterations = total_seconds * 24 // batch_size
    print(f"Running {iterations} inference passes...")
    time = timeit.timeit(
        lambda: batch_process_frame(engine, batch_size),
        number=iterations,
        setup=lambda: (
            batch_process_frame(engine, batch_size),
            torch.cuda.synchronize(),
            torch.cuda.empty_cache(),
            print("Compilation and warmup complete, starting timed runs..."),
        ),
    )
    print(f"Seconds per frame: {time / (iterations * batch_size):.4f}")

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak VRAM used: {peak_vram:.2f} GiB")


if __name__ == "__main__":
    test_vram()
