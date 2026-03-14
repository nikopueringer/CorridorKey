"""MLX integration smoke test — requires Apple Silicon + corridorkey_mlx."""

import numpy as np
import pytest

pytestmark = [pytest.mark.mlx, pytest.mark.slow]


@pytest.fixture
def mlx_engine():
    """Load MLX engine via create_engine at 2048."""
    from CorridorKeyModule.backend import create_engine

    return create_engine(backend="mlx", img_size=2048)


def test_mlx_smoke_2048(mlx_engine):
    """Process one synthetic frame and verify output contract."""
    h, w = 2048, 2048

    # Solid green image + white mask
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[:, :, 1] = 1.0  # green channel
    mask = np.ones((h, w, 1), dtype=np.float32)

    result = mlx_engine.process_frame(image, mask)

    # Keys
    assert set(result.keys()) == {"alpha", "fg", "comp", "processed"}

    # Shapes
    assert result["alpha"].shape == (h, w, 1), f"alpha shape: {result['alpha'].shape}"
    assert result["fg"].shape == (h, w, 3), f"fg shape: {result['fg'].shape}"
    assert result["comp"].shape == (h, w, 3), f"comp shape: {result['comp'].shape}"
    assert result["processed"].shape == (h, w, 4), f"processed shape: {result['processed'].shape}"

    # Dtypes
    for key in ("alpha", "fg", "comp", "processed"):
        assert result[key].dtype == np.float32, f"{key} dtype: {result[key].dtype}"

    # Value ranges (0-1 for alpha/fg; comp/processed may slightly exceed due to sRGB conversion)
    for key in ("alpha", "fg"):
        assert result[key].min() >= 0.0, f"{key} has negative values"
        assert result[key].max() <= 1.0, f"{key} exceeds 1.0"
    for key in ("comp", "processed"):
        assert result[key].min() >= 0.0, f"{key} has negative values"
