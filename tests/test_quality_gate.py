"""Quality gate tests for CorridorKey optimization phases.

These tests compare current inference outputs against pre-computed baseline
.npy files to catch regressions in output quality. They require a GPU and
model weights, so they're marked @pytest.mark.gpu and skipped in CI.

Run locally before merging each optimization phase:
    uv run pytest tests/test_quality_gate.py -v

Baseline must be generated first:
    uv run python benchmarks/bench_phase.py --generate-baseline --clip <path> --alpha <path>

Quality thresholds (per plan):
    lossless (bit-exact):  max_err < 1e-4, MAE < 1e-5, PSNR > 80 dB
    fp16 (Phase 1-2):      max_err < 0.04, MAE < 1e-4, PSNR > 75 dB
    lossy (Phase 3-4):     max_err < 0.02, MAE < 0.005, PSNR > 40 dB

Default thresholds are set to fp16. Override via env var QUALITY_GATE_PHASE
(set to "lossless" or "lossy" as needed).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "baseline")

# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

LOSSLESS_THRESHOLDS = {
    "max_abs_err": 1e-4,
    "mae": 1e-5,
    "psnr_min_db": 80.0,
}

# FP16 weight casting is "practically lossless" but not bit-exact.
# Max errors come from FP16 rounding in low-magnitude regions.
FP16_THRESHOLDS = {
    "max_abs_err": 0.04,
    "mae": 1e-4,
    "psnr_min_db": 75.0,
}

LOSSY_THRESHOLDS = {
    "max_abs_err": 0.06,  # Higher than fp16 (0.04) — includes backbone downsampling + tile seam artifacts
    "mae": 0.005,
    "psnr_min_db": 40.0,
}


def _get_thresholds() -> dict:
    phase = os.environ.get("QUALITY_GATE_PHASE", "fp16").lower()
    if phase == "lossy":
        return LOSSY_THRESHOLDS
    if phase == "lossless":
        return LOSSLESS_THRESHOLDS
    return FP16_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _psnr(baseline: np.ndarray, result: np.ndarray) -> float:
    mse = float(np.mean((baseline.astype(np.float64) - result.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _abs_diff(baseline: np.ndarray, result: np.ndarray) -> np.ndarray:
    return np.abs(baseline.astype(np.float64) - result.astype(np.float64))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _baseline_available() -> bool:
    """Check if baseline .npy files exist."""
    return os.path.isfile(os.path.join(BASELINE_DIR, "frame_001_alpha.npy"))


def _gpu_available() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except ImportError:
        pass
    return False


skip_no_baseline = pytest.mark.skipif(not _baseline_available(), reason="No baseline .npy files found")
skip_no_gpu = pytest.mark.skipif(not _gpu_available(), reason="No GPU available")


@pytest.fixture(scope="session")
def baseline_outputs():
    """Load pre-computed baseline .npy files."""
    if not _baseline_available():
        pytest.skip("Baseline not generated yet")

    outputs = []
    i = 0
    while True:
        i += 1
        frame_id = f"frame_{i:03d}"
        alpha_path = os.path.join(BASELINE_DIR, f"{frame_id}_alpha.npy")
        if not os.path.exists(alpha_path):
            break
        outputs.append(
            {
                "alpha": np.load(os.path.join(BASELINE_DIR, f"{frame_id}_alpha.npy")),
                "fg": np.load(os.path.join(BASELINE_DIR, f"{frame_id}_fg.npy")),
                "processed": np.load(os.path.join(BASELINE_DIR, f"{frame_id}_processed.npy")),
                "comp": np.load(os.path.join(BASELINE_DIR, f"{frame_id}_comp.npy")),
            }
        )
    return outputs


@pytest.fixture(scope="session")
def current_outputs(baseline_outputs):
    """Run inference on the same reference frames with current code.

    Uses the same input frames that generated the baseline. Since we saved
    outputs but not inputs, we re-run inference from the reference clip.
    The clip paths come from env vars or the plan's default location.
    """
    clip_path = os.environ.get(
        "BENCH_CLIP",
        os.path.join(PROJECT_ROOT, "ClipsForInference", "BetterGreenScreenTest_BASE", "Input.mp4"),
    )
    alpha_path = os.environ.get(
        "BENCH_ALPHA",
        os.path.join(
            PROJECT_ROOT,
            "ClipsForInference",
            "BetterGreenScreenTest_BASE",
            "AlphaHint",
            "BetterGreenScreenTest_MASK.mp4",
        ),
    )

    if not os.path.isfile(clip_path) or not os.path.isfile(alpha_path):
        pytest.skip(f"Reference clip not found: {clip_path} / {alpha_path}")

    # Add project root to import path
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from benchmarks.bench_phase import create_engine, get_device, load_mask_frames, load_video_frames, run_benchmark

    device = get_device()
    engine = create_engine(device)

    num_baseline_frames = len(baseline_outputs)
    frames = load_video_frames(clip_path, max_frames=num_baseline_frames)
    masks = load_mask_frames(alpha_path, max_frames=num_baseline_frames)

    num_frames = min(len(frames), len(masks), num_baseline_frames)
    frames = frames[:num_frames]
    masks = masks[:num_frames]

    outputs, _, _ = run_benchmark(engine, frames, masks, device)
    return outputs


# ---------------------------------------------------------------------------
# Quality gate tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@skip_no_baseline
class TestAlphaQuality:
    """Alpha channel quality gates."""

    def test_alpha_max_error(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            diff = _abs_diff(base["alpha"], curr["alpha"])
            max_err = float(diff.max())
            assert max_err < thresholds["max_abs_err"], (
                f"Frame {i + 1} alpha max error {max_err:.6f} exceeds {thresholds['max_abs_err']}"
            )

    def test_alpha_mae(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            diff = _abs_diff(base["alpha"], curr["alpha"])
            mae = float(diff.mean())
            assert mae < thresholds["mae"], f"Frame {i + 1} alpha MAE {mae:.6f} exceeds {thresholds['mae']}"

    def test_alpha_psnr(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            psnr = _psnr(base["alpha"], curr["alpha"])
            assert psnr > thresholds["psnr_min_db"], (
                f"Frame {i + 1} alpha PSNR {psnr:.1f} dB below {thresholds['psnr_min_db']} dB"
            )


@pytest.mark.gpu
@skip_no_baseline
class TestFGQuality:
    """Foreground color quality gates (per-channel)."""

    def test_fg_max_error(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            diff = _abs_diff(base["fg"], curr["fg"])
            max_err = float(diff.max())
            assert max_err < thresholds["max_abs_err"], (
                f"Frame {i + 1} FG max error {max_err:.6f} exceeds {thresholds['max_abs_err']}"
            )

    def test_fg_mae(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            diff = _abs_diff(base["fg"], curr["fg"])
            mae = float(diff.mean())
            assert mae < thresholds["mae"], f"Frame {i + 1} FG MAE {mae:.6f} exceeds {thresholds['mae']}"

    def test_fg_psnr(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            psnr = _psnr(base["fg"], curr["fg"])
            assert psnr > thresholds["psnr_min_db"], (
                f"Frame {i + 1} FG PSNR {psnr:.1f} dB below {thresholds['psnr_min_db']} dB"
            )


@pytest.mark.gpu
@skip_no_baseline
class TestProcessedRGBAQuality:
    """Full pipeline RGBA output quality gates."""

    def test_processed_max_error(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            diff = _abs_diff(base["processed"], curr["processed"])
            max_err = float(diff.max())
            assert max_err < thresholds["max_abs_err"], (
                f"Frame {i + 1} processed RGBA max error {max_err:.6f} exceeds {thresholds['max_abs_err']}"
            )

    def test_processed_psnr(self, baseline_outputs, current_outputs):
        thresholds = _get_thresholds()
        for i, (base, curr) in enumerate(zip(baseline_outputs, current_outputs, strict=True)):
            psnr = _psnr(base["processed"], curr["processed"])
            assert psnr > thresholds["psnr_min_db"], (
                f"Frame {i + 1} processed RGBA PSNR {psnr:.1f} dB below {thresholds['psnr_min_db']} dB"
            )


# ---------------------------------------------------------------------------
# Integrity tests (no baseline needed — run on current outputs alone)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@skip_no_baseline
class TestOutputIntegrity:
    """Sanity checks that don't need baseline comparison."""

    def test_no_nan_or_inf(self, current_outputs):
        for i, result in enumerate(current_outputs):
            for key in ("alpha", "fg", "processed", "comp"):
                arr = result[key]
                assert np.isfinite(arr).all(), f"Frame {i + 1} {key} contains NaN or Inf"

    def test_color_space_ranges(self, current_outputs):
        """FG in valid sRGB [0,1]. Alpha in linear [0,1].

        Lanczos4 resampling produces ringing artifacts that overshoot [0,1].
        Model outputs are strictly [0,1] at 2048x2048 but the resize back
        to original resolution via INTER_LANCZOS4 introduces overshoots
        up to ~0.08. This is expected and handled downstream by clipping.
        """
        lanczos_tolerance = 0.1
        for i, result in enumerate(current_outputs):
            alpha = result["alpha"]
            assert alpha.min() >= -lanczos_tolerance, f"Frame {i + 1} alpha below 0: {alpha.min()}"
            assert alpha.max() <= 1.0 + lanczos_tolerance, f"Frame {i + 1} alpha above 1: {alpha.max()}"

            fg = result["fg"]
            assert fg.min() >= -lanczos_tolerance, f"Frame {i + 1} FG below 0: {fg.min()}"
            assert fg.max() <= 1.0 + lanczos_tolerance, f"Frame {i + 1} FG above 1: {fg.max()}"


# ---------------------------------------------------------------------------
# Lightweight CPU smoke test (runs in CI — no GPU, no baseline, no weights)
# ---------------------------------------------------------------------------


class TestTiledRefiner:
    """Verify tiled refiner produces same output as single-pass refiner."""

    @staticmethod
    def _make_tiling_harness(refiner, tile_size, overlap):
        """Create a minimal object with _tiled_refine capability."""
        from CorridorKeyModule.core.model_transformer import GreenFormer

        class _TileHarness:
            pass

        h = _TileHarness()
        h.refiner = refiner
        h.refiner_tile_size = tile_size
        h.refiner_tile_overlap = overlap
        h._tent_weight = GreenFormer._build_tent_weight(tile_size, overlap)
        h._tiled_refine = GreenFormer._tiled_refine.__get__(h)
        return h

    def test_tiled_vs_single_pass(self):
        """Tiled refiner with tent blending matches single-pass output closely."""
        import torch

        from CorridorKeyModule.core.model_transformer import CNNRefinerModule

        tile_size = 32
        overlap = 8
        img_size = 64

        torch.manual_seed(42)
        refiner = CNNRefinerModule(in_channels=7, hidden_channels=16, out_channels=4)
        refiner.eval()

        rgb = torch.randn(1, 3, img_size, img_size)
        coarse = torch.randn(1, 4, img_size, img_size)

        with torch.no_grad():
            ref = refiner(rgb, coarse)

        harness = self._make_tiling_harness(refiner, tile_size, overlap)

        with torch.no_grad():
            tiled = harness._tiled_refine(rgb, coarse)

        diff = (ref - tiled).abs()
        assert diff.max().item() < 0.5, f"Max diff {diff.max().item():.4f} too large"
        assert diff.mean().item() < 0.05, f"Mean diff {diff.mean().item():.6f} too large"

    def test_tiled_no_nan(self):
        """Tiled refiner produces no NaN/Inf."""
        import torch

        from CorridorKeyModule.core.model_transformer import CNNRefinerModule

        torch.manual_seed(123)
        refiner = CNNRefinerModule(in_channels=7, hidden_channels=16, out_channels=4)
        refiner.eval()

        harness = self._make_tiling_harness(refiner, 32, 8)

        rgb = torch.randn(1, 3, 96, 96)
        coarse = torch.randn(1, 4, 96, 96)

        with torch.no_grad():
            result = harness._tiled_refine(rgb, coarse)

        assert torch.isfinite(result).all(), "Tiled refiner produced NaN/Inf"

    def test_tent_weight_shape(self):
        """Tent weight has correct shape and valid range."""
        from CorridorKeyModule.core.model_transformer import GreenFormer

        tent = GreenFormer._build_tent_weight(512, 64)
        assert tent.shape == (1, 1, 512, 512)
        assert tent.min() > 0, "Tent weight should be strictly positive"
        assert tent.max() <= 1.0, "Tent weight should not exceed 1.0"
        assert tent[0, 0, 256, 256] == 1.0


class TestSmokeNoBoom:
    """Minimal sanity check with synthetic input. No GPU or model weights."""

    def test_no_nan_inf_synthetic(self):
        """Verify color_utils functions produce finite output on random data."""
        from CorridorKeyModule.core import color_utils as cu

        rng = np.random.default_rng(42)
        img = rng.random((32, 32, 3), dtype=np.float32)
        alpha = rng.random((32, 32, 1), dtype=np.float32)

        fg_lin = cu.srgb_to_linear(img)
        assert np.isfinite(fg_lin).all(), "srgb_to_linear produced NaN/Inf"

        fg_srgb = cu.linear_to_srgb(fg_lin)
        assert np.isfinite(fg_srgb).all(), "linear_to_srgb produced NaN/Inf"

        premul = cu.premultiply(fg_lin, alpha)
        assert np.isfinite(premul).all(), "premultiply produced NaN/Inf"

        bg = cu.create_checkerboard(32, 32)
        comp = cu.composite_straight(img, bg, alpha)
        assert np.isfinite(comp).all(), "composite_straight produced NaN/Inf"

    def test_values_in_range_synthetic(self):
        """sRGB<->linear roundtrip stays in [0,1]."""
        from CorridorKeyModule.core import color_utils as cu

        rng = np.random.default_rng(99)
        img = rng.random((16, 16, 3), dtype=np.float32)
        roundtrip = cu.linear_to_srgb(cu.srgb_to_linear(img))
        np.testing.assert_allclose(roundtrip, img, atol=1e-5)
