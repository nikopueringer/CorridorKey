"""Tests documenting the gamma 2.2 vs piecewise sRGB inconsistency.

STATUS: This test documents a KNOWN INCONSISTENCY, not desired behavior.

The codebase uses two different methods to convert between linear and sRGB:

1. **Piecewise sRGB (correct)** — used by ``color_utils.linear_to_srgb()``
   and ``color_utils.srgb_to_linear()``, called from ``inference_engine.py``.
   This follows the IEC 61966-2-1 spec: exponent 2.4 with a linear segment
   below the breakpoint.

2. **Gamma 2.2 approximation** — used by ``clip_manager.py:383`` (VideoMaMa
   frame loading) and ``gvm_core/gvm/utils/inference_utils.py:124`` (GVM
   frame loading).  This uses a simple ``x ** (1/2.2)`` power curve.

The two methods produce visibly different results, especially in darks.
At linear 0.01, the difference is ~4.7% — enough to see in a waveform monitor.

**Why this hasn't been fixed yet:**
The gamma 2.2 paths feed data into third-party models (VideoMaMa, GVM) that
were likely *trained* on gamma 2.2 converted data.  Switching to piecewise
sRGB might degrade their output quality.  Verifying this requires running
inference with model weights, which isn't feasible in automated tests.

**If you fix one path, fix the other too** — or these tests will tell you
something changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from CorridorKeyModule.core import color_utils as cu

# ---------------------------------------------------------------------------
# Document the divergence
# ---------------------------------------------------------------------------


class TestGammaInconsistency:
    """These tests assert that the two conversion methods produce DIFFERENT
    results — documenting the inconsistency so it isn't accidentally
    "fixed" in only one place.
    """

    def test_linear_to_srgb_differs_from_gamma_22(self):
        """Piecewise sRGB and gamma 2.2 must produce different results.

        If this test fails, someone unified the conversion — which is good!
        Remove this test and update the docstring above.
        """
        # Test at a value where the difference is significant
        linear_val = np.float32(0.1)

        piecewise_srgb = float(cu.linear_to_srgb(linear_val))
        gamma_22 = float(linear_val ** (1.0 / 2.2))

        # They should NOT be equal
        assert piecewise_srgb != pytest.approx(
            gamma_22, abs=1e-4
        ), "Piecewise sRGB and gamma 2.2 now match — has the inconsistency been fixed? If so, remove this test."

    def test_srgb_to_linear_differs_from_gamma_22(self):
        """Inverse direction: srgb_to_linear vs x**2.2."""
        srgb_val = np.float32(0.5)

        piecewise_linear = float(cu.srgb_to_linear(srgb_val))
        gamma_22_linear = float(srgb_val**2.2)

        assert piecewise_linear != pytest.approx(gamma_22_linear, abs=1e-4), (
            "Piecewise sRGB and gamma 2.2 inverse now match — has the "
            "inconsistency been fixed? If so, remove this test."
        )


# ---------------------------------------------------------------------------
# Quantify the divergence
# ---------------------------------------------------------------------------


class TestGammaDivergenceMagnitude:
    """Quantify how far apart the two methods are at various values.

    These tests serve as documentation — if someone changes the color math,
    these show exactly where the drift happens and by how much.
    """

    @pytest.mark.parametrize(
        "linear_val,expected_min_diff",
        [
            # In darks, the piecewise linear segment causes the biggest gap.
            # At linear 0.001, piecewise sRGB uses the linear segment (x*12.92)
            # while gamma 2.2 uses x^(1/2.2) — very different behavior.
            (0.001, 0.005),
            # Mid-darks: still a measurable gap
            (0.01, 0.01),
            # Mid-tones: smaller but still measurable
            (0.1, 0.001),
            # Highlights: converges as both curves approach 1.0
            (0.5, 0.001),
        ],
    )
    def test_divergence_at_known_values(self, linear_val, expected_min_diff):
        """The two methods should differ by at least expected_min_diff."""
        x = np.float32(linear_val)
        piecewise = float(cu.linear_to_srgb(x))
        gamma_22 = float(x ** (1.0 / 2.2))

        diff = abs(piecewise - gamma_22)
        assert diff >= expected_min_diff, (
            f"At linear={linear_val}: piecewise={piecewise:.6f}, "
            f"gamma2.2={gamma_22:.6f}, diff={diff:.6f} "
            f"(expected >= {expected_min_diff})"
        )

    def test_both_methods_agree_at_zero_and_one(self):
        """At the endpoints 0.0 and 1.0, both methods agree exactly."""
        for val in [0.0, 1.0]:
            x = np.float32(val)
            piecewise = float(cu.linear_to_srgb(x))
            gamma_22 = float(x ** (1.0 / 2.2))
            assert piecewise == pytest.approx(gamma_22, abs=1e-6), f"At {val}, both methods should agree"

    def test_worst_case_divergence_in_darks(self):
        """Document the worst-case divergence across the 0-1 range.

        This is informational — the exact value may shift if someone tweaks
        tolerances, but the magnitude should stay in the ballpark.
        """
        values = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        piecewise = cu.linear_to_srgb(values).astype(np.float64)
        gamma_22 = (values.astype(np.float64)) ** (1.0 / 2.2)

        max_diff = float(np.max(np.abs(piecewise - gamma_22)))

        # The worst-case difference should be in the low-mid range (~0.01-0.04)
        # and definitely not zero (that would mean the inconsistency is gone)
        assert max_diff > 0.01, "Expected significant divergence in darks"
        assert max_diff < 0.10, "Divergence larger than expected — check math"
