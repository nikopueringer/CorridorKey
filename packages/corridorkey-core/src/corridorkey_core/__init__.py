"""CorridorKey core library for AI-powered chroma keying.

Exposes the inference engine and post-processing utilities needed to run
the GreenFormer model and process its outputs. This package has no
filesystem, pipeline, or UI dependencies and can be embedded in any workflow.
"""

from corridorkey_core.compositing import (
    clean_matte,
    composite_premul,
    composite_straight,
    despill,
    linear_to_srgb,
    premultiply,
    srgb_to_linear,
)
from corridorkey_core.inference_engine import CorridorKeyEngine

__all__ = [
    "CorridorKeyEngine",
    "clean_matte",
    "composite_premul",
    "composite_straight",
    "despill",
    "linear_to_srgb",
    "premultiply",
    "srgb_to_linear",
]
