"""CorridorKey core library for AI-powered chroma keying.

This package has no filesystem, pipeline, or UI dependencies and can be
embedded in any workflow.

Public API:
    create_engine(checkpoint_dir, ...) -> engine
        Returns an engine with process_frame() matching the Torch output
        contract, regardless of whether Torch or MLX is running underneath.
"""

from corridorkey_core.engine_factory import create_engine

__all__ = ["create_engine"]
