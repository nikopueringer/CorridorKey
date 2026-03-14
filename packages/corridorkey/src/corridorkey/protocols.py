"""Protocols for pluggable components in the CorridorKey pipeline.

Any package that implements these protocols can be used as a drop-in
component without modifying corridorkey itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from corridorkey.clip_state import ClipEntry


@runtime_checkable
class AlphaGenerator(Protocol):
    """Protocol for alpha hint generators (GVM, VideoMaMa, MatAnyone2, etc.).

    Implementors receive a clip in RAW or MASKED state and write alpha hint
    frames into the clip's AlphaHint/ directory, then transition the clip
    to READY state.

    Example implementation (corridorkey-gbm):
        class GBMAlphaGenerator:
            def generate(self, clip, on_progress=None, on_warning=None): ...
            @property
            def name(self) -> str: return "gbm"
    """

    @property
    def name(self) -> str:
        """Human-readable generator name (e.g. "gvm", "videomama", "matanyone2")."""
        ...

    def generate(
        self,
        clip: ClipEntry,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        """Generate alpha hints for the given clip.

        Writes frames into clip.root_path/AlphaHint/ and transitions
        clip state to READY on success.

        Args:
            clip: Clip in RAW or MASKED state with a valid input_asset.
            on_progress: Called with (clip_name, current, total).
            on_warning: Called with non-fatal warning messages.

        Raises:
            CorridorKeyError: On unrecoverable failure.
        """
        ...
