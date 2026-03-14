"""Natural sort key for frame filenames.

Handles non-zero-padded frame numbers correctly:
  frame_1, frame_2, frame_10  (not frame_1, frame_10, frame_2)

No external dependency - pure Python implementation.
"""

from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"(\d+)")


def natural_sort_key(text: str) -> list[str | int]:
    """Return a sort key that orders numeric substrings numerically.

    >>> sorted(["f_1", "f_10", "f_2"], key=natural_sort_key)
    ['f_1', 'f_2', 'f_10']
    """
    parts: list[str | int] = []
    for chunk in _SPLIT_RE.split(text):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return parts


def natsorted(items: list[str]) -> list[str]:
    """Return a naturally sorted copy of a list of strings."""
    return sorted(items, key=natural_sort_key)
