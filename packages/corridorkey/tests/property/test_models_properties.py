"""Property-based tests for models.py."""

from __future__ import annotations

from corridorkey.models import InOutRange
from hypothesis import given
from hypothesis import strategies as st


@given(
    st.integers(min_value=0, max_value=10000),
    st.integers(min_value=0, max_value=10000),
)
def test_frame_count_always_positive(in_point: int, out_point: int) -> None:
    """frame_count must be >= 1 when out_point >= in_point."""
    if out_point < in_point:
        return
    r = InOutRange(in_point=in_point, out_point=out_point)
    assert r.frame_count >= 1


@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
)
def test_to_dict_from_dict_roundtrip(in_point: int, out_point: int) -> None:
    """Serialise/deserialise must be lossless."""
    original = InOutRange(in_point=in_point, out_point=out_point)
    restored = InOutRange.from_dict(original.to_dict())
    assert restored.in_point == original.in_point
    assert restored.out_point == original.out_point


@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
)
def test_contains_consistent_with_range(in_point: int, out_point: int, index: int) -> None:
    """contains() must agree with Python's range membership."""
    r = InOutRange(in_point=in_point, out_point=out_point)
    expected = in_point <= index <= out_point
    assert r.contains(index) == expected
