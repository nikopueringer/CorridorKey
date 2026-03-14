"""Property-based tests for natural_sort.py."""

from __future__ import annotations

from corridorkey.natural_sort import natsorted, natural_sort_key
from hypothesis import given
from hypothesis import strategies as st


@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20), max_size=30))
def test_natsorted_is_stable_permutation(items: list[str]) -> None:
    """natsorted must return the same elements, just reordered."""
    result = natsorted(items)
    assert sorted(result) == sorted(items)
    assert len(result) == len(items)


@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20), max_size=30))
def test_natsorted_idempotent(items: list[str]) -> None:
    """Sorting an already-sorted list must not change it."""
    once = natsorted(items)
    twice = natsorted(once)
    assert once == twice


@given(
    st.integers(min_value=0, max_value=9999),
    st.integers(min_value=0, max_value=9999),
)
def test_numeric_order_respected(a: int, b: int) -> None:
    """frame_{a} must sort before frame_{b} when a < b."""
    if a == b:
        return
    lo, hi = (a, b) if a < b else (b, a)
    items = [f"frame_{hi}", f"frame_{lo}"]
    result = natsorted(items)
    assert result == [f"frame_{lo}", f"frame_{hi}"]


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=50))
def test_sort_key_returns_list(text: str) -> None:
    """natural_sort_key must always return a list."""
    key = natural_sort_key(text)
    assert isinstance(key, list)
