"""Unit tests for models.py."""

from __future__ import annotations

import pytest
from corridorkey.models import InOutRange


class TestInOutRange:
    def test_frame_count_inclusive(self):
        r = InOutRange(in_point=0, out_point=9)
        assert r.frame_count == 10

    def test_frame_count_single_frame(self):
        r = InOutRange(in_point=5, out_point=5)
        assert r.frame_count == 1

    def test_contains_in_range(self):
        r = InOutRange(in_point=10, out_point=20)
        assert r.contains(10)
        assert r.contains(15)
        assert r.contains(20)

    def test_contains_out_of_range(self):
        r = InOutRange(in_point=10, out_point=20)
        assert not r.contains(9)
        assert not r.contains(21)

    def test_to_dict(self):
        r = InOutRange(in_point=3, out_point=7)
        assert r.to_dict() == {"in_point": 3, "out_point": 7}

    def test_from_dict_roundtrip(self):
        original = InOutRange(in_point=3, out_point=7)
        restored = InOutRange.from_dict(original.to_dict())
        assert restored.in_point == original.in_point
        assert restored.out_point == original.out_point

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(KeyError):
            InOutRange.from_dict({"in_point": 0})
