"""Unit tests for natural_sort.py."""

from __future__ import annotations

from corridorkey.natural_sort import natsorted, natural_sort_key


class TestNaturalSortKey:
    def test_numeric_order(self):
        items = ["frame_10", "frame_2", "frame_1"]
        assert sorted(items, key=natural_sort_key) == ["frame_1", "frame_2", "frame_10"]

    def test_zero_padded_and_unpadded_mixed(self):
        items = ["frame_002", "frame_10", "frame_1"]
        result = sorted(items, key=natural_sort_key)
        assert result == ["frame_1", "frame_002", "frame_10"]

    def test_no_digits_falls_back_to_alpha(self):
        items = ["charlie", "alpha", "bravo"]
        assert sorted(items, key=natural_sort_key) == ["alpha", "bravo", "charlie"]

    def test_case_insensitive(self):
        items = ["Frame_2", "frame_1"]
        assert sorted(items, key=natural_sort_key) == ["frame_1", "Frame_2"]

    def test_empty_string(self):
        assert natural_sort_key("") == [""]

    def test_digits_only(self):
        assert natural_sort_key("42") == ["", 42, ""]

    def test_mixed_prefix(self):
        items = ["shot_10_v2", "shot_2_v10", "shot_2_v2"]
        result = sorted(items, key=natural_sort_key)
        assert result == ["shot_2_v2", "shot_2_v10", "shot_10_v2"]


class TestNatsorted:
    def test_returns_sorted_copy(self):
        original = ["f_10", "f_2", "f_1"]
        result = natsorted(original)
        assert result == ["f_1", "f_2", "f_10"]
        assert original == ["f_10", "f_2", "f_1"]  # original unchanged

    def test_empty_list(self):
        assert natsorted([]) == []

    def test_single_item(self):
        assert natsorted(["only"]) == ["only"]

    def test_already_sorted(self):
        items = ["a_1", "a_2", "a_3"]
        assert natsorted(items) == items
