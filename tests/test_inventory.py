"""Tests for the unlabeled data inventory helpers."""

from __future__ import annotations

from eegtraj.diagnostics.inventory import gaps, merge, n_windows


def test_merge_joins_overlapping_and_touching_intervals():
    assert merge([(5, 8), (0, 3), (2, 4), (8, 10)]) == [(0, 4), (5, 10)]


def test_gaps_are_the_complement_within_the_range():
    assert gaps([(2, 4), (3, 6), (8, 9)], 0, 10) == [(0, 2), (6, 8), (9, 10)]
    assert gaps([(0, 10)], 0, 10) == []
    assert gaps([], 0, 5) == [(0, 5)]


def test_windows_stay_inside_each_segment():
    assert n_windows([200], 200, 50) == 1
    assert n_windows([449, 199, 250], 200, 50) == 5 + 0 + 2  # (449-200)//50+1, too short, (250-200)//50+1
    assert n_windows([], 200, 50) == 0
