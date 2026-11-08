"""Inventory of EEG available without labels: lift windows, between-lift parts of the HS recordings, and ST.

Durations come from the raw HS files (EEG sample counts). Lift windows are the cached lifts, mapped back to raw
samples with ``hs_start_sample`` and the 100 Hz grid; overlapping lifts are merged before subtracting them from
the series to get the between-lift parts.
"""

from __future__ import annotations

import numpy as np


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of half-open intervals [start, end), sorted."""
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def gaps(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """Parts of [start, end) not covered by ``intervals`` (merged first, clipped to the range)."""
    out, pos = [], start
    for a, b in merge(intervals):
        a, b = max(a, start), min(b, end)
        if a > pos:
            out.append((pos, a))
        pos = max(pos, b)
    if pos < end:
        out.append((pos, end))
    return out


def n_windows(lengths: np.ndarray, window: int, stride: int) -> int:
    """Number of full windows of ``window`` samples at ``stride`` inside each contiguous segment, summed."""
    lengths = np.asarray(lengths, dtype=np.int64)
    fit = lengths >= window
    return int(((lengths[fit] - window) // stride + 1).sum())
