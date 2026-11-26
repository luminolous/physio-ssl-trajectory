"""Window effect of residual_r_specific and the window-matched diagnostic (Head-floor rrs check)."""

from __future__ import annotations

import numpy as np
import pytest

from eegtraj.metrics import pearson_r, residual_r, residual_r_matched, residual_r_other_trial


def trials(n: int = 40, seed: int = 0, movement_window: bool = True):
    """Lifts with variable onset on a common time grid since the cue; the window follows the movement, or is fixed."""
    rng = np.random.default_rng(seed)
    y, ids, bins = [], [], []
    for i in range(n):
        onset = rng.integers(40, 110)
        lo, hi = (onset - 30, onset + 200) if movement_window else (0, 300)
        b = np.arange(lo, hi)
        u = np.clip((b - onset) / 150, 0, 1)
        y.append(np.stack([10 * u, 5 * np.sin(np.pi * u), 20 * u ** 2], axis=1) * rng.uniform(0.8, 1.2))
        ids.append(np.full(b.size, i))
        bins.append(b)
    return np.concatenate(y), np.concatenate(ids), np.concatenate(bins)


def cue(bins: np.ndarray) -> np.ndarray:
    u = np.clip((bins - 75) / 170, 0, 1)
    return np.stack([10 * u, 5 * np.sin(np.pi * u), 20 * u ** 2], axis=1)


def rigid(bins: np.ndarray) -> np.ndarray:
    u = np.clip((bins - 60) / 200, 0, 1)
    return np.stack([9 * u, 4 * np.sin(np.pi * u), 22 * u ** 1.5], axis=1)


def rrs(y, p, t, ids, bins) -> float:
    return (residual_r(y, p, t, ids)["mean"]
            - residual_r_other_trial(y, p, t, ids, bins, np.random.default_rng(1), 5)["mean"])


def test_rigid_prediction_matched_is_zero_on_movement_windows():
    y, ids, bins = trials()
    out = residual_r_matched(y, rigid(bins), cue(bins), ids, bins, np.random.default_rng(1))
    assert out["n_pairs"] > 0 and 0 < out["shared_fraction"] < 1
    assert abs(out["matched"]) < 1e-12


def test_rigid_prediction_rrs_zero_on_fixed_window_only():
    y, ids, bins = trials(movement_window=False)
    assert abs(rrs(y, rigid(bins), cue(bins), ids, bins)) < 1e-12
    y, ids, bins = trials(movement_window=True)
    assert abs(rrs(y, rigid(bins), cue(bins), ids, bins)) > 1e-3


def test_matched_keeps_trial_specific_information():
    y, ids, bins = trials()
    perfect = residual_r_matched(y, y, cue(bins), ids, bins, np.random.default_rng(1))
    assert perfect["own"] == pytest.approx(1.0) and perfect["matched"] > 0.3


def test_matched_equals_brute_force_and_other_trial_pairs():
    y, ids, bins = trials(n=6)
    rng = np.random.default_rng(3)
    p = rigid(bins) + rng.normal(size=y.shape)
    t = cue(bins)
    out = residual_r_matched(y, p, t, ids, bins, np.random.default_rng(5), 2)
    own, cross, forward = [], [], []
    g = np.random.default_rng(5)
    for _ in range(2):
        order = g.permutation(6)
        for a, b in zip(order, np.roll(order, -1)):
            ma, mb = ids == a, ids == b
            shared = np.intersect1d(bins[ma], bins[mb])
            if shared.size < 50:
                continue
            sa, sb = ma & np.isin(bins, shared), mb & np.isin(bins, shared)
            for i, j in ((sa, sb), (sb, sa)):
                own.append([pearson_r((y - t)[i, c], (p - t)[i, c]) for c in range(3)])
                cross.append([pearson_r((y - t)[j, c], (p - t)[i, c]) for c in range(3)])
            forward.append(cross[-2])
    assert out["own"] == pytest.approx(np.mean(own)) and out["cross"] == pytest.approx(np.mean(cross))
    assert out["n_pairs"] == len(own)
    other = residual_r_other_trial(y, p, t, ids, bins, np.random.default_rng(5), 2)["mean"]
    assert np.mean(forward) == pytest.approx(other)


def test_constant_residual_is_skipped():
    y, ids, bins = trials(n=5)
    t = cue(bins)
    out = residual_r_matched(y, t, t, ids, bins, np.random.default_rng(0))
    assert np.isnan(out["matched"])
