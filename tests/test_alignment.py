"""Tests for Euclidean alignment (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegtraj.data.windows import SubjectSamples
from eegtraj.preprocessing.alignment import euclidean_align, lift_bounds, mean_trial_covariance
from eegtraj.preprocessing.reference import common_average


def subject(eeg: np.ndarray, n_lifts: int = 6, n_pre: int = 20) -> SubjectSamples:
    n = eeg.shape[1] // n_lifts
    starts = np.arange(n_lifts) * n + n_pre
    trials = pd.DataFrame({"offset_start": starts, "offset_end": (np.arange(n_lifts) + 1) * n})
    return SubjectSamples("P1", eeg, np.zeros((3, eeg.shape[1]), np.float32), starts, np.arange(n_lifts), trials, 100.0)


def mixed(rng: np.random.Generator, c: int = 8, t: int = 6000) -> np.ndarray:
    return (rng.standard_normal((c, c)) @ rng.standard_normal((c, t)) * 20 + 5).astype(np.float32)


def test_aligned_mean_covariance_is_identity():
    s = subject(mixed(np.random.default_rng(0)))
    aligned, info = euclidean_align(s)
    cov = mean_trial_covariance(aligned.eeg, *lift_bounds(aligned))
    assert info["rank"] == 8
    assert np.allclose(cov, np.eye(8), atol=1e-4)


def test_after_car_the_aligned_covariance_is_the_projector_off_the_common_mode():
    s = subject(common_average(mixed(np.random.default_rng(1))))
    aligned, info = euclidean_align(s)
    cov = mean_trial_covariance(aligned.eeg, *lift_bounds(aligned))
    assert info["rank"] == 7
    assert np.allclose(cov, np.eye(8) - np.full((8, 8), 1 / 8), atol=1e-4)


def test_covariance_uses_lift_samples_only_and_alignment_covers_pre_lift_eeg():
    eeg = mixed(np.random.default_rng(2))
    s = subject(eeg)
    noisy = eeg.copy()
    noisy[:, :20] *= 1000  # pre-lift EEG of lift 0
    t = subject(noisy)
    assert np.allclose(mean_trial_covariance(s.eeg, *lift_bounds(s)), mean_trial_covariance(t.eeg, *lift_bounds(t)))
    aligned, _ = euclidean_align(t)
    assert np.abs(aligned.eeg[:, :20]).max() > 100 * np.abs(aligned.eeg[:, 20:1000]).std()
    assert aligned.eeg.shape == s.eeg.shape and np.array_equal(aligned.sample_idx, s.sample_idx)


def test_alignment_removes_a_subject_specific_mixing():
    rng = np.random.default_rng(3)
    source = rng.standard_normal((8, 6000)).astype(np.float32)
    a = euclidean_align(subject(source))[0].eeg
    mix = rng.standard_normal((8, 8)).astype(np.float32)
    b = euclidean_align(subject(mix @ source))[0].eeg
    # Both end up white; the remaining difference is a rotation, so second-order statistics agree.
    assert np.allclose(mean_trial_covariance(a, *lift_bounds(subject(a))), mean_trial_covariance(b, *lift_bounds(subject(b))),
                       atol=1e-4)
    assert not np.allclose(a, b, atol=1e-2)
    assert pytest.approx(float(np.linalg.norm(a)), rel=1e-3) == float(np.linalg.norm(b))
