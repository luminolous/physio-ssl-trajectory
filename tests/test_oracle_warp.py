"""Tests for the oracle-warp ceiling (eegtraj.diagnostics.oracle_warp)."""

import numpy as np
import pytest

from eegtraj.diagnostics.oracle_warp import (
    Bounds,
    FitSettings,
    Template,
    TrialData,
    fit_all,
    fit_min_jerk,
    fit_piecewise,
    fit_shift_scale,
    min_jerk_basis,
    piecewise_warp,
    shift_scale_warp,
    solve_amp_offset,
)

FS = 100.0


def smooth_template() -> Template:
    """A reach-hold-return-like 3-axis shape from -0.5 s to 7 s after LED-on."""
    bin0 = -50
    t = (bin0 + np.arange(751)) / FS
    reach = 1 / (1 + np.exp(-(t - 1.2) * 6))
    back = 1 / (1 + np.exp(-(t - 5.0) * 6))
    lift = np.exp(-((t - 3.0) / 0.8) ** 2)
    values = np.stack([20 * (reach - back), 10 * (reach - back) + 3 * lift, 8 * lift + 2 * reach])
    landmarks = np.array([0.4, 1.6, 2.1, 4.8, 5.8])
    return Template(values.astype(np.float32), bin0, FS, landmarks)


def trial_times(start_s: float = -0.1, stop_s: float = 6.0) -> np.ndarray:
    return np.round(np.arange(start_s, stop_s, 1 / FS) * FS) / FS


def test_template_on_grid_returns_bin_values():
    tmpl = smooth_template()
    t = (tmpl.bin0 + np.arange(tmpl.values.shape[1])) / FS
    np.testing.assert_array_equal(tmpl(t), tmpl.values.T.astype(np.float64))


def test_identity_parameters_reproduce_template_exactly():
    tmpl = smooth_template()
    t = trial_times()
    expected = tmpl(t)
    np.testing.assert_array_equal(tmpl(shift_scale_warp(t, 0.0, 1.0, tmpl.onset_s)), expected)
    np.testing.assert_array_equal(tmpl(piecewise_warp(t, tmpl.landmarks_s, tmpl.landmarks_s)), expected)
    pred, A, c = solve_amp_offset(expected, expected, "amp_offset", Bounds())
    np.testing.assert_allclose(pred, expected, atol=1e-9)
    np.testing.assert_allclose(A, 1.0, atol=1e-9)
    np.testing.assert_allclose(c, 0.0, atol=1e-9)


def test_identity_trial_fits_to_identity():
    tmpl = smooth_template()
    t = trial_times()
    trial = TrialData(t, tmpl(t), tmpl.landmarks_s.copy())
    fits = fit_all(trial, tmpl, FitSettings())
    for name in ("tau", "tau_scale", "tau_scale_amp", "tau_scale_amp_offset", "piecewise_landmark", "piecewise_fit"):
        assert fits[name].cost < 1e-6, name
    assert abs(fits["tau_scale"].params["tau_s"]) < 1e-3
    assert abs(fits["tau_scale"].params["scale"] - 1) < 1e-3


@pytest.mark.parametrize("tau, scale", [(0.35, 1.0), (-0.2, 1.3), (0.6, 0.8)])
def test_shift_and_scale_are_recovered(tau, scale):
    tmpl = smooth_template()
    t = trial_times()
    y = tmpl(shift_scale_warp(t, tau, scale, tmpl.onset_s))
    fit = fit_shift_scale(TrialData(t, y, tmpl.landmarks_s), tmpl, True, "none", FitSettings())
    assert fit.params["tau_s"] == pytest.approx(tau, abs=0.01)
    assert fit.params["scale"] == pytest.approx(scale, abs=0.01)
    assert fit.cost < 1e-2


def test_amplitude_and_offset_are_recovered():
    tmpl = smooth_template()
    t = trial_times()
    A, c = np.array([1.4, 0.7, 2.0]), np.array([1.5, -2.0, 0.5])
    y = tmpl(shift_scale_warp(t, 0.25, 1.1, tmpl.onset_s)) * A + c
    fit = fit_shift_scale(TrialData(t, y, tmpl.landmarks_s), tmpl, True, "amp_offset", FitSettings())
    np.testing.assert_allclose([fit.params[f"amp_{a}"] for a in "xyz"], A, atol=0.02)
    np.testing.assert_allclose([fit.params[f"offset_{a}"] for a in "xyz"], c, atol=0.1)


def test_amp_offset_respects_bounds():
    rng = np.random.default_rng(0)
    T = rng.normal(size=(200, 3))
    y = T * np.array([10.0, 0.01, 1.0]) + np.array([50.0, 0.0, -50.0])
    b = Bounds()
    _, A, c = solve_amp_offset(T, y, "amp_offset", b)
    assert np.all((A >= b.amp[0]) & (A <= b.amp[1]))
    assert np.all((c >= b.offset_cm[0]) & (c <= b.offset_cm[1]))
    # The bounded solution is the best feasible point of a coarse grid.
    a_grid, c_grid = np.linspace(*b.amp, 271), np.linspace(*b.offset_cm, 201)
    for ax in range(3):
        sse = ((y[:, ax, None, None] - a_grid[:, None] * T[:, ax, None, None] - c_grid) ** 2).sum(0)
        best = np.sum((y[:, ax] - A[ax] * T[:, ax] - c[ax]) ** 2)
        assert best <= sse.min() + 1e-6


def test_piecewise_landmarks_recover_a_phase_warp():
    tmpl = smooth_template()
    trial_bp = tmpl.landmarks_s + np.array([0.2, 0.35, 0.3, 0.6, 0.5])
    t = trial_times(stop_s=6.3)
    y = tmpl(piecewise_warp(t, trial_bp, tmpl.landmarks_s))
    trial = TrialData(t, y, trial_bp)
    assert fit_piecewise(trial, tmpl, False, FitSettings()).cost < 1e-6
    # Refinement starting from wrong landmarks moves back toward the true breakpoints and lowers the cost.
    wrong = TrialData(t, y, trial_bp + np.array([0.1, -0.1, 0.1, -0.1, 0.1]))
    coarse = fit_piecewise(wrong, tmpl, False, FitSettings())
    refined = fit_piecewise(wrong, tmpl, True, FitSettings())
    assert refined.cost < 0.5 * coarse.cost


def test_min_jerk_recovers_synthetic_movement():
    t = trial_times(stop_s=6.0)
    starts, durations = np.array([0.45, 2.1, 4.2, 4.85]), np.array([1.0, 0.5, 0.55, 0.9])
    disp = np.array([[0.0, 0.0, 0.0], [20, 10, 2], [0, 0, 5], [0, 0, -5], [-20, -10, -2]], dtype=float)
    y = min_jerk_basis(t, starts, durations) @ disp
    landmarks = np.array([0.45, 1.45, 2.1, 4.75, 5.75])
    fit = fit_min_jerk(TrialData(t, y, landmarks), FitSettings())
    assert fit.cost < 1e-3
    np.testing.assert_allclose([fit.params[f"{k}_duration_s"] for k in ("reach", "lift", "lower", "return")],
                               durations, atol=0.02)
