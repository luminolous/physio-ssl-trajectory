"""Tests for event-locked analysis and extra cache bands (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from eegtraj.data.band_variant import lift_raw_indices, with_band
from eegtraj.data.cache import config_hash, keep_indices
from eegtraj.data.pre_lift import cut_pre_lift
from eegtraj.diagnostics.neurophysiology import (asymmetry_table, band_window_change, earliest_used_s, epoch_stream, erp_epochs,
                                                 event_sample, filter_delay_ms, lift_pair_correlations, lift_stream,
                                                 paired_summary, peak_latency_s, percent_change,
                                                 relative_times, rt_thirds, sample_range, tf_event_locked, transient_mask)

FS = 100.0


def make_lifts(n_lifts=40, n_samples=800, n_pre=200, led_s=2.0, rt_s=None, signal=None, seed=0):
    """Two channels of noise; ``signal(t_rel_onset, t_rel_led)`` is added to channel 0 on the real time grid."""
    rng = np.random.default_rng(seed)
    rt = np.full(n_lifts, 0.4) if rt_s is None else np.asarray(rt_s)
    onset = led_s + rt
    eeg, pre = np.zeros((2, n_lifts * n_samples), np.float32), np.zeros((2, n_lifts, n_pre), np.float32)
    for i in range(n_lifts):
        j = np.arange(-n_pre, n_samples)
        x = 0.05 * rng.standard_normal((2, j.size))
        if signal is not None:
            x[0] += signal((j + 1) / FS - onset[i], (j + 1) / FS - led_s)
        pre[:, i] = x[:, :n_pre]
        eeg[:, i * n_samples:(i + 1) * n_samples] = x[:, n_pre:]
    trials = pd.DataFrame({"offset_start": np.arange(n_lifts) * n_samples, "offset_end": (np.arange(n_lifts) + 1) * n_samples,
                           "onset_s": onset, "ev_LEDOn": led_s, "ev_tHandStop": onset + 4.0})
    return eeg, pre, trials


def test_event_sample_follows_the_cache_time_grid():
    assert event_sample(2.0, FS) == 199  # sample j is at (j + 1) / fs
    assert (199 + 1) / FS == 2.0


def test_epoch_finds_an_impulse_at_zero_lag_inside_the_lift_and_before_it():
    eeg = np.zeros((1, 100), np.float32)
    pre = np.zeros((1, 1, 20), np.float32)
    eeg[0, 30] = 1  # lift sample j = 30, time 0.31 s
    pre[0, 0, 15] = 2  # j = -5, time -0.04 s
    s = lift_stream(eeg, pre, 0, 0, 100)
    assert s.shape == (1, 120)
    t = relative_times(-3, 3, FS)
    assert t[np.argmax(epoch_stream(s, 20, event_sample(0.31, FS), -3, 3)[0])] == 0
    assert t[np.argmax(epoch_stream(s, 20, event_sample(-0.04, FS), -3, 3)[0])] == 0


def test_epoch_is_nan_where_no_eeg_exists():
    s = np.ones((1, 10), np.float32)
    e = epoch_stream(s, 2, 7, -1, 3)  # stream indices 8, 9, 10, 11, 12 of a stream of 10
    assert (e[0, :2] == 1).all() and np.isnan(e[0, 2:]).all()


def test_erp_baseline_uses_led_time_for_both_alignments():
    eeg, pre, trials = make_lifts(n_lifts=3, signal=lambda to, tl: 5.0 + 0 * to)
    ep = erp_epochs(eeg, pre, trials, FS, {"onset": [-1, 1], "led": [-1, 1]}, [-1.5, -0.5])
    assert abs(np.nanmean(ep["onset"][:, 0])) < 0.05 and abs(np.nanmean(ep["led"][:, 0])) < 0.05


def test_onset_locked_component_is_sharper_with_onset_alignment_and_keeps_its_latency():
    rt = np.random.default_rng(1).uniform(0.2, 1.0, 60)
    bump = lambda to, tl: -np.exp(-0.5 * ((to - 0.1) / 0.08) ** 2)
    eeg, pre, trials = make_lifts(n_lifts=60, rt_s=rt, signal=bump)
    ep = erp_epochs(eeg, pre, trials, FS, {"onset": [-1, 1], "led": [-1, 2]}, [-1.5, -0.5])
    avg_on, avg_led = np.nanmean(ep["onset"][:, 0], axis=0), np.nanmean(ep["led"][:, 0], axis=0)
    t_on = relative_times(*sample_range([-1, 1], FS), FS)
    assert abs(peak_latency_s(avg_on, t_on, [-1, 1]) - 0.1) <= 0.02
    assert abs(avg_on.min()) > 2 * abs(avg_led.min())


def test_rt_split_follows_a_movement_locked_component_when_aligned_to_led():
    rt = np.concatenate([np.full(30, 0.3), np.full(30, 0.8)])
    bump = lambda to, tl: -np.exp(-0.5 * (to / 0.08) ** 2)
    eeg, pre, trials = make_lifts(n_lifts=60, rt_s=rt, signal=bump)
    ep = erp_epochs(eeg, pre, trials, FS, {"led": [-1, 2]}, [-1.5, -0.5])["led"][:, 0]
    fast, slow = rt < 0.5, rt > 0.5
    t = relative_times(*sample_range([-1, 2], FS), FS)
    shift = peak_latency_s(ep[slow].mean(0), t, [0, 2]) - peak_latency_s(ep[fast].mean(0), t, [0, 2])
    assert abs(shift - 0.5) <= 0.02


def test_rt_thirds_are_disjoint_and_ordered():
    rt = np.arange(30.0)
    fast, slow = rt_thirds(rt, 1 / 3)
    assert not (fast & slow).any() and rt[fast].max() < rt[slow].min()


def test_transient_mask_uses_the_lift_time_in_its_series():
    tr = pd.DataFrame({"series": [1, 1, 2], "t0_series_s": [5.0, 20.0, 5.0]})
    mask = transient_mask(tr, {"S1": 13.0, "S2": 1.0}, np.array([1.0, 1.0, 1.0]))
    assert mask.tolist() == [True, False, False]


def test_earliest_used_covers_epochs_and_baseline():
    tr = pd.DataFrame({"onset_s": [2.3, 1.0], "ev_LEDOn": [2.0, 2.0]})
    e = earliest_used_s(tr, {"onset": [-2.0, 1.0]}, [-1.5, -0.5])
    assert e == pytest.approx([0.3, -1.0])


def test_tf_recovers_erd_on_the_right_channel_at_the_right_time():
    def mu(to, tl):
        amp = np.where((to >= 0) & (to <= 2), 0.5, 1.0)  # power falls to 25 % during movement
        return amp * np.sin(2 * np.pi * 10 * to)
    eeg, pre, trials = make_lifts(n_lifts=60, signal=mu)
    freqs = np.arange(4.0, 21.0)
    out = tf_event_locked(eeg, pre, trials, FS, freqs, 0.5, 0.25, {"onset": [-2, 3]}, [-1.5, -0.5], 0.5)
    t = relative_times(*sample_range([-2, 3], FS), FS)
    during = band_window_change(out["power"]["onset"], out["baseline"], freqs, t, [8, 12], [0.5, 1.5])
    before = band_window_change(out["power"]["onset"], out["baseline"], freqs, t, [8, 12], [-1.0, -0.5])
    assert during[0] == pytest.approx(-75, abs=8)
    assert abs(before[0]) < 8
    assert abs(during[1]) < 25  # the noise-only channel has no ERD of that size


def test_tf_drops_edges_and_low_coverage():
    eeg, pre, trials = make_lifts(n_lifts=2, n_samples=400, n_pre=50)
    out = tf_event_locked(eeg, pre, trials, FS, np.array([10.0]), 0.5, 0.25, {"onset": [-2, 3]}, [-1.5, -0.5], 0.5)
    t = relative_times(*sample_range([-2, 3], FS), FS)
    # Onset is at 2.4 s; available EEG is -0.5 to 4 s, so usable power ends about 1.35 s after onset.
    assert np.isnan(out["power"]["onset"][..., t > 1.4]).all()
    assert np.isfinite(out["power"]["onset"][..., (t > -1.9) & (t < 1.2)]).all()


def test_asymmetry_table_and_lift_correlation():
    names = ["C3", "C4", "F7", "F8"]
    t = relative_times(-10, 10, FS)
    asym = {"reference_pair": ["C3", "C4"], "ocular_pair": ["F7", "F8"], "windows_s": {"w": ["onset", [-0.1, 0.1]]},
            "pairs": [["C3", "C4"], ["F7", "F8"]]}
    r = {"subject": "P1", "names": names, "erp": {"onset": np.array([[1.0] * 21, [3.0] * 21, [5.0] * 21, [2.0] * 21])},
         "erp_times": {"onset": t}}
    table = asymmetry_table([r], asym).set_index("pair").diff_uv
    assert table["C3-C4"] == -2 and table["F7-F8"] == 3
    amp = np.arange(6.0)
    lifts = np.zeros((6, 4, 21))
    lifts[:, 0] = amp[:, None]  # C3 - C4 follows F7 - F8 across lifts
    lifts[:, 2] = 2 * amp[:, None]
    assert lift_pair_correlations({"onset": lifts}, {"onset": t}, names, np.ones(6, bool), asym)["w"] == pytest.approx(1)


def test_percent_change_sign():
    assert percent_change(np.full((1, 1, 2), 0.5), np.ones((1, 1)))[0, 0, 0] == pytest.approx(-50)


def test_paired_summary_counts_negative_differences():
    s = paired_summary(np.arange(10.0) - 20, np.arange(10.0))
    assert s["n_negative"] == 10 and s["diff_mean"] == -20
    assert s["wilcoxon_p"] == pytest.approx(2 / 2 ** 10)


def test_filter_delay_of_a_pure_delay():
    two_sample_delay = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    assert filter_delay_ms([two_sample_delay, two_sample_delay], 500.0, [1.0])[1.0] == pytest.approx(8.0, abs=1e-3)


def test_with_band_leaves_the_base_config_and_its_hash_alone():
    cfg = {"eeg": {"bands_hz": {"broadband": [0.1, 40.0]}}}
    before = config_hash(cfg)
    out = with_band(cfg, "mrcp", [0.1, 2.0])
    assert out["eeg"]["bands_hz"]["mrcp"] == [0.1, 2.0] and config_hash(cfg) == before
    with pytest.raises(ValueError):
        with_band(cfg, "broadband", [0.1, 2.0])


def test_lift_indices_match_the_main_cache_grid():
    assert (lift_raw_indices(100, 5, 4) - 100).tolist() == keep_indices(20, 5).tolist()


def test_cut_pre_lift_holds_the_earliest_grid_value_before_the_series_start():
    stream = np.arange(40, dtype=float).reshape(1, 40)
    out, missing = cut_pre_lift(stream, 1, 5, 2)  # raw indices -5 and 0; the lift grid is 0, 5, ...
    assert missing == 1 and out[0].tolist() == [0.0, 0.0]
