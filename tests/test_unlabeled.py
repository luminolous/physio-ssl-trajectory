"""Tests for the unlabeled EEG streams and window dataset (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegtraj.data.unlabeled import (SOURCES, UnlabeledWindows, lift_stats, raw_to_grid, segment_hours,
                                    series_segments, window_starts, windows_from_config)

N_CH = 32


def test_raw_to_grid_keeps_samples_whose_raw_index_is_inside():
    # factor 5: grid sample k is raw 5k + 4.
    assert raw_to_grid(0, 10, 5) == (0, 2)  # raw 4, 9
    assert raw_to_grid(5, 9, 5) == (1, 1)  # raw 9 is outside [5, 9)
    assert raw_to_grid(5, 10, 5) == (1, 2)
    assert raw_to_grid(1, 1 + 5 * 3, 5) == (0, 3)  # a lift at hs_start 1 with 3 samples: raw 5, 10, 15 -> k 0..2 (raw 4, 9, 14)


def test_series_segments_partition_the_recording():
    lifts = [(100, 600), (500, 900), (1500, 2000)]  # first two overlap
    segs = series_segments(lifts, 2500, 5)
    assert [s[0] for s in segs] == ["between", "lift", "between", "lift", "between"]
    ends = [s[2] for s in segs]
    assert all(a[2] == b[1] for a, b in zip(segs[:-1], segs[1:]))  # contiguous, no overlap
    assert segs[0][1] == 0 and ends[-1] == 500  # 2500 raw samples -> 500 grid samples


def test_series_segments_rejects_lifts_outside_the_recording():
    with pytest.raises(ValueError):
        series_segments([(0, 3000)], 2500, 5)


def test_window_starts_never_cross_a_segment():
    seg = pd.DataFrame({"start": [0, 449, 648], "end": [449, 648, 898]})
    st = window_starts(seg, 200, 50)
    assert st.tolist() == [0, 50, 100, 150, 200, 648, 698]  # 449 -> 5, 199 -> 0, 250 -> 2
    for a in st:
        assert any(s <= a and a + 200 <= e for s, e in zip(seg.start, seg.end))


def fake_subject(root: Path, subject: str, seed: int, variants=("broadband_car",)) -> pd.DataFrame:
    """A subject folder that holds only stream files (no trials.csv, kinematics, events, or manifest)."""
    d = root / subject
    d.mkdir(parents=True)
    seg = pd.DataFrame({"series": ["S1", "S1", "S1", "ST"], "source": ["between", "lift", "between", "st"],
                        "start": [0, 300, 900, 1200], "end": [300, 900, 1200, 1600]})
    seg.to_csv(d / "stream_segments.csv", index=False)
    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 3.0, (N_CH, 1600)).astype(np.float32)
    for i, v in enumerate(variants):
        np.save(d / f"stream_eeg_{v}.npy", base + i)
    return seg


def test_dataset_reads_only_stream_files_and_returns_eeg_only(tmp_path):
    for i, s in enumerate(["P1", "P2"]):
        fake_subject(tmp_path, s, i)
    assert sorted(p.name for p in (tmp_path / "P1").iterdir()) == ["stream_eeg_broadband_car.npy", "stream_segments.csv"]
    ds = UnlabeledWindows(tmp_path, ["P1", "P2"], ["broadband_car"], SOURCES, window=200, stride=100)
    x = ds[0]
    assert tuple(x.shape) == (N_CH, 200) and str(x.dtype) == "torch.float32"
    # per subject: between [0,300) -> 2, lift [300,900) -> 5, between [900,1200) -> 2, st [1200,1600) -> 3
    assert len(ds) == 2 * 12


def test_dataset_zscores_with_the_subject_lift_statistics(tmp_path):
    fake_subject(tmp_path, "P1", 0)
    ds = UnlabeledWindows(tmp_path, ["P1"], ["broadband_car"], ["lift"], window=600, stride=600)
    assert len(ds) == 1
    x = ds[0].numpy()  # the whole lift segment
    assert np.allclose(x.mean(axis=1), 0, atol=1e-4) and np.allclose(x.std(axis=1), 1, atol=1e-3)


def test_variants_are_aligned_sample_by_sample(tmp_path):
    fake_subject(tmp_path, "P1", 0, variants=("broadband_car", "broadband_ica_car"))
    ds = UnlabeledWindows(tmp_path, ["P1"], ["broadband_car", "broadband_ica_car"], SOURCES, window=200, stride=50)
    raw, other = ds[3]
    assert raw.shape == other.shape
    assert np.allclose(raw.numpy(), other.numpy(), atol=1e-5)  # the second variant is a constant shift, removed by z-score


def test_sources_filter(tmp_path):
    fake_subject(tmp_path, "P1", 0)
    ds = UnlabeledWindows(tmp_path, ["P1"], ["broadband_car"], ["st"], window=200, stride=100)
    assert len(ds) == 3 and set(ds.source_index.tolist()) == {SOURCES.index("st")}
    with pytest.raises(ValueError):
        UnlabeledWindows(tmp_path, ["P1"], ["broadband_car"], ["kinematics"])


def test_strict_per_fold_drops_every_window_of_the_excluded_subject(tmp_path):
    for i, s in enumerate(["P1", "P2", "P3"]):
        fake_subject(tmp_path, s, i)
    cfg = {"cache_dir": tmp_path, "subjects": ["P1", "P2", "P3"], "variants": ["broadband_car"], "sources": list(SOURCES),
           "window": 200, "stride": 100, "strict_per_fold": True, "exclude_subject": "P2"}
    ds = windows_from_config(cfg)
    assert ds.subjects == ["P1", "P3"]
    assert len(ds) == 2 * 12
    assert "P2" not in ds.eeg
    full = windows_from_config({**cfg, "strict_per_fold": False})
    assert len(full) == 3 * 12
    with pytest.raises(ValueError):
        windows_from_config({**cfg, "exclude_subject": None})


def test_lift_stats_pool_all_lift_segments():
    eeg = np.arange(2 * 10, dtype=np.float32).reshape(2, 10)
    seg = pd.DataFrame({"start": [0, 6], "end": [3, 10]})
    mean, sd = lift_stats(eeg, seg)
    pooled = np.concatenate([eeg[:, 0:3], eeg[:, 6:10]], axis=1)
    assert np.allclose(mean[:, 0], pooled.mean(axis=1)) and np.allclose(sd[:, 0], pooled.std(axis=1), atol=1e-4)


def test_segment_hours(tmp_path):
    fake_subject(tmp_path, "P1", 0)
    h = segment_hours(tmp_path, ["P1"])
    assert h.loc["P1", "lift"] == pytest.approx(600 / 100 / 3600)
    assert h.loc["P1", "all"] == pytest.approx(1600 / 100 / 3600)
