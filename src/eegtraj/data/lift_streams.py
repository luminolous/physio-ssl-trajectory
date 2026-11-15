"""Per-lift streams for the stage 2 decoder (design.md "Streaming"; SSL-17).

Each lift is cut as a stream that starts at a fixed time relative to the cue: stream sample 0 is lift-window sample
j0 = round((t_LED - 1.5 s) * fs), i.e. never at a time derived from the lift's movement. The stream carries
``history`` = 180 samples before j0 (taken from the pre-lift cache when j0 < 180) and covers whole 0.2 s patches up to
the last scored sample. Samples after the lift window (at most one patch, after every scored sample) are zero; the
decoder is causal, so they cannot affect any scored output.

Inputs (all label-free):
- EEG: broadband CAR, z-scored per subject and channel over the subject's lift samples (D9, transductive, as in
  pretraining).
- EOG proxies (``eegtraj.preprocessing.ocular.proxy_rows``: VEOG, HEOG) from the same un-normalized CAR EEG, z-scored
  per subject over the lift samples.
Targets and timing: D1 wrist position at the scored samples (D2 segment); seconds since LED-on per scored sample.
Kinematic events and condition labels are not stored here (training-only labels: ``eegtraj.data.decoder_labels``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eegtraj.data.windows import SubjectSamples, input_channel_names, load_subject_samples
from eegtraj.preprocessing.ocular import proxy_rows

STREAM_START_BEFORE_LED_S = 1.5  # SSL-17
PRE_LIFT_SAMPLES = 299  # history loaded from the pre-lift cache (3 s stored)


@dataclass
class LiftStream:
    """One lift. ``eeg`` (32, H + 20 N), ``eog`` (2, H + 20 N) float32; ``out_idx`` (S,) stream sample of each scored
    sample; ``sample_pos`` (S,) positions in ``SubjectSamples.sample_idx`` (for scoring); ``y`` (S, 3);
    ``t_since_led`` (S,) s; ``j0`` lift-window sample of stream sample 0; ``n_patches``; ``lift_len`` samples in the
    lift window; ``orig_offset`` the lift's offset in the main cache arrays (for training-only per-sample labels)."""

    subject: str
    trial_row: int
    eeg: np.ndarray
    eog: np.ndarray
    out_idx: np.ndarray
    sample_pos: np.ndarray
    y: np.ndarray
    t_since_led: np.ndarray
    j0: int
    n_patches: int
    lift_len: int
    orig_offset: int

    def patch_end_lift_samples(self, patch: int = 20) -> np.ndarray:
        """Lift-window sample index of each patch's last sample."""
        return self.j0 + np.arange(self.n_patches) * patch + patch - 1


@dataclass
class SubjectStreams:
    samples: SubjectSamples
    lifts: list[LiftStream]


def zscore_rows(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    stats = np.asarray(x[:, mask], np.float64)
    mean, sd = stats.mean(axis=1, keepdims=True), stats.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / sd).astype(np.float32)


def build_subject_streams(cache_dir: str | Path, subject: str, history: int = 180, patch: int = 20,
                          before_led_s: float = STREAM_START_BEFORE_LED_S,
                          keep_trial: np.ndarray | None = None) -> SubjectStreams:
    """Streams of every lift of ``subject`` (or of the rows where ``keep_trial`` is True)."""
    s = load_subject_samples(cache_dir, subject, "broadband", "car", "all", "position", history=PRE_LIFT_SAMPLES + 1,
                             pre_lift=True)
    names = input_channel_names(cache_dir, subject, "broadband", "all")
    eog_raw = proxy_rows(s.eeg, names)
    eeg = zscore_rows(s.eeg, s.eeg_stat_mask)
    eog = zscore_rows(eog_raw, s.eeg_stat_mask)
    orig = pd.read_csv(Path(cache_dir) / subject / "trials.csv")
    fs = s.fs_hz
    lifts = []
    for row, tr in enumerate(s.trials.itertuples(index=False)):
        if keep_trial is not None and not keep_trial[row]:
            continue
        pos = np.flatnonzero(s.trial_row == row)
        j = s.sample_idx[pos] - tr.offset_start  # lift-window sample of each scored sample
        j0 = int(round((tr.ev_LEDOn - before_led_s) * fs))
        if j0 - history < -PRE_LIFT_SAMPLES:
            raise ValueError(f"{subject} lift row {row}: stream history reaches before the pre-lift cache")
        out_idx = j - j0
        if out_idx.min() < patch - 1:
            raise ValueError(f"{subject} lift row {row}: a scored sample precedes the first patch end")
        n_patches = int(np.ceil((out_idx.max() + 1) / patch))
        a = tr.offset_start + j0 - history
        b = tr.offset_start + j0 + n_patches * patch
        lift_end = int(tr.offset_end)
        valid_end = min(b, lift_end)

        def cut(x: np.ndarray) -> np.ndarray:
            out = np.zeros((x.shape[0], b - a), np.float32)
            out[:, :valid_end - a] = x[:, a:valid_end]
            return out

        t_led = (j + 1) / fs - tr.ev_LEDOn
        lifts.append(LiftStream(subject, row, cut(eeg), cut(eog), out_idx.astype(np.int64), pos.astype(np.int64),
                                s.target[:, s.sample_idx[pos]].T.astype(np.float32), t_led.astype(np.float32), j0,
                                n_patches, int(tr.offset_end - tr.offset_start), int(orig.offset_start[row])))
    return SubjectStreams(s, lifts)


def aux_patch_targets(lift: LiftStream, aux: np.ndarray, patch: int = 20) -> np.ndarray:
    """Auxiliary class per patch from training-only per-sample classes ``aux`` (main-cache layout); -1 where the
    patch end lies outside the lift window."""
    j = lift.patch_end_lift_samples(patch)
    valid = (j >= 0) & (j < lift.lift_len)
    out = np.full(j.shape, -1, np.int64)
    out[valid] = aux[lift.orig_offset + j[valid]]
    return out
