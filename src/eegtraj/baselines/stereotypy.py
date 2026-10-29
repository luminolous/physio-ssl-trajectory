"""Stereotypy floor: score another trial of the same subject as the "prediction".

For each test trial, a donor trial is drawn uniformly from the other test trials of the
same subject; the donor's target, aligned on time since kinematic onset, is the
prediction. Repeating the draw ``n_permutations`` times gives the spread. Like the
template, this uses onset timing of both trials (oracle timing) and no EEG.
"""

from __future__ import annotations

import numpy as np

from eegtraj.baselines.base import Baseline, FitContext, register
from eegtraj.data.windows import SubjectSamples


@register("stereotypy")
class StereotypyBaseline(Baseline):
    uses_eeg = False

    def __init__(self, n_permutations: int = 10, **params):
        super().__init__(n_permutations=n_permutations, **params)
        self.n_repeats = int(n_permutations)

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        self.seed = ctx.seed  # nothing to learn from training subjects

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext, repeat: int = 0) -> np.ndarray:
        tr = subject.trials
        rows = subject.trial_row[mask]
        idx = subject.sample_idx[mask]
        trial_rows = np.unique(rows)
        if len(trial_rows) < 2:
            raise ValueError(f"{subject.subject}: stereotypy floor needs at least 2 test trials")
        rng = np.random.default_rng([self.seed, int(subject.subject[1:]), repeat])
        # Donor for each trial: uniform over the other trials (shift by 1..n-1 positions).
        shift = rng.integers(1, len(trial_rows), size=len(trial_rows))
        donor = {r: trial_rows[(i + shift[i]) % len(trial_rows)] for i, r in enumerate(trial_rows)}

        fs = subject.fs_hz
        out = np.empty((len(idx), subject.target.shape[0]), dtype=np.float32)
        for r in trial_rows:
            sel = rows == r
            tau = (idx[sel] - tr.offset_start.iat[r] + 1) / fs - tr.onset_s.iat[r]
            d = donor[r]
            d_slice = slice(tr.offset_start.iat[d], tr.offset_end.iat[d])
            d_tau = (np.arange(tr.n_samples.iat[d]) + 1) / fs - tr.onset_s.iat[d]
            d_y = subject.target[:, d_slice]
            # Interpolate the donor on the test trial's time-since-onset grid; hold edges outside its range.
            out[sel] = np.stack([np.interp(tau, d_tau, d_y[a]) for a in range(d_y.shape[0])], axis=1)
        return out

    def info(self) -> dict:
        return {"oracle_timing": True, "n_permutations": self.n_repeats,
                "note": "prediction = another test trial of the same subject, aligned on kinematic onset"}
