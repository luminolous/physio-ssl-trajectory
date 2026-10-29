"""WAY-EEG-GAL loader: per-lift trial records (WS) and continuous series (HS).

Raw values only; no filtering, resampling, or re-referencing (see
``eegtraj.preprocessing``). Facts used here are verified in ``docs/dataset.md``:

- WS window i is an exact copy of HS samples starting at
  ``round((StartTime - LEDOn) * fs)``; ``StartTime`` is LED-on in series time.
- AllLifts event times are seconds from the WS window start.
- EEG and kinematics share 500 Hz samples; EMG is 4000 Hz (8x).
- Position sensors: 1 = object, 2/3 = index/thumb (order unknown), 4 = wrist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eegtraj.data import raw_io

logger = logging.getLogger(__name__)

N_POS_SENSORS = 4
EMG_PER_EEG_SAMPLE = 8
# AllLifts columns named t* that are not event times (they hold peak hand speeds).
NOT_EVENT_COLUMNS = ("tPeakVelHandReach", "tPeakVelHandRetract")
EVENT_COLUMNS_EXTRA = ("LEDOn", "LEDOff")
META_COLUMNS = ("Part", "Run", "Lift", "CurW", "CurS", "PrevW", "PrevS", "StartTime", "BlockType")
TIME_MATCH_TOL_S = 1e-9


@dataclass
class Trial:
    """One lift (one WS window).

    Arrays are float32, channels first:
        eeg: (n_eeg, n_times), raw units (likely µV), fs = ``eeg_fs_hz``.
        emg: (n_emg, n_times * 8), raw units, fs = ``emg_fs_hz``.
        kin_pos: (4, 3, n_times), position of sensors 1..4 in x, y, z (cm, lab frame), fs = ``eeg_fs_hz``.
    Times:
        events: seconds from the window start (sample k of ``eeg`` is at ``(k + 1) / eeg_fs_hz``).
        t0_series_s: window start in series (HS) time.
        hs_start_sample: index of this window's first sample in the HS series.
    """

    subject: str
    series: int
    lift: int
    eeg: np.ndarray
    emg: np.ndarray
    kin_pos: np.ndarray
    eeg_names: list[str]
    emg_names: list[str]
    eeg_fs_hz: float
    emg_fs_hz: float
    events: dict[str, float]
    t0_series_s: float
    hs_start_sample: int
    block_type: int
    weight_g: int
    surface: str
    prev_weight_g: int | None
    prev_surface: str | None
    measures: dict[str, float] = field(default_factory=dict)

    @property
    def n_times(self) -> int:
        return self.eeg.shape[1]


@dataclass
class Series:
    """Continuous HS recording of one series (``S1``..``S9`` or ``ST``).

    eeg: float32 (n_eeg, n_times), fs = ``eeg_fs_hz``.
    emg: float32 (n_emg, n_times_emg) or None (ST has no EMG), fs = ``emg_fs_hz``.
    kin_pos: float32 (4, 3, n_times) or None (ST has no kinematics).
    """

    subject: str
    series: str
    eeg: np.ndarray
    eeg_names: list[str]
    eeg_fs_hz: float
    emg: np.ndarray | None
    emg_fs_hz: float | None
    kin_pos: np.ndarray | None


def _positions(kin: np.ndarray, kin_names: list[str]) -> np.ndarray:
    """(n_times, n_kin) -> float32 (4, 3, n_times)."""
    cols = [raw_io.position_columns(kin_names, s) for s in range(1, N_POS_SENSORS + 1)]
    return np.ascontiguousarray(np.stack([kin[:, c].T for c in cols]), dtype=np.float32)


def load_trials(subject: str, series: str, raw_dir: Path = raw_io.DEFAULT_RAW_DIR,
                lifts: pd.DataFrame | None = None) -> list[Trial]:
    """Load all lifts of one series from WS, with events and metadata from AllLifts.

    Raises ValueError if WS and AllLifts disagree on the number of lifts or their timing.
    """
    ws = raw_io.load_ws(subject, series, raw_dir)
    lifts = raw_io.load_alllifts(subject, raw_dir) if lifts is None else lifts
    rows = lifts[lifts.Run == int(series[1:])].sort_values("Lift").reset_index(drop=True)
    wins = ws["win"]
    if len(wins) != len(rows):
        raise ValueError(f"{subject} {series}: {len(wins)} WS windows but {len(rows)} AllLifts rows")

    kin_names = list(ws["names"]["kin"])
    eeg_names, emg_names = list(ws["names"]["eeg"]), list(ws["names"]["emg"])
    event_cols = [c for c in rows.columns if c.startswith("t") and c not in NOT_EVENT_COLUMNS] + list(EVENT_COLUMNS_EXTRA)
    measure_cols = [c for c in rows.columns if c not in event_cols and c not in META_COLUMNS]

    trials = []
    for i, (w, r) in enumerate(zip(wins, rows.itertuples(index=False))):
        if abs(w["trial_start_time"] - r.StartTime) > TIME_MATCH_TOL_S or abs(w["LEDoff"] - r.LEDOff) > TIME_MATCH_TOL_S:
            raise ValueError(f"{subject} {series} lift {i + 1}: WS timing does not match AllLifts")
        eeg_fs = 1.0 / float(np.median(np.diff(w["eeg_t"])))
        emg_fs = 1.0 / float(np.median(np.diff(w["emg_t"])))
        if len(w["emg"]) != EMG_PER_EEG_SAMPLE * len(w["eeg"]):
            raise ValueError(f"{subject} {series} lift {i + 1}: EMG length is not 8x EEG length")
        t0 = float(r.StartTime - r.LEDOn)
        row = r._asdict()
        trials.append(Trial(
            subject=subject,
            series=int(r.Run),
            lift=int(r.Lift),
            eeg=np.ascontiguousarray(w["eeg"].T, dtype=np.float32),
            emg=np.ascontiguousarray(w["emg"].T, dtype=np.float32),
            kin_pos=_positions(w["kin"], kin_names),
            eeg_names=eeg_names,
            emg_names=emg_names,
            eeg_fs_hz=round(eeg_fs, 6),
            emg_fs_hz=round(emg_fs, 6),
            events={c: float(row[c]) for c in event_cols},
            t0_series_s=t0,
            hs_start_sample=int(round(t0 * eeg_fs)),
            block_type=int(r.BlockType),
            weight_g=int(str(w["weight_id"]).rstrip("g")),
            surface=str(w["surf_id"]),
            prev_weight_g=None if str(w["weight_prev_id"]) == "None" else int(str(w["weight_prev_id"]).rstrip("g")),
            prev_surface=None if str(w["surf_prev_id"]) == "None" else str(w["surf_prev_id"]),
            measures={c: float(row[c]) for c in measure_cols},
        ))
    logger.info("Loaded %s %s: %d trials", subject, series, len(trials))
    return trials


def load_series(subject: str, series: str, raw_dir: Path = raw_io.DEFAULT_RAW_DIR) -> Series:
    """Load the continuous HS recording of one series."""
    hs = raw_io.load_hs(subject, series, raw_dir)
    emg = hs.get("emg")
    kin = hs.get("kin")
    return Series(
        subject=subject,
        series=series,
        eeg=np.ascontiguousarray(hs["eeg"]["sig"].T, dtype=np.float32),
        eeg_names=list(hs["eeg"]["names"]),
        eeg_fs_hz=float(hs["eeg"]["samplingrate"]),
        emg=None if emg is None else np.ascontiguousarray(emg["sig"].T, dtype=np.float32),
        emg_fs_hz=None if emg is None else float(emg["samplingrate"]),
        kin_pos=None if kin is None else _positions(kin["sig"], list(kin["names"])),
    )
