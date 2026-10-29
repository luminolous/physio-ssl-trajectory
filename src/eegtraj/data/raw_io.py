"""Thin readers for the raw WAY-EEG-GAL .mat files.

These return the MATLAB structs as plain Python dicts (``scipy.io.loadmat`` with
``simplify_cells=True``) plus a few named lookups. They do no preprocessing.
See ``docs/dataset.md`` for the file layout.

Facts used here (verified in ``.agents/data-baselines/dataset-notes.md``):

- Signal arrays are stored samples x channels.
- EEG and kinematics: 500 Hz, same samples. EMG: 4000 Hz.
- Position columns are named ``P{axis}{sensor} - position {axis} sensor {sensor}``.
- Sensor 1 = object, sensor 4 = wrist, sensors 2/3 = index finger and thumb (order unknown).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io

DEFAULT_RAW_DIR = Path("dataset/raw")
SERIES = tuple(f"S{i}" for i in range(1, 10))
AXES = ("x", "y", "z")
SENSOR_ROLES = {1: "object", 2: "digit_a", 3: "digit_b", 4: "wrist"}


def subjects(raw_dir: Path = DEFAULT_RAW_DIR) -> list[str]:
    """Subject IDs present on disk, sorted numerically (``["P1", ..., "P12"]``)."""
    return sorted((p.name for p in Path(raw_dir).glob("P*") if p.is_dir()), key=lambda s: int(s[1:]))


def load_ws(subject: str, series: str, raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, Any]:
    """Load ``WS_<subject>_<series>.mat`` and return the ``ws`` struct."""
    return scipy.io.loadmat(Path(raw_dir) / subject / f"WS_{subject}_{series}.mat", simplify_cells=True)["ws"]


def load_hs(subject: str, series: str, raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, Any]:
    """Load ``HS_<subject>_<series>.mat`` (series ``S1``..``S9`` or ``ST``) and return the ``hs`` struct."""
    return scipy.io.loadmat(Path(raw_dir) / subject / f"HS_{subject}_{series}.mat", simplify_cells=True)["hs"]


def load_alllifts(subject: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load ``<subject>_AllLifts.mat`` as a DataFrame with one row per lift.

    Event columns (``t*``, ``LEDOn``, ``LEDOff``) are seconds from the WS window start;
    ``StartTime`` is the LED-on time in series time.
    """
    p = scipy.io.loadmat(Path(raw_dir) / subject / f"{subject}_AllLifts.mat", simplify_cells=True)["P"]
    return pd.DataFrame(p["AllLifts"], columns=list(p["ColNames"]))


def position_columns(kin_names: list[str] | np.ndarray, sensor: int) -> list[int]:
    """Column indices of the x, y, z position of ``sensor`` (1..4) in a kin array."""
    names = list(kin_names)
    return [names.index(f"P{a}{sensor} - position {a} sensor {sensor}") for a in AXES]
