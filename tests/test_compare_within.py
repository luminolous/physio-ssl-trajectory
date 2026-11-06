"""Test for rescoring LOSO predictions on selected series."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    spec = importlib.util.spec_from_file_location("compare_within_subject", Path("scripts/compare_within_subject.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scores_use_only_the_selected_series():
    t = np.linspace(0, 6, 50)
    base = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    y_true = np.concatenate([base, base])
    y_pred = np.concatenate([-base, base])  # wrong in the first trial, perfect in the second
    rows = np.repeat([0, 1], 50)
    series = np.array([1, 7])
    scores = module().scores_on_series(y_true, y_pred, rows, series, (7, 8, 9))
    assert scores["r"] == pytest.approx(1.0)
    assert scores["r2"] == pytest.approx(1.0)
