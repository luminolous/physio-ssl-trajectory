"""Tests for eegtraj.data.splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from eegtraj.data.splits import draw_screening, load_split_config, loso_folds, make_folds, within_subject_folds

CONFIG = Path("configs/splits/default.yaml")


@pytest.fixture
def cfg():
    return load_split_config(CONFIG)


def test_recorded_screening_matches_seeded_draw(cfg):
    assert cfg["screening_subjects"] == ["P3", "P6"]
    assert draw_screening(cfg["subjects"], 2, cfg["seed"]) == ["P3", "P6"]


def test_load_rejects_mismatching_screening(tmp_path, cfg):
    bad = {**cfg, "screening_subjects": ["P1", "P2"]}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_split_config(path)


@pytest.mark.parametrize("mode", ["loso", "within_subject"])
def test_screening_subjects_never_used(cfg, mode):
    for fold in make_folds(cfg, mode):
        used = set(fold.train_subjects) | set(fold.val_subjects) | set(fold.test_subjects)
        assert not used & set(cfg["screening_subjects"]), fold.name


def test_loso_structure_and_disjointness(cfg):
    folds = loso_folds(cfg)
    main = [s for s in cfg["subjects"] if s not in cfg["screening_subjects"]]
    assert [f.test_subjects[0] for f in folds] == main  # every main subject is a test fold exactly once
    for f in folds:
        parts = [set(f.train_subjects), set(f.val_subjects), set(f.test_subjects)]
        assert len(parts[1]) == cfg["n_inner_val"] and len(parts[2]) == 1
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        assert parts[0] | parts[1] | parts[2] == set(main)


def test_within_subject_disjoint_series(cfg):
    for f in within_subject_folds(cfg):
        assert not (f.units("train") & f.units("val") or f.units("train") & f.units("test")
                    or f.units("val") & f.units("test"))
        assert set(f.test_series) == {7, 8, 9}


def test_deterministic_given_seed(cfg):
    assert loso_folds(cfg) == loso_folds(cfg)
    other = loso_folds({**cfg, "seed": cfg["seed"] + 1})
    assert [f.val_subjects for f in other] != [f.val_subjects for f in loso_folds(cfg)]


def test_masks_select_rows(cfg):
    trials = pd.DataFrame([(s, k) for s in cfg["subjects"] for k in range(1, 10)], columns=["subject", "series"])
    fold = loso_folds(cfg)[0]
    m = fold.masks(trials)
    assert not (m["train"] & m["test"]).any() and not (m["train"] & m["val"]).any()
    assert set(trials[m["test"]].subject) == set(fold.test_subjects)
    assert not trials[m["train"] | m["val"] | m["test"]].subject.isin(cfg["screening_subjects"]).any()
