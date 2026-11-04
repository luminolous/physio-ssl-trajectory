"""Tests for deep-model grid selection. Training is replaced by a stub, so no GPU time is used."""

import json
import logging

import pytest
import torch

from eegtraj.baselines import deep
from eegtraj.baselines.base import FitContext, make_baseline
from eegtraj.baselines.deep import expand_grid, load_selection
from tests.test_trainer import make, unit


VAL_MSE = {1e-4: 0.9, 3e-4: 0.7, 1e-3: 0.8}


def stub_training(monkeypatch):
    """Replace training: val MSE depends only on lr. Records lr and the initial weights of each call."""
    calls = []

    def fake_train(model, train, val, cfg, seed, log):
        calls.append({"lr": cfg.lr, "init": next(model.parameters()).detach().clone(), "weight_decay": cfg.weight_decay})
        return model, [{"epoch": 1, "train_loss": 1.0, "val_loss": VAL_MSE[cfg.lr], "seconds": 0.0}]

    monkeypatch.setattr(deep, "train_model", fake_train)
    monkeypatch.setitem(deep.ARCHITECTURES, "eegnet", lambda n_chans, n_times, **kw: torch.nn.Linear(n_chans, 3))
    return calls


def fit(params, seed=1, fold_name="loso_P5"):
    subjects, masks, norms = make(["P1", "P2"])
    ctx = FitContext(eeg_norm=norms, target_norm=unit(3), history=4, lag_step=2, seed=seed, train_masks=masks,
                     val_masks=masks, logger=logging.getLogger("test"), fold_name=fold_name)
    model = make_baseline("eegnet", params)
    model.fit(subjects[:1], subjects[1:], ctx)
    return model


def test_expand_grid():
    assert expand_grid(None) == []
    assert expand_grid({"lr": [1, 2], "weight_decay": [3]}) == [{"lr": 1, "weight_decay": 3}, {"lr": 2, "weight_decay": 3}]


def test_grid_selects_lowest_val_mse_from_same_initial_weights(monkeypatch):
    calls = stub_training(monkeypatch)
    model = fit({"grid": {"lr": list(VAL_MSE)}, "train": {"weight_decay": 0.5}})
    assert [c["lr"] for c in calls] == list(VAL_MSE)
    assert model.selected_train == {"lr": 3e-4} and model.train_cfg.lr == 3e-4
    assert all(c["weight_decay"] == 0.5 for c in calls)
    assert all(torch.equal(c["init"], calls[0]["init"]) for c in calls)
    info = model.info()
    assert info["best_val_mse"] == 0.7 and len(info["grid_results"]) == 3 and info["selected_train"] == {"lr": 3e-4}


def test_without_grid_trains_once_and_adds_no_tuning_fields(monkeypatch):
    calls = stub_training(monkeypatch)
    model = fit({"train": {"lr": 1e-3}})
    assert len(calls) == 1 and "selected_train" not in model.info()


def grid_run(tmp_path, seed=1, smoke=False, baseline="eegnet"):
    metrics = {"baseline": baseline, "smoke": smoke, "seed": seed,
               "per_fold": [{"fold": "loso_P5", "baseline_info": {"selected_train": {"lr": 3e-4}}}]}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return tmp_path


def test_selection_run_trains_only_the_chosen_configuration(monkeypatch, tmp_path):
    calls = stub_training(monkeypatch)
    model = fit({"selection_run": str(grid_run(tmp_path)), "train": {}}, seed=2)
    assert [c["lr"] for c in calls] == [3e-4]
    assert model.info()["selection_run"] == str(tmp_path)


@pytest.mark.parametrize("kwargs", [{"seed": 2}, {"smoke": True}, {"baseline": "cnn_lstm"}])
def test_load_selection_rejects_a_run_that_does_not_fit(tmp_path, kwargs):
    with pytest.raises(ValueError):
        load_selection(grid_run(tmp_path, **kwargs), "eegnet", "loso_P5", seed=2, smoke=False)


def test_load_selection_needs_the_fold(tmp_path):
    with pytest.raises(KeyError):
        load_selection(grid_run(tmp_path), "eegnet", "loso_P7", seed=2, smoke=False)
