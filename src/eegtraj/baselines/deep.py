"""Deep supervised decoders (baselines.md: eegnet, shallowconvnet, cnn-lstm).

Shared setup: input EEG window (C, history) per D3, broadband per D4, output = target at
the last window sample (seq-to-one), MSE, AdamW, early stopping on inner validation,
mixed precision. Each run trains one seed; the 3-seed protocol is run as 3 runs
(``--seed``) and summarized afterwards.

Tuning (optional): a ``grid`` of TrainConfig values is searched per fold, keeping the lowest
inner-validation MSE. A later seed reuses those choices through ``selection_run`` (the run
directory of the grid seed) and trains only the selected configuration.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from eegtraj.baselines.base import Baseline, FitContext, register
from eegtraj.data.windows import SubjectSamples
from eegtraj.models.trainer import GpuWindows, TrainConfig, count_parameters, predict_windows, train_model

SMOKE_TRAIN = dict(max_epochs=2, samples_per_epoch=20_000, val_samples=10_000)


def build_eegnet(n_chans: int, n_times: int, **kwargs) -> torch.nn.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from braindecode.models import EEGNet
    return EEGNet(n_chans=n_chans, n_outputs=3, n_times=n_times, **kwargs)


def build_shallowconvnet(n_chans: int, n_times: int, **kwargs) -> torch.nn.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from braindecode.models import ShallowFBCSPNet
    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=3, n_times=n_times, final_conv_length="auto", **kwargs)


def build_cnn_lstm(n_chans: int, n_times: int, **kwargs) -> torch.nn.Module:
    from eegtraj.models.cnn_lstm import CNNLSTM

    return CNNLSTM(n_chans=n_chans, n_times=n_times, n_outputs=3, **kwargs)


ARCHITECTURES = {"eegnet": build_eegnet, "shallowconvnet": build_shallowconvnet, "cnn_lstm": build_cnn_lstm}


def expand_grid(grid: dict[str, list] | None) -> list[dict]:
    """All combinations of a TrainConfig value grid, e.g. {"lr": [1e-4, 1e-3]} -> [{"lr": 1e-4}, {"lr": 1e-3}]."""
    if not grid:
        return []
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def load_selection(run_dir: str | Path, architecture: str, fold_name: str, seed: int, smoke: bool) -> dict:
    """The TrainConfig values a grid run selected for ``fold_name``. Raises if the run does not fit this one."""
    m = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    if m["baseline"] != architecture or bool(m["smoke"]) != smoke or m["seed"] == seed:
        raise ValueError(f"selection run {run_dir} (baseline {m['baseline']}, smoke {m['smoke']}, seed {m['seed']}) "
                         f"does not fit this run (baseline {architecture}, smoke {smoke}, seed {seed})")
    for fold in m["per_fold"]:
        if fold["fold"] == fold_name:
            return fold["baseline_info"]["selected_train"]
    raise KeyError(f"selection run {run_dir} has no fold {fold_name}")


class DeepBaseline(Baseline):
    """Params: ``model`` (architecture kwargs), ``train`` (TrainConfig fields), and optionally
    ``grid`` (TrainConfig values searched on inner validation) or ``selection_run`` (grid run to reuse)."""

    architecture: str = ""

    def __init__(self, model: dict | None = None, train: dict | None = None, grid: dict[str, list] | None = None,
                 selection_run: str | None = None, **params):
        super().__init__(model=model, train=train, grid=grid, selection_run=selection_run, **params)
        self.model_kwargs = dict(model or {})
        self.train_cfg = TrainConfig(**(train or {}))
        self.grid = expand_grid(grid)
        self.selection_run = selection_run

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        if ctx.smoke:
            self.train_cfg = TrainConfig(**{**self.train_cfg.__dict__, **SMOKE_TRAIN})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx = ctx
        n_chans = train[0].eeg.shape[0]
        tuned = bool(self.grid or self.selection_run)
        if self.selection_run:
            candidates = [load_selection(self.selection_run, self.architecture, ctx.fold_name, ctx.seed, ctx.smoke)]
        else:
            candidates = self.grid or [{}]
        tr = GpuWindows(train, ctx.train_masks, ctx.eeg_norm, ctx.target_norm, ctx.history, self.device)
        va = GpuWindows(val, ctx.val_masks, ctx.eeg_norm, ctx.target_norm, ctx.history, self.device)
        self.grid_results, base, best_val = [], self.train_cfg.__dict__.copy(), float("inf")
        for overrides in candidates:
            if tuned:  # every candidate starts from the same initial weights
                torch.manual_seed(ctx.seed)
            model = ARCHITECTURES[self.architecture](n_chans=n_chans, n_times=ctx.history, **self.model_kwargs)
            self.n_params = count_parameters(model)
            cfg = TrainConfig(**{**base, **overrides})
            ctx.logger.info("%s: %d parameters, device %s, %d train windows, %d val windows, config %s",
                            self.architecture, self.n_params, self.device, len(tr), len(va), cfg)
            model, history = train_model(model, tr, va, cfg, ctx.seed, ctx.logger)
            top = min(history, key=lambda h: h["val_loss"])
            self.grid_results.append({"train": overrides, "best_val_mse": top["val_loss"], "best_epoch": top["epoch"],
                                      "epochs_run": len(history), "history": history})
            if not self.grid_results[:-1] or top["val_loss"] < best_val:
                best_val, self.model, self.history, self.train_cfg = top["val_loss"], model, history, cfg
                self.selected_train = overrides
            del model
        if tuned:
            ctx.logger.info("%s selected %s (val MSE %.4f) from %s", ctx.fold_name, self.selected_train, best_val,
                            {str(r["train"]): round(r["best_val_mse"], 4) for r in self.grid_results})
        del tr, va
        if self.device.type == "cuda":
            ctx.logger.info("peak GPU memory %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.empty_cache()

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext, repeat: int = 0) -> np.ndarray:
        w = GpuWindows([subject], {subject.subject: mask}, ctx.eeg_norm, ctx.target_norm, ctx.history, self.device)
        z = predict_windows(self.model, w, self.train_cfg)
        return ctx.target_norm.inverse(z.T.astype(np.float32)).T

    def info(self) -> dict:
        best = min(self.history, key=lambda h: h["val_loss"])
        out = {"architecture": self.architecture, "n_parameters": self.n_params, "model_kwargs": self.model_kwargs,
               "train_config": self.train_cfg.__dict__, "best_epoch": best["epoch"],
               "best_val_mse": best["val_loss"], "epochs_run": len(self.history), "history": self.history}
        if self.grid or self.selection_run:
            out.update(selected_train=self.selected_train, grid_results=self.grid_results,
                       selection_run=self.selection_run)
        return out


@register("eegnet")
class EEGNetBaseline(DeepBaseline):
    architecture = "eegnet"


@register("shallowconvnet")
class ShallowConvNetBaseline(DeepBaseline):
    architecture = "shallowconvnet"


@register("cnn_lstm")
class CNNLSTMBaseline(DeepBaseline):
    architecture = "cnn_lstm"
