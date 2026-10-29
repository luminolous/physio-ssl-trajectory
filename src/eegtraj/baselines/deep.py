"""Deep supervised decoders (baselines.md: eegnet, shallowconvnet, cnn-lstm).

Shared setup: input EEG window (C, history) per D3, broadband per D4, output = target at
the last window sample (seq-to-one), MSE, AdamW, early stopping on inner validation,
mixed precision. Each run trains one seed; the 3-seed protocol is run as 3 runs
(``--seed``) and summarized afterwards.
"""

from __future__ import annotations

import warnings

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


ARCHITECTURES = {"eegnet": build_eegnet}


class DeepBaseline(Baseline):
    """Params: ``model`` (architecture kwargs) and ``train`` (TrainConfig fields)."""

    architecture: str = ""

    def __init__(self, model: dict | None = None, train: dict | None = None, **params):
        super().__init__(model=model, train=train, **params)
        self.model_kwargs = dict(model or {})
        self.train_cfg = TrainConfig(**(train or {}))

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        if ctx.smoke:
            self.train_cfg = TrainConfig(**{**self.train_cfg.__dict__, **SMOKE_TRAIN})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx = ctx
        n_chans = train[0].eeg.shape[0]
        self.model = ARCHITECTURES[self.architecture](n_chans=n_chans, n_times=ctx.history, **self.model_kwargs)
        self.n_params = count_parameters(self.model)
        tr = GpuWindows(train, ctx.train_masks, ctx.eeg_norm, ctx.target_norm, ctx.history, self.device)
        va = GpuWindows(val, ctx.val_masks, ctx.eeg_norm, ctx.target_norm, ctx.history, self.device)
        ctx.logger.info("%s: %d parameters, device %s, %d train windows, %d val windows, config %s",
                        self.architecture, self.n_params, self.device, len(tr), len(va), self.train_cfg)
        self.model, self.history = train_model(self.model, tr, va, self.train_cfg, ctx.seed, ctx.logger)
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
        return {"architecture": self.architecture, "n_parameters": self.n_params, "model_kwargs": self.model_kwargs,
                "train_config": self.train_cfg.__dict__, "best_epoch": best["epoch"],
                "best_val_mse": best["val_loss"], "epochs_run": len(self.history), "history": self.history}


@register("eegnet")
class EEGNetBaseline(DeepBaseline):
    architecture = "eegnet"
