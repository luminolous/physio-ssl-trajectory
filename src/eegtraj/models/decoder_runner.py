"""LOSO fold loop for the stage 2 decoder (design.md "Stage 2", "Training budget"; tasks.md "Decoder runner").

Per fold (``eegtraj.data.splits``; screening subjects never tested):
1. Fold-level references from the training subjects only: landmark-registered templates (``fold_templates``),
   template_cue (``fit_cue_template``), cue-phase centre tau0 (median reaction time) and scale s0 (median onset ->
   hand stop duration) of the training lifts (global-warp centre and clock-warp tracker), the phase prior
   (``fit_phase_prior``, SSL-20: phase mean and SD at each time since LED-on over the training lifts whose scoring
   segment contains it), auxiliary phase labels (``training_aux_classes``).
2. For every lr in the grid: train with AdamW (weight decay 0.01), early stopping on inner-validation MSE (cm^2, the
   criterion of the deep baselines), patience 5, max 30 epochs; keep the best epoch. The lr with the lowest
   inner-validation MSE is selected. Identical for every compared model.
3. Predict the test subject (and score the inner-validation subjects); score with r, R2, NRMSE (per trial, per axis,
   D7) and residual_r / residual_r_specific against template_cue; template_cue itself is scored as the floor, and
   the head floor ``cue_phi`` (the warp heads at initialization: expected landmark template under the phase prior at
   real time, uniform mixing, A = 1, Delta = 0; no EEG) on the same samples.
4. Neural-branch occlusion (``occlusion.groups``): the test subject is scored again with the EEG rows of a channel
   group set to 0 (the z-scored mean) in the neural-branch input only; the ocular proxies stay intact (computed
   from the unoccluded EEG). Frozen tokens are recomputed from the occluded EEG. The selected model of every fold is
   saved to ``models/<fold>.pt`` (state dict) when ``save_models`` is true.

Encoder settings: ``frozen`` (encoder and gate fixed; tokens precomputed once per subject), ``finetune`` (all
trained). Encoder sources: ``checkpoint`` (stage 1), ``random`` (seeded random init: frozen control, or scratch with
``finetune``), ``none`` (ocular-only).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.scoring import fit_cue_template, subject_scores
from eegtraj.data.decoder_labels import training_aux_classes
from eegtraj.data.lift_streams import (STREAM_START_BEFORE_LED_S, LiftStream, SubjectStreams, aux_patch_targets,
                                      build_subject_streams, full_stream_lift)
from eegtraj.data.splits import Fold, load_split_config, make_folds
from eegtraj.data.target_view import prior_vector
from eegtraj.data.windows import input_channel_names
from eegtraj.metrics import aggregate_subject, per_trial_metrics
from eegtraj.models.decoder import StreamDecoder, decoder_loss
from eegtraj.models.encoder import CausalEncoder, n_parameters
from eegtraj.preprocessing.reference import CHANNEL_SETS
from eegtraj.models.landmark_template import (LandmarkTemplates, PhasePrior, TemplateBank, expected_template,
                                              fit_phase_prior, fold_templates, lift_landmarks)

logger = logging.getLogger(__name__)
SCORES = ("r", "r2", "nrmse", "residual_r", "residual_r_other_trial", "residual_r_specific")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_encoder(enc_cfg: dict[str, Any], cache_dir: str, subject: str) -> tuple[CausalEncoder | None, dict]:
    """Encoder from a stage 1 checkpoint, a seeded random init, or None. Returns (encoder, info)."""
    source = enc_cfg["source"]
    if source == "none":
        return None, {"source": "none"}
    if source == "checkpoint":
        ckpt = torch.load(enc_cfg["checkpoint"], map_location="cpu", weights_only=False)
        arch = ckpt["encoder"]
        info = {"source": "checkpoint", "checkpoint": enc_cfg["checkpoint"], "variant": ckpt["variant"],
                "pretrain_step": ckpt["step"], "kind": ckpt["kind"]}
    elif source == "random":
        arch = yaml.safe_load(Path(enc_cfg["architecture_config"]).read_text(encoding="utf-8"))
        info = {"source": "random", "random_seed": int(enc_cfg["random_seed"])}
    else:
        raise ValueError(f"unknown encoder source {source!r}")
    prior_cfg = yaml.safe_load(Path(arch["prior_config"]).read_text(encoding="utf-8"))
    prior = torch.from_numpy(prior_vector(prior_cfg, input_channel_names(cache_dir, subject, "broadband", "all")))
    kw = {k: v for k, v in arch.items() if k != "prior_config"}
    if source == "random":
        torch.manual_seed(int(enc_cfg["random_seed"]))  # same construction as scripts/pretrain.py step 0
    enc = CausalEncoder(prior, **kw)
    if source == "checkpoint":
        enc.load_state_dict(ckpt["state_dict"])
    return enc, info


class Batch(dict):
    def to(self, device) -> "Batch":
        return Batch({k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in self.items()})


def collate(lifts: list[LiftStream], cue: dict, aux: dict | None, tokens: dict | None, history: int,
            patch: int) -> Batch:
    """Pad a list of lifts. Padding: zeros after each stream (causal model), out_idx 0 with valid False, aux -1."""
    b = len(lifts)
    n_max = max(l.n_patches for l in lifts)
    s_max = max(len(l.out_idx) for l in lifts)
    t_max = history + patch * n_max
    eeg = torch.zeros(b, lifts[0].eeg.shape[0], t_max)
    eog = torch.zeros(b, 2, t_max)
    out_idx = torch.zeros(b, s_max, dtype=torch.long)
    valid = torch.zeros(b, s_max, dtype=torch.bool)
    y = torch.zeros(b, s_max, 3)
    t_led = torch.zeros(b, s_max)
    cue_t = torch.zeros(b, s_max, 3)
    aux_t = torch.full((b, n_max), -1, dtype=torch.long)
    tok = None
    if tokens is not None:
        d = next(iter(tokens.values())).shape[-1]
        tok = torch.zeros(b, n_max, d)
    for i, l in enumerate(lifts):
        key = (l.subject, l.trial_row)
        t, s = l.eeg.shape[1], len(l.out_idx)
        eeg[i, :, :t] = torch.from_numpy(l.eeg)
        eog[i, :, :t] = torch.from_numpy(l.eog)
        out_idx[i, :s] = torch.from_numpy(l.out_idx)
        valid[i, :s] = True
        y[i, :s] = torch.from_numpy(l.y)
        t_led[i, :s] = torch.from_numpy(l.t_since_led)
        if key in cue:
            cue_t[i, :s] = torch.from_numpy(cue[key])
        if aux is not None and key in aux:
            aux_t[i, :l.n_patches] = torch.from_numpy(aux[key])
        if tok is not None:
            tok[i, :l.n_patches] = tokens[key]
    return Batch(eeg=eeg, eog=eog, out_idx=out_idx, valid=valid, y=y, t_led=t_led, cue=cue_t, aux=aux_t, tokens=tok)


def model_inputs(model: StreamDecoder, batch: Batch) -> dict:
    frozen_tokens = batch["tokens"] is not None and model.encoder is not None and model.freeze_encoder
    return {"eeg": None if frozen_tokens or model.encoder is None else batch["eeg"],
            "eog": batch["eog"] if model.ocular is not None else None, "out_idx": batch["out_idx"],
            "t_since_led": batch["t_led"], "cue": batch["cue"], "tokens": batch["tokens"] if frozen_tokens else None}


@torch.no_grad()
def precompute_tokens(model: StreamDecoder, lifts: list[LiftStream], device, batch: int = 32) -> dict:
    model.eval()
    out = {}
    for i in range(0, len(lifts), batch):
        part = lifts[i:i + batch]
        n_max = max(l.n_patches for l in part)
        t_max = model.history + model.patch * n_max
        eeg = torch.zeros(len(part), part[0].eeg.shape[0], t_max)
        for k, l in enumerate(part):
            eeg[k, :, :l.eeg.shape[1]] = torch.from_numpy(l.eeg)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            tok = model.neural_tokens(eeg.to(device)).float().cpu()
        for k, l in enumerate(part):
            out[(l.subject, l.trial_row)] = tok[k, :l.n_patches]
    return out


def evaluate_mse(model, lifts, cue, tokens, device, cfg) -> float:
    model.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(lifts), int(cfg["batch_lifts"])):
            b = collate(lifts[i:i + int(cfg["batch_lifts"])], cue, None, tokens, model.history, model.patch).to(device)
            with torch.autocast(device.type, torch.bfloat16, enabled=cfg["amp"] == "bf16" and device.type == "cuda"):
                out = model(**model_inputs(model, b))
            err = ((out["y"].float() - b["y"]) ** 2).sum(-1)
            se += float(err[b["valid"]].sum())
            n += int(b["valid"].sum()) * 3
    return se / max(n, 1)


@torch.no_grad()
def predict(model, lifts, cue, tokens, device, cfg) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    model.eval()
    res = {}
    for i in range(0, len(lifts), int(cfg["batch_lifts"])):
        part = lifts[i:i + int(cfg["batch_lifts"])]
        b = collate(part, cue, None, tokens, model.history, model.patch).to(device)
        out = model(**model_inputs(model, b))
        for k, l in enumerate(part):
            s = len(l.out_idx)
            r = {"y": out["y"][k, :s].float().cpu().numpy(),
                 "sigma": torch.exp(0.5 * out["logvar"][k, :s].float()).cpu().numpy()}
            for key in ("phi", "phi_centre", "phi_sigma", "weights", "amp", "delta"):
                if key in out:
                    r[key] = out[key][k, :s].float().cpu().numpy()
            res[(l.subject, l.trial_row)] = r
    return res


def train_fold(make_model, train: list[LiftStream], val: list[LiftStream], cue: dict, aux: dict, tokens, device,
               cfg: dict[str, Any], seed: int) -> tuple[StreamDecoder, dict]:
    """lr grid x early stopping on inner-validation MSE. Returns the selected model (best epoch) and a report."""
    tr = cfg["train"]
    grid, best = [], None
    for lr in tr["lr_grid"]:
        set_seeds(seed)
        model = make_model().to(device)
        params = [q for q in model.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(tr["weight_decay"]))
        rng = np.random.default_rng(seed)
        history, best_state, best_val, bad = [], None, float("inf"), 0
        for epoch in range(int(tr["max_epochs"])):
            t0 = time.time()
            model.train()
            order = rng.permutation(len(train))
            acc: dict[str, float] = {}
            n_batches = 0
            for i in range(0, len(order), int(tr["batch_lifts"])):
                batch = collate([train[j] for j in order[i:i + int(tr["batch_lifts"])]], cue, aux, tokens,
                                model.history, model.patch).to(device)
                with torch.autocast(device.type, torch.bfloat16, enabled=tr["amp"] == "bf16" and device.type == "cuda"):
                    out = model(**model_inputs(model, batch))
                terms = decoder_loss(out, batch["y"], batch["valid"], batch["aux"], model.gate_penalty(), cfg["loss"])
                if not torch.isfinite(terms["loss"]):
                    raise FloatingPointError(f"non-finite loss: { {k: float(v) for k, v in terms.items()} }")
                opt.zero_grad(set_to_none=True)
                terms["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, float(tr["grad_clip"]))
                opt.step()
                for k, v in terms.items():
                    acc[k] = acc.get(k, 0.0) + v.detach().item()
                n_batches += 1
            val_mse = evaluate_mse(model, val, cue, tokens, device, tr)
            row = {"lr": lr, "epoch": epoch + 1, "val_mse": val_mse, "sec": round(time.time() - t0, 1),
                   **{f"train_{k}": v / n_batches for k, v in acc.items()}}
            history.append(row)
            logger.info("lr %g epoch %d: train loss %.4f nll %.4f aux %.4f | val MSE %.3f cm^2 (%.1f s)", lr,
                        epoch + 1, row["train_loss"], row["train_nll"], row["train_aux"], val_mse, row["sec"])
            if val_mse < best_val:
                best_val, bad = val_mse, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
                if bad >= int(tr["patience"]):
                    break
        model.load_state_dict(best_state)
        top = min(history, key=lambda h: h["val_mse"])
        grid.append({"lr": lr, "best_val_mse": top["val_mse"], "best_epoch": top["epoch"], "epochs_run": len(history),
                     "history": history})
        if best is None or top["val_mse"] < best[1]:
            best = (model, top["val_mse"], lr)
    model, val_mse, lr = best
    logger.info("selected lr %g (val MSE %.3f) from %s", lr, val_mse, {g["lr"]: round(g["best_val_mse"], 3) for g in grid})
    return model, {"selected_lr": lr, "selected_val_mse": val_mse, "grid": grid}


def cue_phase_params(train: list) -> tuple[float, float]:
    """tau0 = median reaction time (LED-on -> kinematic onset), s0 = median onset -> hand stop duration (s) over the
    training subjects' lifts."""
    marks = np.concatenate([lift_landmarks(s.trials) for s in train])
    led = np.concatenate([s.trials.ev_LEDOn.to_numpy(np.float64) for s in train])
    return float(np.median(marks[:, 0] - led)), float(np.median(marks[:, -1] - marks[:, 0]))


@torch.no_grad()
def cue_phi_prediction(st: SubjectStreams, tmpl: LandmarkTemplates, prior: PhasePrior) -> dict:
    """Head floor per lift (the warp heads at initialization): expected uniform mixture of the landmark templates
    under phi ~ N(mu(t), sd(t)^2), t = time since LED-on."""
    bank = TemplateBank.from_templates(tmpl)
    uniform = torch.full((len(tmpl.conditions),), 1.0 / len(tmpl.conditions))
    out = {}
    for l in st.lifts:
        mu, sd = prior(l.t_since_led.astype(np.float64))
        mu, sd = torch.from_numpy(mu).float(), torch.from_numpy(sd).float()
        y = expected_template(bank, mu, sd, uniform.expand(mu.shape[0], -1))
        out[(l.subject, l.trial_row)] = {"y": y.numpy().astype(np.float32)}
    return out


OCCLUSION_GROUPS = {**{k: CHANNEL_SETS[k] for k in ("prefrontal", "frontal", "contralateral", "ipsilateral")},
                    "occipital": ("O1", "Oz", "O2", "PO9", "PO10")}


def cue_at(template, t_since_led: np.ndarray) -> np.ndarray:
    """template_cue (S, 3) at arbitrary times since LED-on (same binning and edge hold as ``TemplateCueBaseline``)."""
    b = np.round(np.asarray(t_since_led) * template.fs_hz).astype(np.int64) - template.bin0
    return template.template[:, np.clip(b, 0, template.template.shape[1] - 1)].T.astype(np.float32)


def full_stream_arrays(model, st_full: SubjectStreams, scored: dict, template, frozen_tokens: bool, device,
                       cfg: dict) -> tuple[dict, float]:
    """Predictions on the whole stream of every lift (first patch end to the end of the lift window), from streams
    built with ``to_lift_end``. Output times depend on LED-on and the lift window only, not on the kinematic onset.

    Returns arrays ``full_y``, ``full_delta`` (if any), ``full_trial_row``, ``full_t_led`` (concatenated in lift order)
    and the max |difference| to the scored predictions ``scored`` at the scored samples (causality check)."""
    lifts = [full_stream_lift(l, model.patch) for l in st_full.lifts]
    cue = {(l.subject, l.trial_row): cue_at(template, l.t_since_led) for l in lifts}
    tokens = precompute_tokens(model, lifts, device) if frozen_tokens else None
    preds = predict(model, lifts, cue, tokens, device, cfg)
    out = {"full_y": [], "full_delta": [], "full_trial_row": [], "full_t_led": []}
    max_diff = 0.0
    for l_full, l in zip(lifts, st_full.lifts):
        p = preds[(l.subject, l.trial_row)]
        out["full_y"].append(p["y"])
        if "delta" in p:
            out["full_delta"].append(p["delta"])
        out["full_trial_row"].append(np.full(len(l_full.out_idx), l.trial_row, np.int64))
        out["full_t_led"].append(l_full.t_since_led)
        ref = scored[(l.subject, l.trial_row)]["y"]
        max_diff = max(max_diff, float(np.abs(p["y"][l.out_idx - l_full.out_idx[0]] - ref).max()))
    return {k: np.concatenate(v) for k, v in out.items() if v}, max_diff


def occluded_lifts(lifts: list[LiftStream], rows: list[int]) -> list[LiftStream]:
    """Copies of ``lifts`` with the EEG rows ``rows`` set to 0 (neural-branch input only; ``eog`` unchanged)."""
    out = []
    for l in lifts:
        eeg = l.eeg.copy()
        eeg[rows] = 0.0
        out.append(dataclasses.replace(l, eeg=eeg))
    return out


def score_subject(st: SubjectStreams, preds: dict, template, ctx: FitContext, cfg: dict) -> tuple[dict, dict, dict]:
    """Scores of one subject's predictions and of the template_cue floor. Also returns the arrays to save."""
    s = st.samples
    pos = np.concatenate([l.sample_pos for l in st.lifts])
    order = np.argsort(pos)
    pos = pos[order]
    mask = np.zeros(s.sample_idx.size, bool)
    mask[pos] = True
    if not np.array_equal(np.flatnonzero(mask), pos):
        raise RuntimeError("duplicate scored samples")
    arrays = {k: np.concatenate([preds[(l.subject, l.trial_row)][k] for l in st.lifts])[order]
              for k in preds[(st.lifts[0].subject, st.lifts[0].trial_row)]}
    sc = cfg["scoring"]
    model_scores = subject_scores(s, mask, arrays["y"], template, ctx, int(sc["seed"]), int(sc["n_permutations"]))
    per_axis = aggregate_subject(per_trial_metrics(s.target[:, s.sample_idx[mask]].T, arrays["y"], s.trial_row[mask]))
    model_scores["r_per_axis"] = per_axis["r"]["per_axis"]
    model_scores["n_trials"] = int(np.unique(s.trial_row[mask]).size)
    cue = template.predict(s, mask, ctx)
    floor = subject_scores(s, mask, cue, template, ctx, int(sc["seed"]), int(sc["n_permutations"]))
    arrays.update(y_true=s.target[:, s.sample_idx[mask]].T.astype(np.float32), sample_idx=s.sample_idx[mask],
                  trial_row=s.trial_row[mask])
    return model_scores, floor, arrays


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["seed"])
    split = load_split_config(cfg["split_config"])
    folds: list[Fold] = make_folds(split, cfg.get("folds", "loso"))
    smoke = cfg.get("smoke_run", False)
    if smoke:
        folds = folds[:int(cfg["smoke"]["n_folds"])]
        cfg["train"]["max_epochs"] = int(cfg["smoke"]["max_epochs"])
    m = cfg["model"]
    if m["setting"] not in ("frozen", "finetune"):
        raise ValueError("setting must be frozen or finetune")
    enc_cfg = m["encoder"] if m["neural"] else {"source": "none"}
    logger.info("model %s: head %s, neural %s, ocular %s, setting %s, encoder %s, decoder seed %d, device %s",
                cfg["name"], m["head"], m["neural"], m["ocular"], m["setting"], enc_cfg, seed, device)

    streams: dict[str, SubjectStreams] = {}

    def get(subject: str) -> SubjectStreams:
        if subject not in streams:
            keep = None
            if smoke:
                n = len(np.unique(load_trials_rows(cfg["cache_dir"], subject)))
                keep = np.arange(n) < int(cfg["smoke"]["trials_per_subject"])
            streams[subject] = build_subject_streams(cfg["cache_dir"], subject, keep_trial=keep)
        return streams[subject]

    token_cache: dict = {}
    fold_rows, per_subject, floors, head_floors, saved, occlusion = [], {}, {}, {}, {}, {}
    full_stream_check: dict[str, float] = {}
    occ_groups = list(cfg.get("occlusion", {}).get("groups", [])) if m["neural"] else []
    channel_names = input_channel_names(cfg["cache_dir"], "P1", "broadband", "all")
    if cfg.get("save_models"):
        (run_dir / "models").mkdir(exist_ok=True)
    for fold in folds:
        t_fold = time.time()
        names = sorted(set(fold.train_subjects) | set(fold.val_subjects) | set(fold.test_subjects),
                       key=lambda s: int(s[1:]))
        subj = {s: get(s) for s in names}
        train_samples = [subj[s].samples for s in fold.train_subjects]
        masks_all = {s: np.ones(subj[s].samples.sample_idx.size, bool) for s in names}
        ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=seed,
                         train_masks={s: masks_all[s] for s in fold.train_subjects}, val_masks={}, logger=logger)
        template = fit_cue_template(train_samples, ctx)
        tmpl = fold_templates([subj[s].samples for s in names], fold, conditions=tuple(cfg["templates"]["conditions"]),
                              n_grid=int(cfg["templates"]["n_grid"]))
        tau0, s0 = cue_phase_params(train_samples)
        prior = fit_phase_prior([subj[s].samples for s in names], fold.train_subjects, tmpl.fractions)
        prior_dict = {"t0": prior.t0, "dt": prior.dt, "mu": prior.mu, "sd": prior.sd} if m.get("phase_prior", True) else None
        cue = {}
        for s in names:
            pred = template.predict(subj[s].samples, masks_all[s], ctx)
            for l in subj[s].lifts:
                cue[(s, l.trial_row)] = pred[l.sample_pos].astype(np.float32)
        aux = {}
        for s in fold.train_subjects:
            classes = training_aux_classes(cfg["cache_dir"], s, fold)
            for l in subj[s].lifts:
                aux[(s, l.trial_row)] = aux_patch_targets(l, classes)

        def make_model() -> StreamDecoder:
            enc, _ = build_encoder(enc_cfg, cfg["cache_dir"], names[0])
            return StreamDecoder(m["head"], enc, bool(m["ocular"]), TemplateBank.from_templates(tmpl),
                                 freeze_encoder=m["setting"] == "frozen", d_fused=int(m["d_fused"]),
                                 d_ocular=int(m["d_ocular"]), ocular_channels=int(m["ocular_channels"]),
                                 a_max=float(m["a_max"]), tau0=tau0, s0=s0,
                                 clock_log_range=float(m["clock_log_range"]), stream_start_s=-STREAM_START_BEFORE_LED_S,
                                 phase_prior=prior_dict, sigma_log_range=float(m["sigma_log_range"]),
                                 dropout=float(m["dropout"]))

        probe_model = make_model().to(device)
        _, enc_info = build_encoder(enc_cfg, cfg["cache_dir"], names[0])
        tokens = None
        if probe_model.encoder is not None and m["setting"] == "frozen":
            for s in names:
                if s not in token_cache:
                    token_cache[s] = precompute_tokens(probe_model, subj[s].lifts, device)
            tokens = {k: v for s in names for k, v in token_cache[s].items()}
        n_params = {"total": sum(q.numel() for q in probe_model.parameters()), "trainable": n_parameters(probe_model)}
        del probe_model
        train = [l for s in fold.train_subjects for l in subj[s].lifts]
        val = [l for s in fold.val_subjects for l in subj[s].lifts]
        logger.info("fold %s: train %s (%d lifts), val %s (%d lifts), test %s; tau0 %.2f s, s0 %.2f s; params %s",
                    fold.name, fold.train_subjects, len(train), fold.val_subjects, len(val), fold.test_subjects, tau0,
                    s0, n_params)
        model, report = train_fold(make_model, train, val, cue, aux, tokens, device, cfg, seed)

        inner = {}
        for s in fold.val_subjects:
            preds = predict(model, subj[s].lifts, cue, tokens, device, cfg["train"])
            inner[s], _, _ = score_subject(subj[s], preds, template, ctx, cfg)
        for s in fold.test_subjects:
            t_pred = time.time()
            preds = predict(model, subj[s].lifts, cue, tokens, device, cfg["train"])
            scores, floor, arrays = score_subject(subj[s], preds, template, ctx, cfg)
            head_floors[s], _, _ = score_subject(subj[s], cue_phi_prediction(subj[s], tmpl, prior), template, ctx, cfg)
            per_subject[s], floors[s] = scores, floor
            saved[s] = arrays
            if cfg.get("save_full_stream"):  # flexibility features on the whole stream (design.md "Experiment 3")
                keep = np.isin(np.arange(len(subj[s].samples.trials)), [l.trial_row for l in subj[s].lifts])
                st_full = build_subject_streams(cfg["cache_dir"], s, keep_trial=keep, to_lift_end=True)
                full, diff = full_stream_arrays(model, st_full, preds, template, tokens is not None, device, cfg["train"])
                saved[s].update(full)
                full_stream_check[s] = diff
                logger.info("fold %s test %s: full-stream predictions %d samples, max |diff| at scored samples %.2e",
                            fold.name, s, full["full_y"].shape[0], diff)
            occlusion[s] = {}
            for g in occ_groups:
                rows = [channel_names.index(c) for c in OCCLUSION_GROUPS[g]]
                occ = occluded_lifts(subj[s].lifts, rows)
                occ_tokens = (precompute_tokens(model, occ, device) if tokens is not None else None)
                occ_scores, _, _ = score_subject(subj[s], predict(model, occ, cue, occ_tokens, device, cfg["train"]),
                                                 template, ctx, cfg)
                occlusion[s][g] = {"scores": occ_scores, "drop": {k: scores[k] - occ_scores[k] for k in SCORES}}
                logger.info("fold %s test %s occlude %s (%s): r %.3f (drop %.4f), rrs %.4f (drop %.4f)", fold.name, s, g,
                            ",".join(OCCLUSION_GROUPS[g]), occ_scores["r"], scores["r"] - occ_scores["r"],
                            occ_scores["residual_r_specific"],
                            scores["residual_r_specific"] - occ_scores["residual_r_specific"])
            logger.info("fold %s test %s: r %.3f (x %.3f y %.3f z %.3f) R2 %.3f NRMSE %.3f rrs %.4f | template_cue r %.3f"
                        " | head floor cue_phi r %.3f (%.1f s)", fold.name, s, scores["r"], *scores["r_per_axis"].values(),
                        scores["r2"], scores["nrmse"], scores["residual_r_specific"], floor["r"], head_floors[s]["r"],
                        time.time() - t_pred)
        fold_rows.append({"fold": fold.name, "train_subjects": list(fold.train_subjects),
                          "val_subjects": list(fold.val_subjects), "test_subjects": list(fold.test_subjects),
                          "tau0_s": tau0, "s0_s": s0,
                          "phase_prior": {"t0": prior.t0, "dt": prior.dt, "mu": prior.mu.tolist(), "sd": prior.sd.tolist()}, "template_counts": {str(k): v for k, v in tmpl.counts.items()},
                          "encoder": enc_info, "parameters": n_params, **report, "inner_val": inner,
                          "test": {s: per_subject[s] for s in fold.test_subjects},
                          "floor_template_cue": {s: floors[s] for s in fold.test_subjects},
                          "floor_cue_phi": {s: head_floors[s] for s in fold.test_subjects},
                          "seconds": round(time.time() - t_fold, 1)})
        if cfg.get("save_models"):
            torch.save({"fold": fold.name, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "tau0": tau0, "s0": s0, "phase_prior": prior_dict, "template_values": tmpl.values,
                        "encoder": enc_info, "model_cfg": m}, run_dir / "models" / f"{fold.name}.pt")
        if cfg.get("save_predictions"):
            np.savez_compressed(run_dir / "predictions.npz",
                                **{f"{s}__{k}": v for s, d in saved.items() for k, v in d.items()})

    def agg(table: dict) -> dict:
        # template_cue's residual metrics are NaN by definition (no residual): nan-aware mean
        return {k: {"mean": float(np.nanmean([v[k] for v in table.values()])),
                    "sd": float(np.nanstd([v[k] for v in table.values()], ddof=1)) if len(table) > 1 else None,
                    "n_subjects": len(table)} for k in SCORES}

    metrics = {"name": cfg["name"], "folds": cfg.get("folds", "loso"), "model": m, "encoder": enc_cfg, "decoder_seed": seed, "smoke": smoke,
               "loss_weights": cfg["loss"], "train": cfg["train"],
               "eeg_normalization_note": "transductive: each subject's own label-free EEG statistics (D9)",
               "stream_start": "LED-on - 1.5 s (SSL-17)", "aggregate": agg(per_subject),
               "aggregate_floor_template_cue": agg(floors), "per_subject": per_subject,
               "floor_template_cue": floors, "aggregate_floor_cue_phi": agg(head_floors), "floor_cue_phi": head_floors,
               "occlusion_groups": {g: list(OCCLUSION_GROUPS[g]) for g in occ_groups}, "occlusion": occlusion,
               "aggregate_occlusion_drop": {g: agg({s: occlusion[s][g]["drop"] for s in occlusion}) for g in occ_groups},
               "full_stream_max_abs_diff": full_stream_check,
               "per_fold": fold_rows, "wall_seconds": round(time.time() - t_start, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    a, f = metrics["aggregate"], metrics["aggregate_floor_template_cue"]
    logger.info("aggregate over %d test subjects: r %.3f, R2 %.3f, NRMSE %.3f, rrs %.4f | template_cue r %.3f | head "
                "floor cue_phi r %.3f", a["r"]["n_subjects"], a["r"]["mean"], a["r2"]["mean"], a["nrmse"]["mean"],
                a["residual_r_specific"]["mean"], f["r"]["mean"], metrics["aggregate_floor_cue_phi"]["r"]["mean"])
    return metrics


def load_trials_rows(cache_dir: str, subject: str) -> np.ndarray:
    import pandas as pd

    return np.arange(len(pd.read_csv(Path(cache_dir) / subject / "trials.csv")))
