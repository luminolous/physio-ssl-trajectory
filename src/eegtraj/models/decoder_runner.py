"""LOSO fold loop for the stage 2 decoder (design.md "Stage 2", "Training budget"; tasks.md "Decoder runner").

Per fold (``eegtraj.data.splits``; screening subjects never tested):
1. Fold-level references from the training subjects only: landmark-registered templates (``fold_templates``),
   template_cue (``fit_cue_template``), global-warp centre tau0 / s0 (median training landmarks), auxiliary phase
   labels (``training_aux_classes``).
2. For every lr in the grid: train with AdamW (weight decay 0.01), early stopping on inner-validation MSE (cm^2, the
   criterion of the deep baselines), patience 5, max 30 epochs; keep the best epoch. The lr with the lowest
   inner-validation MSE is selected. Identical for every compared model.
3. Predict the test subject (and score the inner-validation subjects); score with r, R2, NRMSE (per trial, per axis,
   D7) and residual_r / residual_r_specific against template_cue; template_cue itself is scored as the floor.

Encoder settings: ``frozen`` (encoder and gate fixed; tokens precomputed once per subject), ``finetune`` (all
trained). Encoder sources: ``checkpoint`` (stage 1), ``random`` (seeded random init: frozen control, or scratch with
``finetune``), ``none`` (ocular-only).
"""

from __future__ import annotations

import copy
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
from eegtraj.data.lift_streams import LiftStream, SubjectStreams, aux_patch_targets, build_subject_streams
from eegtraj.data.splits import Fold, load_split_config, make_folds
from eegtraj.data.target_view import prior_vector
from eegtraj.data.windows import input_channel_names
from eegtraj.metrics import aggregate_subject, per_trial_metrics
from eegtraj.models.decoder import StreamDecoder, decoder_loss
from eegtraj.models.encoder import CausalEncoder, n_parameters
from eegtraj.models.landmark_template import TemplateBank, fold_templates

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
            for key in ("phi", "weights", "amp"):
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
    folds: list[Fold] = make_folds(split, "loso")
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
    fold_rows, per_subject, floors, saved = [], {}, {}, {}
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
        tau0 = float(tmpl.landmarks_since_cue_s[0])
        s0 = float(tmpl.landmarks_since_cue_s[-1] - tmpl.landmarks_since_cue_s[0])
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
                                 a_max=float(m["a_max"]), tau0=tau0, s0=s0, dropout=float(m["dropout"]))

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
            per_subject[s], floors[s] = scores, floor
            saved[s] = arrays
            logger.info("fold %s test %s: r %.3f (x %.3f y %.3f z %.3f) R2 %.3f NRMSE %.3f rrs %.4f | template_cue r %.3f"
                        " (%.1f s)", fold.name, s, scores["r"], *scores["r_per_axis"].values(), scores["r2"],
                        scores["nrmse"], scores["residual_r_specific"], floor["r"], time.time() - t_pred)
        fold_rows.append({"fold": fold.name, "train_subjects": list(fold.train_subjects),
                          "val_subjects": list(fold.val_subjects), "test_subjects": list(fold.test_subjects),
                          "tau0_s": tau0, "s0_s": s0, "template_counts": {str(k): v for k, v in tmpl.counts.items()},
                          "encoder": enc_info, "parameters": n_params, **report, "inner_val": inner,
                          "test": {s: per_subject[s] for s in fold.test_subjects},
                          "floor_template_cue": {s: floors[s] for s in fold.test_subjects},
                          "seconds": round(time.time() - t_fold, 1)})
        if cfg.get("save_predictions"):
            np.savez_compressed(run_dir / "predictions.npz",
                                **{f"{s}__{k}": v for s, d in saved.items() for k, v in d.items()})

    def agg(table: dict) -> dict:
        # template_cue's residual metrics are NaN by definition (no residual): nan-aware mean
        return {k: {"mean": float(np.nanmean([v[k] for v in table.values()])),
                    "sd": float(np.nanstd([v[k] for v in table.values()], ddof=1)) if len(table) > 1 else None,
                    "n_subjects": len(table)} for k in SCORES}

    metrics = {"name": cfg["name"], "model": m, "encoder": enc_cfg, "decoder_seed": seed, "smoke": smoke,
               "loss_weights": cfg["loss"], "train": cfg["train"],
               "eeg_normalization_note": "transductive: each subject's own label-free EEG statistics (D9)",
               "stream_start": "LED-on - 1.5 s (SSL-17)", "aggregate": agg(per_subject),
               "aggregate_floor_template_cue": agg(floors), "per_subject": per_subject,
               "floor_template_cue": floors, "per_fold": fold_rows, "wall_seconds": round(time.time() - t_start, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    a, f = metrics["aggregate"], metrics["aggregate_floor_template_cue"]
    logger.info("aggregate over %d test subjects: r %.3f, R2 %.3f, NRMSE %.3f, rrs %.4f | template_cue r %.3f",
                a["r"]["n_subjects"], a["r"]["mean"], a["r2"]["mean"], a["nrmse"]["mean"],
                a["residual_r_specific"]["mean"], f["r"]["mean"])
    return metrics


def load_trials_rows(cache_dir: str, subject: str) -> np.ndarray:
    import pandas as pd

    return np.arange(len(pd.read_csv(Path(cache_dir) / subject / "trials.csv")))
