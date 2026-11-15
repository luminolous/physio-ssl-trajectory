"""Tests for the stage 2 decoder: causal interpolation, phase tracker, warp heads, causality, labels, loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eegtraj.data.decoder_labels import LabelLeakageError, patch_aux_targets, training_aux_classes
from eegtraj.data.splits import Fold
from eegtraj.models.decoder import (HEADS, AUX_CLASSES, StreamDecoder, aux_classes, causal_interp, decoder_loss,
                                    tracker_phase)
from eegtraj.models.encoder import CausalEncoder
from eegtraj.models.landmark_template import LandmarkTemplates, TemplateBank, time_to_phase

CACHE = Path("dataset/cache/98f5fa6e66")
H, P = 180, 20  # history and patch samples


def small_encoder() -> CausalEncoder:
    torch.manual_seed(0)
    prior = torch.full((32,), 0.6)
    return CausalEncoder(prior, d_model=32, n_layers=2, n_heads=2, mlp_width=64, dropout=0.0)


def templates() -> LandmarkTemplates:
    grid = np.linspace(0, 1, 201)
    shape = np.stack([10 * np.sin(np.pi * grid), 5 * grid ** 2, 3 * np.sin(2 * np.pi * grid)])
    values = np.stack([shape * k for k in (0.9, 1.0, 1.1)]).astype(np.float32)  # (3 conditions, 3, G)
    return LandmarkTemplates(conditions=(165, 330, 660), grid=grid, values=values,
                             fractions=np.array([0.0, 0.18, 0.27, 0.83, 1.0]),
                             landmarks_since_cue_s=np.array([0.4, 1.4, 1.9, 5.1, 6.0]),
                             counts={165: 1, 330: 1, 660: 1}, subjects=("P1",))


def decoder(head: str, neural: bool = True, ocular: bool = True, freeze: bool = False) -> StreamDecoder:
    torch.manual_seed(1)
    bank = TemplateBank.from_templates(templates())
    dec = StreamDecoder(head, small_encoder() if neural else None, ocular, bank, freeze_encoder=freeze, d_fused=32,
                        dropout=0.0)
    return dec.eval()


def stream(b: int, n_patches: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    t = H + P * n_patches
    return torch.randn(b, 32, t, generator=g), torch.randn(b, 2, t, generator=g)


def test_causal_interp_uses_only_finished_patches():
    vals = torch.tensor([[[0.0], [10.0], [20.0], [40.0]]])  # patch ends at 19, 39, 59, 79
    idx = torch.tensor([[0, 19, 29, 39, 49, 59, 79, 99, 150]])
    out = causal_interp(vals, idx, P)[0, :, 0].tolist()
    assert out == pytest.approx([0.0, 0.0, 0.0, 0.0, 5.0, 10.0, 20.0, 40.0, 40.0])
    for s in range(P - 1, 100):  # from e_0 on, a value at s never changes when a patch ending after s changes
        later = (torch.arange(4) * P + P - 1) > s
        v2 = vals.clone()
        v2[0, later] += 100.0
        assert torch.allclose(causal_interp(v2, torch.tensor([[s]]), P), causal_interp(vals, torch.tensor([[s]]), P))


def test_tracker_phase_is_monotone_bounded_and_integrates_the_rate():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        rate = torch.rand(3, 40, generator=g) * 2.0
        idx = torch.arange(0, 40 * P + 30).expand(3, -1)
        phi, phi_end = tracker_phase(rate, idx, P, 100.0)
        assert (phi[:, 1:] >= phi[:, :-1] - 1e-7).all() and phi.min() >= 0 and phi.max() <= 1
        assert torch.allclose(phi_end[:, 0], torch.zeros(3))
        assert torch.allclose(phi[:, P - 1], torch.zeros(3))  # phi(e_0) = 0
        expect = torch.clamp(rate[:, 0] * 0.2, max=1.0)
        assert torch.allclose(phi[:, 2 * P - 1], expect, atol=1e-6)  # phi(e_1) = rate_0 * 0.2 s


def test_warp_head_with_true_phase_reproduces_the_template():
    tmpl = templates()
    marks = np.array([2.4, 3.5, 4.0, 7.1, 8.0])  # one lift's landmarks (window time, s)
    fs, start = 100.0, 0.5  # stream starts 1.5 s before LED-on (2.0 s)
    n_patches = 45
    t = start + (np.arange(n_patches * P) + 1) / fs
    seg = np.flatnonzero((t >= marks[0] - 0.5) & (t <= marks[-1]))
    phi_true = time_to_phase(t[seg], marks, tmpl.fractions)
    for c, cond in enumerate(tmpl.conditions):
        y_true = tmpl(phi_true, cond)
        dec = decoder("proposed")  # A = 1 and Delta = 0 at initialization (zero-initialized output layers)
        eeg, eog = stream(1, n_patches)
        w = torch.zeros(1, seg.size, 3)
        w[..., c] = 1.0
        out = dec(eeg, eog, torch.from_numpy(seg)[None], phi_override=torch.tensor(phi_true, dtype=torch.float32)[None],
                  weights_override=w)
        assert torch.allclose(out["amp"], torch.ones_like(out["amp"]))
        assert np.abs(out["y"][0].detach().numpy() - y_true).max() < 1e-3


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("branches", [(True, True), (True, False), (False, True)])
def test_full_decoder_is_causal(head, branches):
    dec = decoder(head, *branches)
    with torch.no_grad():  # zero-initialized output layers would make y independent of the input
        for q in dec.out.parameters():
            q.normal_(std=0.3)
    eeg, eog = stream(2, 20)
    idx = torch.arange(0, 20 * P).expand(2, -1)
    t_led = (idx + 1) / 100.0 - 1.5
    cue = torch.randn(2, idx.shape[1], 3)
    base = dec(eeg, eog, idx, t_led, cue)
    for k in (37, 140, 333):
        e2, o2 = eeg.clone(), eog.clone()
        e2[..., H + k:] = torch.randn_like(e2[..., H + k:]) * 3
        o2[..., H + k:] = torch.randn_like(o2[..., H + k:]) * 3
        out = dec(e2, o2, idx, t_led, cue)
        for key in ("y", "logvar") + (("phi",) if "phi" in base else ()):
            assert torch.allclose(out[key][:, :k], base[key][:, :k], atol=1e-5), (head, key, k)
        assert not torch.allclose(out["y"][:, k + P:], base["y"][:, k + P:], atol=1e-6)  # later outputs do change


def test_phi_monotone_in_the_decoder_for_random_inputs():
    for head in ("proposed", "global_warp"):
        dec = decoder(head).train()
        for seed in range(5):
            eeg, eog = stream(3, 30, seed)
            idx = torch.arange(0, 30 * P).expand(3, -1)
            out = dec(eeg, eog, idx, (idx + 1) / 100.0 - 1.5)
            if head == "proposed":
                assert (out["phi"][:, 1:] >= out["phi"][:, :-1] - 1e-7).all()
            assert out["phi"].min() >= 0 and out["phi"].max() <= 1


def test_neural_tokens_equal_encoder_streaming():
    dec = decoder("direct", ocular=False)
    eeg, _ = stream(1, 12)
    tok = dec.encoder.stream(eeg[0])
    win = eeg.unfold(-1, 200, P).permute(0, 2, 1, 3).reshape(-1, 32, 200)
    assert torch.allclose(dec.encoder(win)[:, -1], tok, atol=1e-5)


def test_frozen_setting_keeps_encoder_fixed_and_in_eval_mode():
    dec = decoder("proposed", freeze=True).train()
    assert not dec.encoder.training and dec.fusion.training
    assert all(not q.requires_grad for q in dec.encoder.parameters())
    eeg, eog = stream(2, 10)
    idx = torch.arange(0, 10 * P).expand(2, -1)
    out = dec(eeg, eog, idx)
    out["y"].sum().backward()
    assert all(q.grad is None for q in dec.encoder.parameters())
    assert dec.gate_penalty().item() == 0.0
    ft = decoder("proposed", freeze=False).train()
    ft(eeg, eog, idx)["y"].sum().backward()
    assert ft.encoder.gate.logits.grad is not None


def test_loss_terms():
    dec = decoder("proposed")
    eeg, eog = stream(2, 10)
    idx = torch.arange(0, 10 * P).expand(2, -1)
    out = dec(eeg, eog, idx)
    y = torch.randn(2, idx.shape[1], 3)
    valid = torch.ones(2, idx.shape[1], dtype=torch.bool)
    valid[1, 100:] = False
    aux = torch.randint(0, len(AUX_CLASSES), (2, 10))
    aux[1, 5:] = -1
    t = decoder_loss(out, y, valid, aux, torch.tensor(0.3), {"aux": 0.5, "delta": 2.0, "amp": 3.0, "gate": 1.0})
    lv = out["logvar"]
    nll = 0.5 * (lv + (y - out["y"]) ** 2 * torch.exp(-lv) + np.log(2 * np.pi))
    v = valid.unsqueeze(-1).float()
    assert t["nll"].item() == pytest.approx(((nll * v).sum() / (v.sum() * 3)).item(), rel=1e-5)
    assert t["loss"].item() == pytest.approx((t["nll"] + 0.5 * t["aux"] + 2 * t["delta"] + 3 * t["amp"] + 0.3).item(),
                                             rel=1e-5)
    no_aux = decoder_loss(out, y, valid, None, torch.tensor(0.0), {"aux": 0.5})
    assert no_aux["aux"].item() == 0.0


def test_aux_mapping_and_patch_targets():
    assert aux_classes(np.arange(7)).tolist() == [0, 1, 2, 3, 4, 4, 4]
    aux = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    out = patch_aux_targets(aux, np.array([1, 3, 7, 0]), np.array([True, True, True, False]))
    assert out.tolist() == [0, 1, 4, -1]


@pytest.mark.skipif(not (CACHE / "P1" / "phase.npy").exists(), reason="needs the main cache")
def test_auxiliary_labels_are_read_for_training_subjects_only():
    fold = Fold(name="loso_P1", train_subjects=("P2", "P4"), val_subjects=("P5",), test_subjects=("P1",))
    lab = training_aux_classes(CACHE, "P2", fold)
    assert set(np.unique(lab)) <= set(range(5)) and lab.ndim == 1
    for s in ("P1", "P5", "P3"):  # test, inner validation, and a subject outside the fold
        with pytest.raises(LabelLeakageError):
            training_aux_classes(CACHE, s, fold)
