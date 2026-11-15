"""Tests for the stage 2 decoder: causal interpolation, phase tracker, warp heads, causality, labels, loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eegtraj.data.decoder_labels import LabelLeakageError, patch_aux_targets, training_aux_classes
from eegtraj.data.splits import Fold
from eegtraj.models.decoder import (HEADS, AUX_CLASSES, StreamDecoder, aux_classes, causal_interp, decoder_loss,
                                    clock_phase)
from eegtraj.models.encoder import CausalEncoder
from eegtraj.models.landmark_template import LandmarkTemplates, TemplateBank, expected_template, time_to_phase

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


def toy_prior() -> dict:
    """Phase prior on t in [-1.5, 12] s: mean = cue phase, SD growing with time."""
    grid = np.arange(-1.5, 12.0 + 0.005, 0.01)
    return {"t0": -1.5, "dt": 0.01, "mu": ((grid - 0.4) / 5.8).astype(np.float32),
            "sd": (0.02 + 0.02 * np.clip(grid, 0, None)).astype(np.float32)}


def decoder(head: str, neural: bool = True, ocular: bool = True, freeze: bool = False,
            prior: bool = False) -> StreamDecoder:
    torch.manual_seed(1)
    bank = TemplateBank.from_templates(templates())
    dec = StreamDecoder(head, small_encoder() if neural else None, ocular, bank, freeze_encoder=freeze, d_fused=32,
                        dropout=0.0, tau0=0.4, s0=5.8, phase_prior=toy_prior() if prior else None)
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


def test_clock_phase_is_monotone_bounded_and_integrates_the_clock():
    g = torch.Generator().manual_seed(0)
    t0, tau0, s0 = -1.5 + 0.01, 0.4, 5.8
    for _ in range(20):
        rate = 0.5 + 1.5 * torch.rand(3, 40, generator=g)
        idx = torch.arange(0, 40 * P + 30).expand(3, -1)
        phi, phi_end, tau = clock_phase(rate, idx, P, 100.0, t0, tau0, s0)
        assert (tau[:, 1:] > tau[:, :-1]).all()  # strictly increasing clock
        assert (phi[:, 1:] >= phi[:, :-1] - 1e-7).all() and phi.min() >= 0 and phi.max() <= 1
        assert torch.allclose(tau[:, P - 1], torch.full((3,), -1.3))  # tau(e_0) = real time at e_0
        assert torch.allclose(tau[:, 2 * P - 1], -1.3 + rate[:, 0] * 0.2, atol=1e-6)  # tau(e_1) = tau(e_0) + k_0 0.2 s
        assert torch.allclose(phi, ((tau - tau0) / s0).clamp(0, 1))


def test_clock_at_real_time_gives_the_cue_phase():
    idx = torch.arange(0, 45 * P).expand(2, -1)
    t_led = -1.5 + (idx + 1) / 100.0
    phi, _, tau = clock_phase(torch.ones(2, 45), idx, P, 100.0, -1.5 + 0.01, 0.4, 5.8)
    assert torch.allclose(tau, t_led, atol=1e-5)
    assert torch.allclose(phi, ((t_led - 0.4) / 5.8).clamp(0, 1), atol=1e-6)


@pytest.mark.parametrize("head", ["proposed", "global_warp"])
def test_warp_heads_start_at_the_cue_phase(head):
    """Zero-initialized outputs: u = 0 (real-time clock) and tau = tau0, s = s0, so both warp heads start at the head
    floor phi = clip((t - t_LED - tau0) / s0, 0, 1) with A = 1, Delta = 0 and uniform mixing."""
    torch.manual_seed(1)
    dec = StreamDecoder(head, small_encoder(), True, TemplateBank.from_templates(templates()), d_fused=32,
                        dropout=0.0, tau0=0.4, s0=5.8).eval()
    eeg, eog = stream(2, 45)
    idx = torch.arange(0, 45 * P).expand(2, -1)
    t_led = -1.5 + (idx + 1) / 100.0
    out = dec(eeg, eog, idx, t_led)
    phi = ((t_led - 0.4) / 5.8).clamp(0, 1)
    assert torch.allclose(out["phi"], phi, atol=1e-5)
    floor = torch.from_numpy(np.mean([templates()(phi.numpy(), c) for c in (165, 330, 660)], axis=0)).float()
    assert torch.allclose(out["y"], floor, atol=1e-3)


def test_expected_template_limits():
    bank = TemplateBank.from_templates(templates())
    phi = torch.linspace(-0.2, 1.2, 50)
    w = torch.softmax(torch.randn(50, 3), -1)
    assert torch.allclose(expected_template(bank, phi, torch.zeros(50), w), bank(phi, w), atol=1e-6)
    wide = expected_template(bank, torch.full((1,), 0.5), torch.full((1,), 0.3), w[:1])
    grid = torch.linspace(-3, 4, 20001)  # brute-force Gaussian average
    dens = torch.exp(-0.5 * ((grid - 0.5) / 0.3) ** 2)
    brute = (dens[:, None] * bank(grid, w[:1].expand(grid.numel(), -1))).sum(0) / dens.sum()
    assert torch.allclose(wide[0], brute, atol=0.03)  # cm; clip kinks slow Gauss-Hermite convergence


@pytest.mark.parametrize("head", ["proposed", "global_warp"])
def test_warp_heads_with_prior_start_at_the_head_floor(head):
    """Zero-initialized outputs: real-time clock and prior SD, A = 1, Delta = 0, uniform mixing."""
    dec = decoder(head, prior=True)
    eeg, eog = stream(2, 45)
    idx = torch.arange(0, 45 * P).expand(2, -1)
    t_led = -1.5 + (idx + 1) / 100.0
    out = dec(eeg, eog, idx, t_led)
    pr = toy_prior()
    mu = torch.from_numpy(np.interp(t_led.numpy(), -1.5 + 0.01 * np.arange(pr["mu"].size), pr["mu"])).float()
    sd = torch.from_numpy(np.interp(t_led.numpy(), -1.5 + 0.01 * np.arange(pr["sd"].size), pr["sd"])).float()
    assert torch.allclose(out["phi_centre"], mu, atol=1e-4) and torch.allclose(out["phi_sigma"], sd, atol=1e-4)
    floor = expected_template(TemplateBank.from_templates(templates()), mu, sd, torch.full((2, idx.shape[1], 3), 1 / 3))
    assert torch.allclose(out["y"], floor, atol=1e-3)
    assert (out["phi_centre"][:, 1:] >= out["phi_centre"][:, :-1] - 1e-6).all()


def test_sigma_gets_gradients_and_stays_bounded():
    dec = decoder("proposed", prior=True).train()
    eeg, eog = stream(2, 20)
    idx = torch.arange(0, 20 * P).expand(2, -1)
    out = dec(eeg, eog, idx, -1.5 + (idx + 1) / 100.0)
    out["y"].sum().backward()
    assert dec.out["sigma"].weight.grad is not None and dec.out["sigma"].weight.grad.abs().sum() > 0
    with torch.no_grad():
        dec.out["sigma"].weight.normal_(std=10.0)
        out = dec(eeg, eog, idx, -1.5 + (idx + 1) / 100.0)
    ratio = out["phi_sigma"] / dec.prior_at(out["clock"])[1]
    assert ratio.min() >= 0.1 - 1e-4 and ratio.max() <= 10.0 + 1e-3


def test_clock_rate_bounded():
    dec = decoder("proposed").train()
    with torch.no_grad():
        dec.out["rate"].weight.normal_(std=5.0)
    eeg, eog = stream(2, 20)
    out = dec(eeg, eog, torch.arange(0, 20 * P).expand(2, -1))
    assert out["rate"].min() >= 0.5 - 1e-6 and out["rate"].max() <= 2.0 + 1e-6


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


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("branches", [(True, True), (True, False), (False, True)])
def test_full_decoder_is_causal(head, branches, prior):
    if prior and head not in ("proposed", "global_warp"):
        pytest.skip("phase prior only affects the warp heads")
    dec = decoder(head, *branches, prior=prior)
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


@pytest.mark.skipif(not (CACHE / "P1" / "trials.csv").exists(), reason="needs the main cache")
def test_phase_prior_uses_training_subjects_only_and_is_monotone():
    from eegtraj.data.windows import load_subject_samples
    from eegtraj.models.landmark_template import fit_landmark_templates, fit_phase_prior

    subs = [load_subject_samples(CACHE, s, "broadband", "car", "all", "position", history=300, pre_lift=True)
            for s in ("P1", "P2")]
    tmpl = fit_landmark_templates(subs, ["P1"])
    a = fit_phase_prior(subs, ["P1"], tmpl.fractions)
    b = fit_phase_prior(subs[:1], ["P1"], tmpl.fractions)
    assert np.array_equal(a.mu, b.mu) and np.array_equal(a.sd, b.sd)  # P2 ignored
    assert (np.diff(a.mu) >= 0).all() and (a.sd > 0).all()
    mu, _ = a(np.array([-1.0, 0.4, 3.0, 6.0]))
    assert mu[0] < 0 < mu[2] < mu[3] <= 1.1
