"""Toy-data tests for the pretraining objectives (LeJEPA masked-predictor variant, classic JEPA, MAE)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eegtraj.models.encoder import CausalEncoder
from eegtraj.models.pretraining import (SIGReg, Pretrainer, effective_rank, ema_momentum, embedding_std,
                                        linear_probe_r, patches, sample_block_mask)


def prior() -> torch.Tensor:
    p = torch.full((32,), 0.6)
    p[:2] = 0.2
    p[7:9] = 0.9
    return p


def encoder() -> CausalEncoder:
    torch.manual_seed(0)
    return CausalEncoder(prior(), d_model=32, n_layers=2, n_heads=2, mlp_width=64, dropout=0.0)


def pretrainer(objective: str, guided: bool = True, **kw) -> Pretrainer:
    pre = Pretrainer(encoder(), objective, guided, prior(), predictor={"d_model": 32, "mlp_width": 64},
                     sigreg={"n_slices": 64}, seed=0, **kw)
    return pre


def toy_batch(b: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured windows: a few shared oscillations mixed into the channels plus noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(200) / 100.0
    freqs = torch.tensor([2.0, 5.0, 11.0])
    phase = torch.rand(b, 3, 1, generator=g) * 6.28
    src = torch.sin(2 * torch.pi * freqs[None, :, None] * t + phase)  # (B, 3, 200)
    mix = torch.randn(32, 3, generator=torch.Generator().manual_seed(99))
    raw = torch.einsum("cs,bst->bct", mix, src) + 0.1 * torch.randn(b, 32, 200, generator=g)
    cleaned = raw - 0.05 * raw.mean(1, keepdim=True)
    return raw, cleaned


def test_sigreg_small_for_gaussian_large_for_collapse():
    sig = SIGReg(n_slices=256)
    z = torch.randn(2048, 16, generator=torch.Generator().manual_seed(0))
    gauss = sig(z, step=0).item()
    collapsed = sig(torch.zeros_like(z) + 0.01 * z[:, :1], step=0).item()
    assert gauss < 3.0  # N x statistic: O(1) finite-sample value for a true N(0, I)
    assert collapsed > 100 and sig(3 * z, step=0).item() > 100  # collapse and wrong scale are penalized
    assert sig(z, step=3).item() == pytest.approx(sig(z, step=3).item())  # same step -> same slices
    assert not torch.equal(sig.directions(16, 1, "cpu"), sig.directions(16, 2, "cpu"))


def test_block_mask_is_contiguous_and_in_range():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        keep, masked = sample_block_mask(8, 10, (0.3, 0.5), g)
        assert 3 <= masked.shape[1] <= 5 and keep.shape[1] + masked.shape[1] == 10
        assert ((masked[:, 1:] - masked[:, :-1]) == 1).all()  # one contiguous block
        assert torch.equal(torch.sort(torch.cat([keep, masked], 1), 1).values, torch.arange(10).expand(8, -1))


def test_per_sample_keep_matches_per_window_calls():
    enc = encoder().eval()
    x = torch.randn(2, 32, 200)
    keep = torch.tensor([[0, 1, 2, 7, 8, 9], [0, 4, 5, 6, 7, 8]])
    out = enc(x, keep=keep)
    for i in range(2):
        assert torch.allclose(out[i], enc(x[i:i + 1], keep=keep[i])[0], atol=1e-5)


def test_patches_layout():
    x = torch.arange(2 * 3 * 40, dtype=torch.float32).reshape(2, 3, 40)
    p = patches(x, 20)
    assert p.shape == (2, 2, 60)
    assert torch.equal(p[1, 1].reshape(3, 20), x[1, :, 20:40])


@pytest.mark.parametrize("objective", ["lejepa", "jepa", "mae"])
def test_loss_decreases_on_toy_data(objective):
    torch.manual_seed(0)
    pre = pretrainer(objective, total_steps=60)
    opt = torch.optim.AdamW([p for p in pre.parameters() if p.requires_grad], lr=1e-3)
    raw, cleaned = toy_batch(32)
    losses = []
    for step in range(60):
        out = pre(raw, cleaned, step)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        pre.after_step(step)
        losses.append(out["loss"].item())
    assert np.isfinite(losses).all()
    assert np.mean(losses[-10:]) < 0.8 * np.mean(losses[:5]), (objective, losses[:5], losses[-10:])


def test_lejepa_embeddings_do_not_collapse_on_random_data():
    torch.manual_seed(0)
    pre = pretrainer("lejepa", lam=0.5)
    opt = torch.optim.AdamW(pre.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    for step in range(80):
        raw = torch.randn(32, 32, 200, generator=g)
        out = pre(raw, raw, step)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    with torch.no_grad():
        z = pre.encoder(torch.randn(256, 32, 200, generator=g)).reshape(-1, 32).numpy()
    assert embedding_std(z).min() > 0.05
    assert effective_rank(z) > 8  # of 32 dimensions


def test_unguided_target_branch_receives_the_raw_window(monkeypatch):
    seen = []
    for guided in (False, True):
        pre = pretrainer("lejepa", guided=guided)
        raw, cleaned = toy_batch(4)
        enc_forward = pre.encoder.forward

        def spy(x, keep=None, use_gate=True, _f=enc_forward):
            if not use_gate:
                seen.append(x.detach().clone())
            return _f(x, keep=keep, use_gate=use_gate)

        monkeypatch.setattr(pre.encoder, "forward", spy)
        pre(raw, cleaned, 0)
        expected = raw if not guided else cleaned * prior()[:, None]
        assert torch.equal(seen[-1], expected), guided
    with pytest.raises(ValueError):
        pretrainer("lejepa", guided=True)(raw, None, 0)


def test_jepa_target_is_ema_without_gradient_and_gated_context():
    pre = pretrainer("jepa", total_steps=10)
    assert all(not p.requires_grad for p in pre.target_encoder.parameters())
    before = [p.clone() for p in pre.target_encoder.parameters()]
    opt = torch.optim.SGD([p for p in pre.parameters() if p.requires_grad], lr=0.1)
    raw, cleaned = toy_batch(8)
    out = pre(raw, cleaned, 0)
    opt.zero_grad()
    out["loss"].backward()
    opt.step()
    m = pre.after_step(0)
    assert m == pytest.approx(0.996)
    online = list(pre.encoder.parameters())
    for b, t, o in zip(before, pre.target_encoder.parameters(), online):
        assert torch.allclose(t, 0.996 * b + 0.004 * o.detach(), atol=1e-6)
    assert ema_momentum(10, 10) == 1.0 and ema_momentum(5, 10) == pytest.approx(0.998)
    assert pre.encoder.gate.logits.grad is not None  # the gate is trained through the context view


def test_gate_penalty_enters_the_loss():
    pre = pretrainer("mae", lambda_gate=10.0)
    with torch.no_grad():
        pre.encoder.gate.logits += 1.0
    raw, cleaned = toy_batch(4)
    out = pre(raw, cleaned, 0)
    assert out["gate"].item() > 0
    assert out["loss"].item() == pytest.approx(out["pred"].item() + 10.0 * out["gate"].item(), rel=1e-5)


def test_health_metrics():
    rng = np.random.default_rng(0)
    full = rng.standard_normal((500, 16))
    assert effective_rank(full) > 14
    assert effective_rank(np.outer(rng.standard_normal(500), np.ones(16))) < 1.5
    assert embedding_std(np.ones((10, 4))).max() == 0.0
    x = rng.standard_normal((600, 8))
    y = x[:, :3] * [1.0, -2.0, 0.5] + 0.1 * rng.standard_normal((600, 3))
    trial = np.repeat(np.arange(30), 20)
    r = linear_probe_r(x[:400], y[:400], x[400:], y[400:], trial[400:])
    assert r > 0.95
    assert abs(linear_probe_r(x[:400], rng.standard_normal((400, 3)), x[400:], y[400:], trial[400:])) < 0.3
