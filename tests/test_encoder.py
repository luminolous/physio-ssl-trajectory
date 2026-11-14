"""Tests for the neural-branch encoder (gate, tokenizer, causal transformer, streaming)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from eegtraj.data.target_view import prior_vector
from eegtraj.models.encoder import CausalEncoder, ChannelGate, n_parameters

CFG = yaml.safe_load(Path("configs/pretrain/encoder.yaml").read_text(encoding="utf-8"))
NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9",
         "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"]


def prior() -> torch.Tensor:
    cfg = yaml.safe_load(Path(CFG["prior_config"]).read_text(encoding="utf-8"))
    return torch.from_numpy(prior_vector(cfg, NAMES))


def encoder(dropout: float = 0.0) -> CausalEncoder:
    torch.manual_seed(0)
    kw = {k: v for k, v in CFG.items() if k != "prior_config"}
    return CausalEncoder(prior(), **{**kw, "dropout": dropout}).eval()


def test_shapes_and_parameter_count():
    enc = encoder()
    out = enc(torch.randn(4, 32, 200))
    assert out.shape == (4, 10, 128)
    n = n_parameters(enc)
    assert 400_000 < n < 650_000, n  # design: about 0.5 M


def test_gate_starts_at_the_prior_and_penalty_pulls_back():
    g = ChannelGate(prior())
    assert torch.allclose(g.weights, prior(), atol=1e-6)
    assert g.prior_penalty().item() == pytest.approx(0.0, abs=1e-10)
    with torch.no_grad():
        g.logits += 1.0
    assert g.prior_penalty().item() > 0
    x = torch.ones(1, 32, 5)
    assert torch.allclose(g(x)[0, :, 0], g.weights)
    with pytest.raises(ValueError):
        ChannelGate(torch.ones(32))


def test_causality_future_samples_do_not_change_earlier_tokens():
    enc = encoder()
    x = torch.randn(2, 32, 200)
    base = enc(x)
    for cut in (20, 87, 140, 199):
        y = x.clone()
        y[..., cut:] = torch.randn_like(y[..., cut:]) * 5
        out = enc(y)
        done = cut // 20  # tokens whose patch ends before ``cut`` (patch k ends at sample 20k + 19)
        assert torch.allclose(out[:, :done], base[:, :done], atol=1e-5), cut
        assert not torch.allclose(out[:, done:], base[:, done:], atol=1e-3)


def test_causality_holds_in_training_mode_without_dropout():
    enc = encoder(dropout=0.0).train()
    x = torch.randn(3, 32, 200)
    y = x.clone()
    y[..., 120:] = 0.0
    assert torch.allclose(enc(x)[:, :6], enc(y)[:, :6], atol=1e-5)


def test_dropping_tokens_keeps_positions_and_causality():
    enc = encoder()
    x = torch.randn(2, 32, 200)
    keep = torch.tensor([0, 1, 2, 7, 8, 9])
    full_prefix = enc(x)[:, :3]
    kept = enc(x, keep=keep)
    assert kept.shape == (2, 6, 128)
    assert torch.allclose(kept[:, :3], full_prefix, atol=1e-5)  # tokens before the block see the same past
    with pytest.raises(ValueError):
        enc(x, keep=torch.tensor([3, 1]))


def test_target_branch_skips_the_gate():
    enc = encoder()
    x = torch.randn(2, 32, 200)
    with torch.no_grad():
        enc.gate.logits.fill_(-20.0)  # gate ~ 0: the gated input is ~ zero
    assert not torch.allclose(enc(x, use_gate=False), enc(x), atol=1e-3)
    assert torch.allclose(enc(x), enc(torch.zeros_like(x)), atol=1e-4)


def test_streaming_equals_per_window_last_token():
    enc = encoder()
    eeg = torch.randn(32, 200 + 20 * 15)
    streamed = enc.stream(eeg, chunk=4)
    assert streamed.shape == (16, 128)
    for i in (0, 5, 15):
        window = eeg[:, i * 20:i * 20 + 200].unsqueeze(0)
        assert torch.allclose(streamed[i], enc(window)[0, -1], atol=1e-5)
    with pytest.raises(ValueError):
        enc.stream(torch.randn(32, 210))
