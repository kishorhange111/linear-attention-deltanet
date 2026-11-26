import json

import pytest
import torch
import torch.nn.functional as F

from linattn.model import LM, Config
from linattn.ops import delta_chunked, delta_recurrent, linear_chunked, linear_recurrent


def _qkv(L=64, dk=8, dv=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k = (torch.randn(2, 3, L, dk, generator=g, dtype=torch.float64) for _ in range(2))
    v = torch.randn(2, 3, L, dv, generator=g, dtype=torch.float64)
    return q, k, v, g


@pytest.mark.parametrize("C", [1, 4, 16, 64])
def test_linear_chunked_equals_recurrence(C):
    q, k, v, g = _qkv()
    o_r, s_r = linear_recurrent(q, k, v)
    o_c, s_c = linear_chunked(q, k, v, C=C)
    torch.testing.assert_close(o_c, o_r)
    torch.testing.assert_close(s_c, s_r)
    la = F.logsigmoid(torch.randn(2, 3, 64, generator=g, dtype=torch.float64))
    o_r, s_r = linear_recurrent(q, k, v, la)
    o_c, s_c = linear_chunked(q, k, v, la, C=C)
    torch.testing.assert_close(o_c, o_r)
    torch.testing.assert_close(s_c, s_r)


def test_linear_attention_is_masked_matmul():
    q, k, v, _ = _qkv()
    ref = (q @ k.transpose(-1, -2)).tril() @ v
    torch.testing.assert_close(linear_chunked(q, k, v, C=16)[0], ref)


@pytest.mark.parametrize("C", [1, 4, 16, 64])
def test_delta_chunked_wy_form_equals_recurrence(C):
    q, k, v, g = _qkv()
    k = F.normalize(k, dim=-1)
    beta = torch.rand(2, 3, 64, generator=g, dtype=torch.float64)
    o_r, s_r = delta_recurrent(q, k, v, beta)
    o_c, s_c = delta_chunked(q, k, v, beta, C=C)
    torch.testing.assert_close(o_c, o_r)
    torch.testing.assert_close(s_c, s_r)


def test_delta_rule_overwrites_where_linear_attention_accumulates():
    """Write (k, v₁) then (k, v₂): DeltaNet with β = 1 returns v₂, linear attention v₁ + v₂."""
    dk = 4
    k = torch.zeros(1, 1, 2, dk, dtype=torch.float64)
    k[..., 0] = 1.0  # the same unit key twice
    v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    beta = torch.ones(1, 1, 2, dtype=torch.float64)
    _, S_delta = delta_recurrent(k, k, v, beta)
    _, S_lin = linear_recurrent(k, k, v)
    read = k[0, 0, 0]
    torch.testing.assert_close(read @ S_delta[0, 0], torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(read @ S_lin[0, 0], torch.tensor([1.0, 1.0], dtype=torch.float64))


def test_delta_rule_stores_orthogonal_associations_exactly():
    dk = 8
    keys = torch.linalg.qr(torch.randn(dk, dk, dtype=torch.float64))[0][:, :5].T  # 5 orthonormal
    vals = torch.randn(5, 3, dtype=torch.float64)
    _, S = delta_recurrent(
        keys[None, None],
        keys[None, None],
        vals[None, None],
        torch.ones(1, 1, 5, dtype=torch.float64),
    )
    torch.testing.assert_close(keys @ S[0, 0], vals)


def test_delta_state_stays_bounded_with_normalised_keys():
    """With ‖k‖ = 1 and β ∈ (0, 1), I − β k kᵀ is a contraction: no blow-up over long L."""
    g = torch.Generator().manual_seed(0)
    L = 2048
    k = F.normalize(torch.randn(1, 1, L, 8, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(1, 1, L, 4, generator=g, dtype=torch.float64)
    beta = torch.rand(1, 1, L, generator=g, dtype=torch.float64)
    _, S = delta_chunked(k, k, v, beta, C=64)
    _, S_lin = linear_chunked(k, k, v, C=64)
    assert S.norm() < 10 < S_lin.norm()


@pytest.mark.parametrize("conv", [0, 4])
@pytest.mark.parametrize("mixer", ["softmax", "linear", "gated", "deltanet"])
def test_models_are_causal_and_train(mixer, conv):
    torch.manual_seed(0)
    m = LM(Config(d=32, heads=2, layers=1, ctx=40, mixer=mixer, chunk=8, conv=conv))
    x = torch.randint(0, 256, (2, 40))
    y = m(x)
    x2 = x.clone()
    x2[:, 30:] = torch.randint(0, 256, (2, 10))
    torch.testing.assert_close(m(x2)[:, :30], y[:, :30], atol=1e-5, rtol=1e-5)
    assert m(x[:, :37]).shape == (2, 37, 256)  # padded to the chunk size internally
    F.cross_entropy(y.flatten(0, 1), x.flatten()).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_train_script_runs_and_resumes(tmp_path):
    from linattn.train import main

    args = [
        "--out",
        str(tmp_path),
        "--device",
        "cpu",
        "--mixers",
        "deltanet,linear",
        "--Ks",
        "4",
        "--seeds",
        "0",
        "--steps",
        "3",
        "--conv",
        "4",
    ]
    main(args)
    main(args)
    rows = [json.loads(s) for s in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [r["mixer"] for r in rows] == ["deltanet", "linear"]
    assert all(r["conv"] == 4 for r in rows)
    assert "deltanet+conv4" in (tmp_path / "summary.md").read_text()
