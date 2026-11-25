"""Three recurrent "attention" rules, each in a sequential form and a chunkwise-parallel form.

All shapes are ``(b, h, L, d)``. The state ``S`` is ``(d_k, d_v)`` per head, and the output
at step t reads the state *after* the update, ``o_t = S_tᵀ q_t``.

=================  ===============================================  ======================
rule               state update                                     what it can do
=================  ===============================================  ======================
linear attention   S_t = S_{t−1} + k_t v_tᵀ                         add only; stale
                                                                    associations interfere
gated (decay)      S_t = α_t S_{t−1} + k_t v_tᵀ                     forget everything
                                                                    uniformly (RetNet, GLA,
                                                                    Mamba-2 style)
DeltaNet           S_t = S_{t−1} + β_t k_t (v_t − S_{t−1}ᵀ k_t)ᵀ    *replace* the value
                       = (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ   stored under k_t
=================  ===============================================  ======================

DeltaNet is one step of online gradient descent on ‖Sᵀk − v‖² per token (the Widrow–Hoff
delta rule, Schlag et al. 2021). It is not a sum of independent outer products, so the
usual "linear attention is a masked matmul" trick doesn't apply directly. The chunkwise
form (Yang et al. 2024) works with *pseudo-values* u_t = β_t (v_t − S_{t−1}ᵀ k_t), so that
S_t = S_0 + Σ_{i≤t} k_i u_iᵀ inside a chunk. Substituting gives a unit lower-triangular
system

    (I + diag(β) · tril(K Kᵀ, −1)) U = diag(β) (V − K S_0)

which is one triangular solve per chunk (the "WY representation"). Then

    O = Q S_0 + tril(Q Kᵀ) U,      S_end = S_0 + Kᵀ U.

All work inside a chunk is matmuls. Only the C × C solve and the chunk-to-chunk state pass
are sequential, which is what makes DeltaNet trainable at Transformer speed.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- recurrences
def linear_recurrent(q, k, v, log_alpha=None):
    b, h, L, dk = q.shape
    S = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(L):
        if log_alpha is not None:
            S = S * log_alpha[..., t, None, None].exp()
        S = S + k[..., t, :, None] * v[..., t, None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[..., t, :], S))
    return torch.stack(out, 2), S


def delta_recurrent(q, k, v, beta):
    b, h, L, dk = q.shape
    S = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(L):
        kt, vt, bt = k[..., t, :], v[..., t, :], beta[..., t, None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)  # what the state currently returns for k_t
        S = S + kt[..., :, None] * (bt * (vt - pred))[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[..., t, :], S))
    return torch.stack(out, 2), S


# ------------------------------------------------------------------------------ chunked
def _chunks(x, C):
    b, h, L = x.shape[:3]
    if L % C:
        raise ValueError(f"length {L} must be a multiple of the chunk size {C}")
    return x.view(b, h, L // C, C, *x.shape[3:])


def linear_chunked(q, k, v, log_alpha=None, C: int = 16):
    """Intra-chunk: a decay-masked ``(Q Kᵀ ∘ D) V``; inter-chunk: carry S."""
    qc, kc, vc = _chunks(q, C), _chunks(k, C), _chunks(v, C)
    b, h, n = qc.shape[:3]
    if log_alpha is None:
        log_alpha = q.new_zeros(q.shape[:3])
    la = _chunks(log_alpha, C).cumsum(-1)  # (b, h, n, C): log of decay since chunk start
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    # D[t, s] = Π_{s < r ≤ t} α_r for s ≤ t
    D = (la[..., :, None] - la[..., None, :]).masked_fill(~causal, float("-inf")).exp()
    S = q.new_zeros(b, h, q.shape[-1], v.shape[-1])
    out = []
    for i in range(n):
        Q, K, V, a = qc[:, :, i], kc[:, :, i], vc[:, :, i], la[:, :, i]
        inter = (Q * a[..., None].exp()) @ S  # contribution of earlier chunks, decayed
        intra = ((Q @ K.transpose(-1, -2)) * D[:, :, i]) @ V
        out.append(inter + intra)
        decay_to_end = (a[..., -1:] - a).exp()  # (b, h, C)
        S = S * a[..., -1, None, None].exp() + (K * decay_to_end[..., None]).transpose(-1, -2) @ V
    return torch.cat(out, 2), S


def delta_chunked(q, k, v, beta, C: int = 16):
    qc, kc, vc, bc = _chunks(q, C), _chunks(k, C), _chunks(v, C), _chunks(beta, C)
    b, h, n = qc.shape[:3]
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    # A = I + diag(β) tril(K Kᵀ, −1); W = A⁻¹ diag(β) K and Ũ = A⁻¹ diag(β) V, for all chunks
    # at once, since they don't depend on the state.
    A = eye + bc[..., None] * (kc @ kc.transpose(-1, -2)).masked_fill(~strict, 0)
    W = torch.linalg.solve_triangular(A, bc[..., None] * kc, upper=False, unitriangular=True)
    U_tilde = torch.linalg.solve_triangular(A, bc[..., None] * vc, upper=False, unitriangular=True)
    QK = (qc @ kc.transpose(-1, -2)).masked_fill(~incl, 0)
    S = q.new_zeros(b, h, q.shape[-1], v.shape[-1])
    out = []
    for i in range(n):  # the only sequential part: pass the state between chunks
        U = U_tilde[:, :, i] - W[:, :, i] @ S  # pseudo-values given the incoming state
        out.append(qc[:, :, i] @ S + QK[:, :, i] @ U)
        S = S + kc[:, :, i].transpose(-1, -2) @ U
    return torch.cat(out, 2), S
