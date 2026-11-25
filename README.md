# linear-attention-deltanet

[![CI](https://github.com/kishorhange111/linear-attention-deltanet/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/linear-attention-deltanet/actions/workflows/ci.yml)

Three recurrent replacements for softmax attention, written from scratch:
- **linear attention**;
- **gated (decaying) linear attention**;
- **DeltaNet**.

Each comes in a sequential form and a chunkwise-parallel training form, **proven equal to
each other** in float64. They are compared with softmax attention on multi-query
associative recall.

## The three update rules ([`ops.py`](src/linattn/ops.py))

| rule | state update | consequence |
|---|---|---|
| linear attention | S ← S + k vᵀ | associations superimpose and interfere; nothing is forgotten |
| gated | S ← α S + k vᵀ | uniform forgetting (RetNet / GLA / Mamba-2 style) |
| **DeltaNet** | S ← S + β k (v − Sᵀk)ᵀ | *replaces* the value stored under k: one step of online regression |

**Chunkwise linear attention.** Within a chunk, the output is a decay-masked
`(Q Kᵀ ∘ D) V`; across chunks, the state is carried forward.

**Chunkwise DeltaNet.** This is the interesting one (Yang et al. 2024). The update is not a
sum of independent outer products. But with *pseudo-values* `u_t = β_t (v_t − S_{t−1}ᵀ k_t)`,
the state inside a chunk is again `S_0 + Σ k_i u_iᵀ`. Substituting gives one unit
lower-triangular system per chunk:

```
(I + diag(β)·tril(K Kᵀ, −1)) U = diag(β)(V − K S₀)      ← one triangular solve (WY form)
O = Q S₀ + tril(Q Kᵀ) U        S_end = S₀ + Kᵀ U
```

The solve is computed for all chunks at once because it doesn't depend on the state. The
only sequential work left is passing the state from chunk to chunk.

## Tests

There are 17, on CPU, in float64. They check:
- chunked equals recurrent for linear attention, gated linear attention and **DeltaNet's
  WY form**, at chunk sizes 1, 4, 16 and 64, for outputs and final states;
- linear attention equals the masked matmul `tril(Q Kᵀ) V`;
- **writing (k, v₁) then (k, v₂): DeltaNet returns v₂, while linear attention returns
  v₁ + v₂**;
- DeltaNet stores orthogonal associations exactly;
- with normalised keys and β ∈ (0, 1), DeltaNet's state stays bounded over 2,048 steps,
  while linear attention's grows;
- all four mixers are causal (including when padded to the chunk size) and train;
- the experiment script runs and resumes.

## Experiment ([`train.py`](src/linattn/train.py))

**MQAR** (Zoology): the model sees K key→value pairs, then queries in random order. The
models are 2-layer, d = 128, 2 heads, with the same MLP, optimiser, 3,000 steps and
batch 64. The sweep covers K ∈ {16, 32, 64} × 2 seeds × {softmax, linear, gated, DeltaNet},
and reports query accuracy and ms per step.

```
pytest -q                                          # 17 tests, CPU
python -m linattn.train --out results --device cuda
```

**Results are pending** from the Colab run.
