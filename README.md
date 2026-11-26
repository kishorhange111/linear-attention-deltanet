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
models are 2-layer, d = 128, 2 heads, with the same MLP and optimiser, trained for 3,000
steps at batch 64. Values are drawn from 128 tokens, so chance is 0.8%. `--conv 4` adds a
width-4 causal depthwise convolution (then SiLU) on q, k and v, as DeltaNet, GLA and Mamba
do; `--softmax-conv` does the same for softmax.

```
pytest -q                                                   # 21 tests, CPU
python -m linattn.train --out results --device cuda --conv 4 --softmax-conv 4
```

## Results (Tesla T4, one seed, lr 1e-3)

Query accuracy on fresh sequences. Raw rows, including the accuracy curve every 1,000
steps, are in [`results/runs.jsonl`](results/runs.jsonl).

| mixer | short conv | K = 16 | K = 32 | K = 64 | ms / step (K = 16 / 32 / 64) |
|---|---|---|---|---|---|
| softmax | no | 10.2% | 6.7% | 9.6% | 6.7 / 12.1 / 25.4 |
| DeltaNet | no | 15.4% | – | – | 22.9 |
| softmax | **yes** | **100%** | **100%** | **100%** | 8.9 / 15.9 / 40.2 |
| DeltaNet | **yes** | **100%** | **99.9%** | **99.8%** | 24.1 / 38.1 / 78.6 |
| gated linear | yes | 99.8% | 99.5% | 99.0% | 25.0 / 43.8 / 77.9 |
| linear | yes | 89.8% | 82.9% | 56.8% | 20.2 / 34.1 / 66.2 |

Two checks on the failing softmax baseline, at K = 16: lr 3e-3 reaches only 23.5%, and
10,000 steps instead of 3,000 reaches 10.3%.

- **The short convolution decides whether anything learns.** Without it, softmax (with
  learned positions) and DeltaNet both stall near 10–15%, and softmax is still at 10% after
  10,000 steps. With it, every mixer except plain linear attention solves the task. In MQAR
  a value sits one position after its key. The convolution hands each position its
  predecessor, so binding a key to its value no longer has to be learned through attention.
  Without it, the 2-layer models here never found that mechanism within this budget.
- **With the convolution, the update rule is what separates the recurrent mixers.**
  - Plain linear attention adds every association into one state, `S ← S + k vᵀ`. Its
    accuracy falls from 90% to 57% as K grows from 16 to 64, which is the interference
    the delta rule is meant to fix.
  - DeltaNet's correcting update holds 99.8% at K = 64. Gated linear attention, which can
    forget, holds 99.0%.
- **Softmax is still the fastest here.** At these lengths (64–256 tokens), fused quadratic
  attention is 2–3× faster per step than the chunked recurrent kernels, which are plain
  PyTorch and not fused. The recurrent mixers' advantage, constant-size state and linear
  cost, matters at lengths this sweep does not reach.
- **Learning can be abrupt.** Gated attention at K = 64 was at 1.1% after 1,000 steps and
  98.5% after 2,000.

One seed per cell; the planned second seed was not run.
