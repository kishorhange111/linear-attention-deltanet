"""Multi-query associative recall (MQAR, Arora et al. 2023) across the four mixers.

    python -m linattn.train --out results --device cuda

The sequence lists K key→value pairs, then queries keys in random order; the model must
output each queried key's value. Softmax attention can look a pair up. A recurrent state
has to *store* K associations in a d_k × d_v matrix per head. Additive linear attention
superimposes them and they interfere. DeltaNet's update writes each value into the state
by correcting what the state currently returns for that key.

Each (mixer, K) run trains a 2-layer, d = 128 model for ``--steps`` steps and reports
accuracy on fresh sequences. The sweep also records step time for the chunked kernels.
Runs append to ``<out>/runs.jsonl`` and are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .model import LM, Config

KEYS, VALS = (1, 128), (128, 256)


def mqar_batch(bs, K, gen, device):
    keys = torch.stack([torch.randperm(KEYS[1] - KEYS[0], generator=gen)[:K] for _ in range(bs)])
    keys = keys + KEYS[0]
    vals = torch.randint(VALS[0], VALS[1], (bs, K), generator=gen)
    order = torch.stack([torch.randperm(K, generator=gen) for _ in range(bs)])
    qk, qv = keys.gather(1, order), vals.gather(1, order)
    x = torch.cat([torch.stack([keys, vals], 2).flatten(1), torch.stack([qk, qv], 2).flatten(1)], 1)
    y = torch.full_like(x, -100)
    y[:, 2 * K :: 2] = qv
    return x.to(device), y.to(device)


def accuracy(model, K, seed, device, batches=8):
    eg = torch.Generator().manual_seed(10_000 + seed)
    correct = total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = mqar_batch(256, K, eg, device)
            pred = model(x).argmax(-1)
            m = y != -100
            correct += int((pred[m] == y[m]).sum())
            total += int(m.sum())
    return correct / total


def label(r) -> str:
    return r["mixer"] + (f"+conv{r['conv']}" if r.get("conv") else "")


def run(mixer, K, steps, device, bs=64, lr=1e-3, seed=0, d=128, conv=0, curve_every=1000):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cfg = Config(d=d, mixer=mixer, ctx=4 * K, conv=conv)
    model = LM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    t0, curve, eval_sec = time.time(), [], 0.0
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = mqar_batch(bs, K, gen, device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if curve_every and step % curve_every == 0 and step < steps:
            te = time.time()
            model.eval()
            curve.append([step, round(accuracy(model, K, seed, device, batches=2), 4)])
            model.train()
            eval_sec += time.time() - te
    if device == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0 - eval_sec
    model.eval()
    return {
        "mixer": mixer,
        "conv": conv,
        "lr": lr,
        "K": K,
        "seed": seed,
        "steps": steps,
        "accuracy": accuracy(model, K, seed, device),
        "curve": curve,
        "final_loss": loss.item(),
        "ms_per_step": 1000 * sec / steps,
        "params": sum(p.numel() for p in model.parameters()),
    }


def report(rows) -> str:
    Ks = sorted({r["K"] for r in rows})
    mixers = list(dict.fromkeys(label(r) for r in rows))
    out = [
        "| mixer | " + " | ".join(f"K = {k}" for k in Ks) + " | ms / step |",
        "|---|" + "---|" * (len(Ks) + 1),
    ]
    for m in mixers:
        cells = []
        for k in Ks:
            accs = [r["accuracy"] for r in rows if label(r) == m and r["K"] == k]
            cells.append(" / ".join(f"{100 * a:.1f}%" for a in accs) if accs else "—")
        ms = [r["ms_per_step"] for r in rows if label(r) == m]
        out.append(f"| {m} | " + " | ".join(cells) + f" | {sum(ms) / len(ms):.0f} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mixers", default="softmax,deltanet,gated,linear")
    ap.add_argument("--Ks", default="16,32,64")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--conv", type=int, default=0, help="short conv width for recurrent mixers")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    rows = [json.loads(s) for s in open(path)] if os.path.exists(path) else []
    key = lambda r: (r["mixer"], r.get("conv", 0), r.get("lr", 1e-3), r["K"], r["seed"])  # noqa: E731
    done = {key(r) for r in rows}
    for seed in map(int, a.seeds.split(",")):
        for K in map(int, a.Ks.split(",")):
            for m in a.mixers.split(","):
                conv = 0 if m == "softmax" else a.conv
                if (m, conv, a.lr, K, seed) in done:
                    continue
                r = run(m, K, a.steps, a.device, lr=a.lr, seed=seed, conv=conv)
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                rows.append(r)
                print(json.dumps(r), flush=True)
    text = report(rows)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
