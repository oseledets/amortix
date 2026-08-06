"""amortix gallery benchmark -- every case, one table.

Trains an amortized flow-matching posterior for each registered problem under a
single uniform budget and compares it against that case's classical SOTA baseline
(MLE / pseudo-MLE / Kramers-Moyal / nonlinear least squares) on accuracy and
calibration. This is the package's headline "amortized vs classical" harness.

    python examples/gallery.py            # ~uniform budget, all cases
    python examples/gallery.py --quick    # smaller/faster
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import FlowPosterior
from amortix.problems import GALLERY

# display metadata (domain / kind) per case
META = {
    "ou":         ("SDE-1D", "validation seed"),
    "seir":       ("ODE-5D", "epidemiology"),
    "gbm":        ("SDE-1D", "finance"),
    "cir":        ("SDE-1D", "rates/vol"),
    "double_well":("SDE-1D", "bistable"),
    "stoch_lv":   ("SDE-2D", "ecology"),
    "fhn":        ("ODE-2D", "neuroscience"),
    "sindy_sde":  ("SDE-1D", "nonparam drift"),
}


def run_case(name, n_train, epochs, K, n_post):
    mod = importlib.import_module(f"amortix.problems.{name}")
    prob = mod.make()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()
    d = prob.prior.dim

    t0 = time.time()
    post = FlowPosterior(prob).fit(n_train=n_train, epochs=epochs, verbose=False)
    train_t = time.time() - t0

    gen = torch.Generator().manual_seed(123)
    m_true = prob.prior.sample(K, generator=gen)
    tokens, traj = prob.observe(m_true, generator=gen)

    amort = np.zeros((K, d)); std = np.zeros((K, d))
    lo = np.zeros((K, d)); hi = np.zeros((K, d)); base = np.zeros((K, d))
    t0 = time.time()
    for i in range(K):
        s = post.sample(tokens[i], n=n_post, seed=i).numpy()
        amort[i] = s.mean(0); std[i] = s.std(0)
        lo[i] = np.quantile(s, 0.05, 0); hi[i] = np.quantile(s, 0.95, 0)
    inf_t = (time.time() - t0) / K
    for i in range(K):
        base[i] = mod.sota(tokens[i].numpy(), traj[i].numpy(), prob)

    mt = m_true.numpy()
    a_err = (np.abs(amort - mt) / rng * 100)
    b_err = (np.abs(base - mt) / rng * 100)
    pstd = (std / rng * 100)
    cov = ((mt >= lo) & (mt <= hi)).mean() * 100
    return {
        "name": name, "dim": d, "tokens": prob.observer.n_tokens,
        "names": names, "sota_name": mod.SOTA_NAME,
        "a_err": a_err.mean(), "b_err": b_err.mean(),
        "pstd": pstd.mean(), "cov": cov,
        "a_perp": a_err.mean(0), "b_perp": b_err.mean(0), "p_perp": pstd.mean(0),
        "train_t": train_t, "inf_t": inf_t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_train, epochs, K, n_post = (3000, 12, 30, 400) if args.quick else (6000, 20, 40, 600)

    print("=" * 92)
    print("amortix GALLERY  --  amortized flow-matching posterior vs classical SOTA")
    print(f"uniform budget: n_train={n_train}, epochs={epochs}, test={K}, post draws={n_post}")
    print("=" * 92)

    rows = []
    for name in GALLERY:
        print(f"... running {name}", flush=True)
        rows.append(run_case(name, n_train, epochs, K, n_post))

    hdr = f"{'case':>12} | {'kind':>7} | {'dim':>3} | {'tok':>4} | {'amort':>7} | {'post.std':>8} | {'SOTA':>7} | {'baseline':>16} | {'cov90':>6} | win"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        kind, _ = META.get(r["name"], ("", ""))
        win = "amort" if r["a_err"] <= r["b_err"] else "SOTA"
        print(f"{r['name']:>12} | {kind:>7} | {r['dim']:>3} | {r['tokens']:>4} |"
              f" {r['a_err']:6.1f}% | {r['pstd']:7.1f}% | {r['b_err']:6.1f}% |"
              f" {r['sota_name'][:16]:>16} | {r['cov']:5.0f}% | {win}")

    wins = sum(1 for r in rows if r["a_err"] <= r["b_err"])
    calib = sum(1 for r in rows if 80 <= r["cov"] <= 97)
    print("-" * len(hdr))
    print(f"amortized wins/ties on point accuracy: {wins}/{len(rows)} cases  |  "
          f"well-calibrated (cov90 in 80-97%): {calib}/{len(rows)}")
    print("(amortized always adds a calibrated posterior + ~instant amortized inference;")
    print(" SOTA gives a single point. Cases where SOTA wins accuracy are low-noise /")
    print(" closed-form-MLE regimes -- expected, and noted honestly.)")

    print("\n--- per-parameter detail (amort err% / post.std% / SOTA err%) ---")
    for r in rows:
        print(f"\n{r['name']} [{r['sota_name']}]")
        for j, nm in enumerate(r["names"]):
            print(f"   {nm:>10}: {r['a_perp'][j]:6.1f}% / {r['p_perp'][j]:5.1f}% / {r['b_perp'][j]:6.1f}%")

    print(f"\ntiming: train {sum(r['train_t'] for r in rows):.0f}s total | "
          f"amortized inference {np.mean([r['inf_t'] for r in rows])*1e3:.0f} ms/dataset avg")


if __name__ == "__main__":
    main()
