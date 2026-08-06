"""Does conditional flow matching reproduce a KNOWN posterior?

The linear-Gaussian testbed has an analytic posterior (a Gaussian restricted to
the prior box, with strong correlations by construction). So any mismatch here is
the inference machinery's fault -- not the simulator, the summary statistics or
the data budget. This is the instrument for debugging CFM itself.

Measured per held-out dataset, against exact posterior samples:
  mean err   -- |E_flow[m] - E_true[m]| as % of prior range
  std ratio  -- flow std / true std  (1.0 = right spread; <1 over-confident)
  corr err   -- mean |corr_flow - corr_true| over off-diagonal entries
  W1/std     -- 1-D Wasserstein distance, normalized by the true posterior std

    uv run python examples/vs_exact.py --n_train 20000 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from amortix import FlowPosterior
from amortix.problems.linear_gaussian import make, exact_posterior


def corr(x):
    c = np.corrcoef(x, rowvar=False)
    return c


def evaluate(post, prob, n_test, n_post, seed=11):
    gen = torch.Generator().manual_seed(seed)
    m_true = prob.prior.sample(n_test, generator=gen)
    tokens, y = prob.observe(m_true, generator=gen)
    draws = post.sample_batch(tokens, n=n_post, seed=0).numpy()
    rng = (prob.prior.high - prob.prior.low).numpy()
    d = prob.prior.dim

    mean_err, std_ratio, corr_err, w1 = [], [], [], []
    off = ~np.eye(d, dtype=bool)
    for i in range(n_test):
        ex = exact_posterior(y[i], prob, n=n_post, seed=i).numpy()
        fl = draws[i]
        mean_err.append(np.abs(fl.mean(0) - ex.mean(0)) / rng * 100)
        std_ratio.append(fl.std(0) / (ex.std(0) + 1e-9))
        corr_err.append(np.abs(corr(fl)[off] - corr(ex)[off]).mean())
        w1.append([wasserstein_distance(fl[:, j], ex[:, j]) / (ex[:, j].std() + 1e-9)
                   for j in range(d)])
    return (np.mean(mean_err, 0), np.mean(std_ratio, 0),
            float(np.mean(corr_err)), np.mean(w1, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--n_post", type=int, default=800)
    ap.add_argument("--bases", nargs="+", default=["standard", "data"])
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    prob = make()
    print(f"linear-Gaussian testbed: {prob.prior.dim} params, {prob.observer.n_tokens} obs, "
          f"exact posterior known")
    print(f"budget {args.n_train}/{args.epochs}, {args.n_test} datasets x {args.n_post} draws\n")
    print(f"{'base':>10} | {'mean err':>9} | {'std ratio':>10} | {'corr err':>9} | {'W1/std':>7}")
    print("-" * 60)

    payload = []
    for base in args.bases:
        post = FlowPosterior(prob, base=base)
        post.fit(n_train=args.n_train, epochs=args.epochs, verbose=False)
        me, sr, ce, w1 = evaluate(post, prob, args.n_test, args.n_post)
        print(f"{base:>10} | {me.mean():8.2f}% | {sr.mean():10.3f} | {ce:9.3f} | {w1.mean():7.3f}")
        payload.append(dict(base=base, mean_err=me.tolist(), std_ratio=sr.tolist(),
                            corr_err=ce, w1=w1.tolist()))
    print("-" * 60)
    print("targets: mean err -> 0, std ratio -> 1.000, corr err -> 0, W1/std -> 0")
    print("(std ratio < 1 = over-confident posterior; corr err shows whether the flow")
    print(" reproduces the true parameter correlations at all)")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    with open(os.path.join(repo, "results", f"VS_EXACT{suffix}.json"), "w") as f:
        json.dump(dict(n_train=args.n_train, epochs=args.epochs, results=payload), f, indent=2)


if __name__ == "__main__":
    main()
