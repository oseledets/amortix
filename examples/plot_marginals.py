"""Marginals of the learned posterior against the exact ones.

A single distance says how far off we are; the marginals say in what way --
shifted, too wide, too narrow, skewed. On `linear_gaussian` the reference is
exact, so this is a direct picture of the answer rather than a proxy.

Also prints, per parameter, the 1-D Wasserstein distance normalized by the exact
posterior's own sd (so 0.1 means "a tenth of a posterior width off"), and the
width ratio.

    uv run python examples/plot_marginals.py --steps 3000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from amortix import FlowPosterior
from amortix.problems.linear_gaussian import make, exact_posterior

N_SHOW, N_DRAW = 3, 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    prob = make()
    names = prob.prior.names
    post = FlowPosterior(prob).fit(n_train=args.n_train, steps=args.steps,
                                   batch=args.batch, verbose=True)

    gen = torch.Generator().manual_seed(5)
    m_true = prob.prior.sample(N_SHOW, generator=gen)
    tok, y = prob.observe(m_true, generator=gen)
    ours = post.sample_batch(tok, n=N_DRAW, seed=0).numpy()
    exact = [exact_posterior(y[i], prob, n=N_DRAW, seed=i).numpy() for i in range(N_SHOW)]

    print(f"\n{'dataset':>8} {'param':>7} | {'W1 / exact sd':>13} | {'width ratio':>11} | "
          f"{'mean shift / sd':>15}")
    print("-" * 62)
    for i in range(N_SHOW):
        for j, nm in enumerate(names):
            e, o = exact[i][:, j], ours[i][:, j]
            sd = e.std() + 1e-12
            print(f"{i:>8} {nm:>7} | {wasserstein_distance(o, e) / sd:12.3f} | "
                  f"{o.std() / sd:10.3f} | {abs(o.mean() - e.mean()) / sd:14.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(N_SHOW, len(names), figsize=(3.1 * len(names), 2.5 * N_SHOW))
    ax = np.atleast_2d(ax)
    for i in range(N_SHOW):
        for j, nm in enumerate(names):
            a = ax[i, j]
            lo = min(exact[i][:, j].min(), ours[i][:, j].min())
            hi = max(exact[i][:, j].max(), ours[i][:, j].max())
            bins = np.linspace(lo, hi, 60)
            a.hist(exact[i][:, j], bins=bins, density=True, alpha=0.55,
                   color="#4C72B0", label="exact posterior")
            a.hist(ours[i][:, j], bins=bins, density=True, histtype="step",
                   lw=2, color="#C44E52", label="amortix")
            a.axvline(float(m_true[i, j]), color="k", lw=1.2, ls=":")
            if i == 0:
                a.set_title(nm)
            if j == 0:
                a.set_ylabel(f"dataset {i}")
            a.set_yticks([])
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"posterior marginals vs the exact posterior "
                 f"({args.steps} optimizer steps); dotted = true parameter")
    fig.tight_layout()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "results", "marginals_linear_gaussian.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
