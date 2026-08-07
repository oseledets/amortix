"""Corner plot of the learned posterior against the exact one.

Diagonal: marginals. Lower triangle: pairwise joints. Filled/blue is the exact
posterior, red outlines are ours, the dotted line is the true parameter.

The pairwise panels are the point. A coordinate-wise velocity field can reproduce
every marginal perfectly and still get every dependence wrong -- that was the
state of this code before the parameter self-attention fix, when the flow left the
copula exactly unchanged. Marginals alone cannot show that; the off-diagonal
panels can.

    uv run python examples/plot_corner.py --steps 8000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.problems.linear_gaussian import make, exact_posterior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--dataset", type=int, default=0)
    ap.add_argument("--n_draw", type=int, default=6000)
    args = ap.parse_args()

    prob = make()
    names = prob.prior.names
    print(f"training ({args.n_train} sims, {args.steps} steps) ...")
    post = FlowPosterior(prob).fit(n_train=args.n_train, steps=args.steps, verbose=True)

    gen = torch.Generator().manual_seed(5)
    m_true = prob.prior.sample(3, generator=gen)
    tok, y = prob.observe(m_true, generator=gen)
    i = args.dataset
    ours = post.sample_batch(tok[i:i + 1], n=args.n_draw, seed=0).numpy()[0]
    exact = exact_posterior(y[i], prob, n=args.n_draw, seed=99).numpy()
    truth = m_true[i].numpy()

    # numbers to accompany the picture
    print(f"\n{'pair':>9} | {'corr exact':>10} | {'corr ours':>9}")
    d = len(names)
    for a in range(d):
        for b in range(a + 1, d):
            ce = np.corrcoef(exact[:, a], exact[:, b])[0, 1]
            co = np.corrcoef(ours[:, a], ours[:, b])[0, 1]
            print(f"{names[a]}-{names[b]:>4} | {ce:10.3f} | {co:9.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(d, d, figsize=(2.5 * d, 2.5 * d))
    for a in range(d):
        for b in range(d):
            axx = ax[a, b]
            if b > a:
                axx.axis("off")
                continue
            if a == b:
                lo = min(exact[:, a].min(), ours[:, a].min())
                hi = max(exact[:, a].max(), ours[:, a].max())
                bins = np.linspace(lo, hi, 50)
                axx.hist(exact[:, a], bins=bins, density=True, alpha=0.55,
                         color="#4C72B0", label="exact")
                axx.hist(ours[:, a], bins=bins, density=True, histtype="step",
                         lw=2, color="#C44E52", label="amortix")
                axx.axvline(truth[a], color="k", lw=1.2, ls=":")
                axx.set_yticks([])
                if a == 0:
                    axx.legend(fontsize=7)
            else:
                for pts, colors, filled in ((exact, "Blues", True), (ours, "Reds", False)):
                    H, xe, ye = np.histogram2d(pts[:, b], pts[:, a], bins=40)
                    H = H.T / H.max()
                    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
                    lv = [0.15, 0.4, 0.75]
                    if filled:
                        axx.contourf(xc, yc, H, levels=lv + [1.0], cmap=colors, alpha=0.6)
                    else:
                        axx.contour(xc, yc, H, levels=lv, cmap=colors, linewidths=1.4)
                axx.plot(truth[b], truth[a], "k+", ms=9, mew=1.6)
            if a == d - 1:
                axx.set_xlabel(names[b])
            if b == 0 and a > 0:
                axx.set_ylabel(names[a])
    fig.suptitle("posterior vs the exact posterior — blue filled = exact, red = amortix, "
                 "+ / dotted = truth", y=0.995)
    fig.tight_layout()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "results", "corner_linear_gaussian.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
