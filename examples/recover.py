"""Train an amortized posterior for one gallery case and benchmark it vs the
classical baseline of that case.

    uv run python examples/recover.py ou            # any name from amortix.problems.GALLERY
    uv run python examples/recover.py sindy_sde --n_train 12000 --epochs 40
    uv run python examples/recover.py ou --plot     # posterior histograms vs truth/baseline

Reports, per parameter: mean absolute error (% of prior range), posterior std in
the same units, the classical baseline's error, and 90% interval coverage.
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.problems import GALLERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=GALLERY)
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_post", type=int, default=600)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    mod = importlib.import_module(f"amortix.problems.{args.case}")
    prob = mod.make()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()

    print(f"=== {args.case} :: amortized posterior vs {mod.SOTA_NAME} ===")
    print(f"params {names} | tokens {prob.observer.n_tokens} | "
          f"budget {args.n_train}/{args.epochs}")
    post = FlowPosterior(prob).fit(n_train=args.n_train, epochs=args.epochs)

    gen = torch.Generator().manual_seed(123)
    m_true = prob.prior.sample(args.n_test, generator=gen)
    tokens, traj = prob.observe(m_true, generator=gen)

    t0 = time.time()
    draws = post.sample_batch(tokens, n=args.n_post, seed=0).numpy()   # [K, n_post, d]
    amort_t = (time.time() - t0) / args.n_test

    t0 = time.time()
    base = np.stack([mod.sota(tokens[i].numpy(), traj[i].numpy(), prob)
                     for i in range(args.n_test)])
    base_t = (time.time() - t0) / args.n_test

    mt = m_true.numpy()
    a_err = (np.abs(draws.mean(1) - mt) / rng * 100).mean(0)
    b_err = (np.abs(base - mt) / rng * 100).mean(0)
    pstd = (draws.std(1) / rng * 100).mean(0)
    lo, hi = np.quantile(draws, 0.05, axis=1), np.quantile(draws, 0.95, axis=1)
    cov = ((mt >= lo) & (mt <= hi)).mean(0) * 100

    print(f"\n{'param':>10} | {'amort':>7} | {'post.std':>8} | {mod.SOTA_NAME[:14]:>14} | {'cov90':>6}")
    for j, nm in enumerate(names):
        print(f"{nm:>10} | {a_err[j]:6.2f}% | {pstd[j]:7.2f}% | {b_err[j]:13.2f}% | {cov[j]:5.0f}%")
    print(f"{'ALL':>10} | {a_err.mean():6.2f}% | {pstd.mean():7.2f}% | "
          f"{b_err.mean():13.2f}% | {cov.mean():5.0f}%")
    print(f"\ninference per dataset: amortized {amort_t*1e3:.0f} ms | "
          f"{mod.SOTA_NAME} {base_t*1e3:.0f} ms")
    print("(errors are % of prior range; cov90 should sit near 90%. Coverage alone is a\n"
          " weak calibration test -- use examples/calib_gallery.py for strict SBC.)")

    if args.plot:
        _plot(args.case, names, draws[0], mt[0], base[0])


def _plot(case, names, draws, truth, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = len(names)
    fig, ax = plt.subplots(1, d, figsize=(3.2 * d, 3.2))
    ax = np.atleast_1d(ax)
    for j, nm in enumerate(names):
        ax[j].hist(draws[:, j], bins=40, density=True, alpha=0.6, color="#4C72B0",
                   label="amortized posterior")
        ax[j].axvline(truth[j], color="k", lw=2, label="truth")
        ax[j].axvline(baseline[j], color="#C44E52", ls="--", lw=2, label="baseline")
        ax[j].set_title(nm)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{case}: posterior vs truth vs classical baseline")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{case}_posterior.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
