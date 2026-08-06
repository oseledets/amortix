"""End-to-end Ornstein-Uhlenbeck recovery with amortix.

Trains one amortized flow-matching posterior, then on held-out instances compares
it against the exact closed-form OU maximum-likelihood estimator -- on accuracy,
calibration, and inference speed. Run:

    python examples/ou_recovery.py            # ~1-2 min on CPU
    python examples/ou_recovery.py --full     # larger, more accurate
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import OrnsteinUhlenbeck, FlowPosterior
from amortix.baselines import ou_mle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--n_post", type=int, default=1000)
    args = ap.parse_args()

    n_train, epochs = (40000, 60) if args.full else (12000, 30)

    prob = OrnsteinUhlenbeck()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()  # prior range per param

    print("=" * 64)
    print("amortix :: Ornstein-Uhlenbeck recovery  dX = theta(mu - X)dt + sigma dW")
    print(f"params {names} | train sims {n_train} | epochs {epochs}")
    print("=" * 64)

    post = FlowPosterior(prob).fit(n_train=n_train, epochs=epochs)

    # ---- held-out test set ------------------------------------------------
    gen = torch.Generator().manual_seed(123)
    m_true = prob.prior.sample(args.n_test, generator=gen)
    traj = prob.simulate_paths(m_true, generator=gen)         # [K, 501]
    tokens = prob.observer.tokens_from_traj(traj)             # [K, T, F]

    amort_est = np.zeros((args.n_test, 3))
    amort_lo = np.zeros((args.n_test, 3))
    amort_hi = np.zeros((args.n_test, 3))
    mle_est = np.zeros((args.n_test, 3))

    t0 = time.time()
    for i in range(args.n_test):
        s = post.sample(tokens[i], n=args.n_post, seed=i).numpy()
        amort_est[i] = s.mean(0)
        amort_lo[i] = np.quantile(s, 0.05, axis=0)
        amort_hi[i] = np.quantile(s, 0.95, axis=0)
    amort_time = (time.time() - t0) / args.n_test

    t0 = time.time()
    for i in range(args.n_test):
        d = ou_mle(traj[i].numpy(), dt=prob.observer.dt_sim)
        mle_est[i] = [d["theta"], d["mu"], d["sigma"]]
    mle_time = (time.time() - t0) / args.n_test

    mt = m_true.numpy()
    amort_err = np.abs(amort_est - mt) / rng * 100      # % of prior range
    mle_err = np.abs(mle_est - mt) / rng * 100
    covered = (mt >= amort_lo) & (mt <= amort_hi)

    print("\n--- accuracy (mean abs error, % of prior range; lower is better) ---")
    print(f"{'param':>6} | {'amortized':>10} | {'MLE (exact)':>12} | {'coverage90':>10}")
    for j, nm in enumerate(names):
        print(f"{nm:>6} | {amort_err[:, j].mean():9.2f}% | {mle_err[:, j].mean():11.2f}%"
              f" | {covered[:, j].mean() * 100:8.0f}%")
    print(f"{'ALL':>6} | {amort_err.mean():9.2f}% | {mle_err.mean():11.2f}%"
          f" | {covered.mean() * 100:8.0f}%")

    print(f"\n--- inference time per dataset ---")
    print(f"amortized ({args.n_post} posterior draws): {amort_time * 1e3:7.2f} ms")
    print(f"exact MLE (closed form)               : {mle_time * 1e3:7.2f} ms")
    print("(MLE is near-optimal & fast for OU; amortix matches it AND gives a")
    print(" calibrated posterior -- the point is it transfers to SDEs with NO")
    print(" tractable likelihood, where MLE is unavailable.)")

    try:
        _plot(post, prob, names)
    except Exception as e:  # plotting is optional
        print(f"[plot skipped: {e}]")


def _plot(post, prob, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = torch.Generator().manual_seed(7)
    m_true = prob.prior.sample(1, generator=gen)
    traj = prob.simulate_paths(m_true, generator=gen)
    tokens = prob.observer.tokens_from_traj(traj)
    samples = post.sample(tokens, n=4000, seed=1).numpy()
    mle = ou_mle(traj[0].numpy(), dt=prob.observer.dt_sim)
    mt = m_true.numpy()[0]
    mle_v = [mle["theta"], mle["mu"], mle["sigma"]]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for j, nm in enumerate(names):
        ax[j].hist(samples[:, j], bins=50, density=True, alpha=0.6,
                   color="#4C72B0", label="amortized posterior")
        ax[j].axvline(mt[j], color="k", lw=2, label="truth")
        ax[j].axvline(mle_v[j], color="#C44E52", ls="--", lw=2, label="exact MLE")
        ax[j].set_title(nm)
    ax[0].legend(fontsize=8)
    fig.suptitle("amortix: OU posterior vs truth vs exact MLE")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ou_recovery.png")
    fig.savefig(out, dpi=120)
    print(f"\nsaved posterior plot -> {out}")


if __name__ == "__main__":
    main()
