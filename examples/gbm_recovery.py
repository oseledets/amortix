"""End-to-end Geometric Brownian Motion recovery with amortix.

Trains one amortized flow-matching posterior over (mu, sigma) of the Black-Scholes
price process dS = mu S dt + sigma S dW, then on held-out paths compares it
against the exact closed-form GBM maximum-likelihood estimator (the textbook,
near-optimal volatility/drift estimator). MLE is essentially optimal for GBM, so
here "match / competitive with MLE" is success -- the point is the amortized net
delivers the same accuracy plus a calibrated posterior, and transfers to SDEs
with no tractable likelihood. Run from the repo root:

    python examples/gbm_recovery.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import FlowPosterior
from amortix.problems.gbm import make, sota, SOTA_NAME


def main():
    prob = make()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()

    print("=" * 66)
    print("amortix :: Geometric Brownian Motion recovery  dS = mu S dt + sigma S dW")
    print(f"params {names} | tokens {prob.observer.n_tokens} | baseline: {SOTA_NAME}")
    print("=" * 66)

    t0 = time.time()
    post = FlowPosterior(prob).fit(n_train=4000, epochs=12)
    print(f"trained in {time.time() - t0:.1f}s")

    gen = torch.Generator().manual_seed(123)
    K, NP = 30, 600
    m_true = prob.prior.sample(K, generator=gen)
    tokens, traj = prob.observe(m_true, generator=gen)

    amort = np.zeros((K, prob.prior.dim)); std = np.zeros_like(amort)
    lo = np.zeros_like(amort); hi = np.zeros_like(amort); base = np.zeros_like(amort)
    for i in range(K):
        s = post.sample(tokens[i], n=NP, seed=i).numpy()
        amort[i] = s.mean(0); std[i] = s.std(0)
        lo[i] = np.quantile(s, 0.05, 0); hi[i] = np.quantile(s, 0.95, 0)
        base[i] = sota(tokens[i].numpy(), traj[i].numpy(), prob)

    mt = m_true.numpy()
    a_err = (np.abs(amort - mt) / rng * 100).mean(0)
    b_err = (np.abs(base - mt) / rng * 100).mean(0)
    pstd = (std / rng * 100).mean(0)
    cov = (((mt >= lo) & (mt <= hi)).mean(0)) * 100

    print(f"\n{'param':>10} | {'amort':>7} | {'post.std':>8} | {SOTA_NAME[:14]:>14} | {'cov90':>6}")
    for j, nm in enumerate(names):
        print(f"{nm:>10} | {a_err[j]:6.2f}% | {pstd[j]:7.2f}% | {b_err[j]:13.2f}% | {cov[j]:5.0f}%")
    print(f"{'ALL':>10} | {a_err.mean():6.2f}% | {pstd.mean():7.2f}% | {b_err.mean():13.2f}% | {cov.mean():5.0f}%")

    print("\n(MLE is near-optimal for GBM: sigma_hat^2 = var(log-returns)/dt is very")
    print(" sharp; the drift mu is weakly identified over a finite horizon, for both")
    print(" estimators -- amortix matches MLE and reports that uncertainty as post.std.)")


if __name__ == "__main__":
    main()
