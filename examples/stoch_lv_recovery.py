"""Stochastic Lotka-Volterra predator-prey recovery -- amortix ecology case.

Trains one amortized flow-matching posterior over the four LV drift rates
(alpha, beta, delta, gamma) of a 2D multiplicative-noise SDE, then on held-out
trajectories compares it against the classical SOTA baseline: a deterministic
LV-ODE fit by nonlinear least squares. Run from the repo root:

    python examples/stoch_lv_recovery.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import FlowPosterior
from amortix.problems.stoch_lv import make, sota, SOTA_NAME


def main():
    prob = make()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()
    post = FlowPosterior(prob).fit(n_train=4000, epochs=12)   # small for validation
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


if __name__ == "__main__":
    main()
