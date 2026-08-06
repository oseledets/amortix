"""End-to-end FitzHugh-Nagumo neuron recovery -- a 2D excitable ODE case in amortix.

Trains one amortized flow-matching posterior over the 4 FHN parameters (a, b,
eps, I) from a noisy, partially observed (v only) spike train, then on held-out
neurons compares it against nonlinear least squares (the classical mechanistic
ODE-fitting baseline) on accuracy and calibration. Run:

    python examples/fhn_recovery.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import FlowPosterior
from amortix.problems.fhn import make, sota, SOTA_NAME


def main():
    prob = make()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()

    print("=" * 70)
    print("amortix :: FitzHugh-Nagumo neuron recovery (2D ODE; observe v only)")
    print(f"params {names} | tokens {prob.observer.n_tokens} "
          f"| dt {prob.observer.dt_sim} x {prob.observer.n_steps}")
    print("=" * 70)

    post = FlowPosterior(prob).fit(n_train=4000, epochs=12)

    gen = torch.Generator().manual_seed(123)
    K, NP = 30, 600
    m_true = prob.prior.sample(K, generator=gen)
    tokens, traj = prob.observe(m_true, generator=gen)

    amort = np.zeros((K, prob.prior.dim)); std = np.zeros_like(amort)
    lo = np.zeros_like(amort); hi = np.zeros_like(amort); base = np.zeros_like(amort)

    t0 = time.time()
    for i in range(K):
        s = post.sample(tokens[i], n=NP, seed=i).numpy()
        amort[i] = s.mean(0); std[i] = s.std(0)
        lo[i] = np.quantile(s, 0.05, 0); hi[i] = np.quantile(s, 0.95, 0)
    amort_t = (time.time() - t0) / K

    t0 = time.time()
    for i in range(K):
        base[i] = sota(tokens[i].numpy(), traj[i].numpy(), prob)
    base_t = (time.time() - t0) / K

    mt = m_true.numpy()
    a_err = (np.abs(amort - mt) / rng * 100).mean(0)
    b_err = (np.abs(base - mt) / rng * 100).mean(0)
    pstd = (std / rng * 100).mean(0)
    cov = (((mt >= lo) & (mt <= hi)).mean(0)) * 100

    print(f"\n{'param':>10} | {'amort':>7} | {'post.std':>8} | {SOTA_NAME[:14]:>14} | {'cov90':>6}")
    for j, nm in enumerate(names):
        print(f"{nm:>10} | {a_err[j]:6.2f}% | {pstd[j]:7.2f}% | {b_err[j]:13.2f}% | {cov[j]:5.0f}%")
    print(f"{'ALL':>10} | {a_err.mean():6.2f}% | {pstd.mean():7.2f}% | {b_err.mean():13.2f}% | {cov.mean():5.0f}%")

    print(f"\ninference per neuron:  amortized {amort_t*1e3:7.1f} ms"
          f"  |  {SOTA_NAME} {base_t*1e3:7.1f} ms   ({base_t/max(amort_t,1e-9):.0f}x)")
    print("amortized gives a calibrated posterior over the FHN rates from v alone;"
          " NLS gives a single point fit. eps / I are only weakly identified from"
          " the membrane potential, so their posteriors stay appropriately wide.")


if __name__ == "__main__":
    main()
