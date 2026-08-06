"""End-to-end double-well bistable SDE recovery with amortix.

    dX = (theta1 X - theta2 X^3) dt + sigma dW

Trains one amortized flow-matching posterior over (theta1, theta2, sigma), then
on held-out paths compares it against the classical Kramers-Moyal / SINDy
least-squares estimator on accuracy, posterior width, and calibration. This is a
"show the power" case: when a trajectory only visits one well the (theta1, theta2)
posterior is broad/curved, and the amortized net reports that uncertainty
honestly while the point-estimate baseline cannot. Run:

    python3 examples/double_well_recovery.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import FlowPosterior
from amortix.problems.double_well import make, sota, SOTA_NAME


def main():
    prob = make()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()

    print("=" * 70)
    print("amortix :: double-well recovery  dX = (theta1 X - theta2 X^3) dt + sigma dW")
    print(f"params {names} | tokens {prob.observer.n_tokens} | train 6000 | epochs 16")
    print("=" * 70)

    post = FlowPosterior(prob).fit(n_train=6000, epochs=16)

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

    print(f"\ninference per path:  amortized {amort_t*1e3:7.1f} ms"
          f"  |  {SOTA_NAME} {base_t*1e3:7.2f} ms")
    print("Kramers-Moyal LS gives a single drift/diffusion point estimate; amortix"
          " gives a calibrated posterior whose width widens exactly when a path"
          " stays in one well and theta2 (well location) is poorly identified.")


if __name__ == "__main__":
    main()
