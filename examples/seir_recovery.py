"""End-to-end SEIRD epidemic recovery -- the paper's disease-dynamics case in amortix.

Trains one amortized flow-matching posterior over the 5 epidemic rates, then on
held-out outbreaks compares it against nonlinear least-squares (the classical
curve-fitting baseline, same family arXiv:2604.22496 benchmarks CFM against) on
accuracy, calibration, and speed. Run:

    python examples/seir_recovery.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import SEIRD, FlowPosterior
from amortix.baselines import seird_nls

N_TRAIN, EPOCHS = 10000, 30
N_TEST, N_POST = 40, 1000


def main():
    prob = SEIRD()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()
    obs = prob.observer
    n_obs = len(obs.obs_steps)

    print("=" * 66)
    print("amortix :: SEIRD epidemic recovery (two-phase beta; observe I, D)")
    print(f"params {names} | tokens {obs.n_tokens} | train {N_TRAIN} | epochs {EPOCHS}")
    print("=" * 66)

    post = FlowPosterior(prob).fit(n_train=N_TRAIN, epochs=EPOCHS)

    gen = torch.Generator().manual_seed(123)
    m_true = prob.prior.sample(N_TEST, generator=gen)
    tokens, _ = prob.observe(m_true, generator=gen)         # noisy observations

    amort = np.zeros((N_TEST, 5)); std = np.zeros((N_TEST, 5))
    lo = np.zeros((N_TEST, 5)); hi = np.zeros((N_TEST, 5))
    nls = np.zeros((N_TEST, 5))

    t0 = time.time()
    for i in range(N_TEST):
        s = post.sample(tokens[i], n=N_POST, seed=i).numpy()
        amort[i] = s.mean(0); std[i] = s.std(0)
        lo[i] = np.quantile(s, 0.05, 0); hi[i] = np.quantile(s, 0.95, 0)
    amort_t = (time.time() - t0) / N_TEST

    t0 = time.time()
    for i in range(N_TEST):
        I_obs = tokens[i, :n_obs, 1].numpy()
        D_obs = tokens[i, n_obs:, 1].numpy()
        d = seird_nls(I_obs, D_obs, obs.obs_steps.numpy(), obs.dt_sim,
                      obs.n_steps, prob.t_switch,
                      prob.prior.low.numpy(), prob.prior.high.numpy())
        nls[i] = [d[k] for k in names]
    nls_t = (time.time() - t0) / N_TEST

    mt = m_true.numpy()
    a_err = np.abs(amort - mt) / rng * 100
    n_err = np.abs(nls - mt) / rng * 100
    pstd = std / rng * 100
    cov = ((mt >= lo) & (mt <= hi)).mean(0) * 100

    print(f"\n{'param':>8} | {'amort err':>9} | {'post.std':>8} | {'NLS err':>8} | {'cov90':>6}")
    for j, nm in enumerate(names):
        print(f"{nm:>8} | {a_err[:, j].mean():8.2f}% | {pstd[:, j].mean():7.2f}%"
              f" | {n_err[:, j].mean():7.2f}% | {cov[j]:5.0f}%")
    print(f"{'ALL':>8} | {a_err.mean():8.2f}% | {pstd.mean():7.2f}%"
          f" | {n_err.mean():7.2f}% | {cov.mean():5.0f}%")

    print(f"\ninference per outbreak:  amortized {amort_t*1e3:7.1f} ms"
          f"  |  NLS {nls_t*1e3:7.1f} ms   ({nls_t/max(amort_t,1e-9):.0f}x)")
    print("amortized gives a calibrated posterior (uncertainty + speed); NLS gives"
          " a single point. On held-out outbreaks they should be comparable in"
          " accuracy -- validating the amortized net on a paper case.")


if __name__ == "__main__":
    main()
