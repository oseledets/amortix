"""Posterior-pair sample dumps for linear-Gaussian and OU (paper figure)."""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/amortix_exp"))
sys.path.insert(0, os.path.expanduser("~/amortix_exp/examples"))
OUT = os.path.expanduser("~/amortix_exp/paper2")

from amortix import FlowPosterior
from amortix.mcmc import observed_indices, log_likelihood_ou, metropolis

which = sys.argv[1]

if which == "lg":
    from amortix.problems import linear_gaussian as lg
    prob = lg.make()
    torch.manual_seed(0)
    post = FlowPosterior(prob)
    post.fit(n_train=40000, epochs=35, seed=0, verbose=False, device="cuda")
    gen = torch.Generator().manual_seed(2024)
    m_true = prob.prior.sample(2, gen)
    tokens, _ = prob.observe(m_true, generator=gen)
    smp = post.sample_batch(tokens, n=4000, seed=0).numpy()
    # exact posterior draws for the same observations
    # tokens from _VectorObserver are [value, index]; column 0 is y
    refs = []
    for i in range(2):
        y_i = tokens[i, :, 0]
        refs.append(lg.exact_posterior(np.asarray(y_i), prob, n=4000,
                                       seed=100 + i).numpy())
    np.savez(f"{OUT}/pairs_lg.npz", amortix=smp, ref=np.stack(refs),
             m_true=m_true.numpy(), names=np.array(prob.prior.names))
    print("[pairs lg] saved", flush=True)

elif which == "ou":
    from amortix.problems import ou
    prob = ou.make()
    idx = observed_indices(prob)
    dt = prob.observer.dt_sim
    gaps = np.diff(idx).astype(np.float64) * dt
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    torch.manual_seed(0)
    post = FlowPosterior(prob)
    post.fit(n_train=20000, steps=6000, seed=0, verbose=False, device="cuda")
    gen = torch.Generator().manual_seed(2024)
    m_true = prob.prior.sample(2, gen)
    traj = prob.simulate_paths(m_true, gen)
    tokens = prob.observer.tokens_from_traj(traj)
    smp = post.sample_batch(tokens, n=4000, seed=0).numpy()
    refs = []
    for i in range(2):
        s = traj[i, idx, 0].numpy().astype(np.float64)

        def lp(v, s=s):
            th, sg = float(v[0]), float(v[1])
            if th <= 0 or sg <= 0:
                return -np.inf
            base = log_likelihood_ou(s, v, gaps, scheme="euler", dt_fine=dt)
            var0 = sg ** 2 / (2.0 * th)
            return base - 0.5 * (s[0] ** 2 / var0
                                 + math.log(2 * math.pi * var0))
        c, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=4000,
                          prior_low=lo, prior_high=hi, seed=100 + i)
        refs.append(np.asarray(c))
    np.savez(f"{OUT}/pairs_ou.npz", amortix=smp, ref=np.stack(refs),
             m_true=m_true.numpy(), names=np.array(prob.prior.names))
    print("[pairs ou] saved", flush=True)

print("DONE_C3", flush=True)
