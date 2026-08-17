"""Random-design versions of the two calibration workhorses, GBM and OU.

These exist so that every claim of the paper can be made in the package's
default mode -- one network answering p(m | any K observation points) --
with exact or exact-likelihood references available *per design*:

* GBM: the conjugate normal--inverse-chi^2 posterior conditions on log-price
  increments over arbitrary gaps (``gbm_exact_from_points``);
* OU: the per-gap Euler--Maruyama transition density is Gaussian for any
  gap pattern, so adaptive Metropolis gives a reference for any design.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..prior import BoxUniform
from ..designs import DesignObserver, DesignProblem


class GBMDesign(DesignProblem):
    """dS = mu S dt + sigma S dW, S0 = 1, observed at arbitrary times."""

    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(low=[-0.20, 0.10], high=[0.40, 0.60],
                                names=["mu", "sigma"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        dt = self.observer.dt_sim
        n = self.observer.n_steps
        mu, sigma = m[:, 0], m[:, 1]
        logS = torch.zeros(B)
        out = torch.zeros(B, n + 1, 1)
        out[:, 0, 0] = 1.0
        sq = math.sqrt(dt)
        for i in range(n):
            z = torch.randn(B, generator=generator)
            logS = logS + (mu - 0.5 * sigma ** 2) * dt + sigma * sq * z
            out[:, i + 1, 0] = torch.exp(logS)
        return out


class OUDesign(DesignProblem):
    """dX = -theta X dt + sigma dW, stationary start, arbitrary times."""

    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(low=[0.3, 0.2], high=[3.0, 1.5],
                                names=["theta", "sigma"])
        self.observer = DesignObserver(dt_sim=0.02, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        dt = self.observer.dt_sim
        n = self.observer.n_steps
        theta, sigma = m[:, 0], m[:, 1]
        x = (sigma / torch.sqrt(2.0 * theta)) * torch.randn(B, generator=generator)
        out = torch.zeros(B, n + 1, 1)
        out[:, 0, 0] = x
        sq = math.sqrt(dt)
        for i in range(n):
            z = torch.randn(B, generator=generator)
            x = x + (-theta * x) * dt + sigma * sq * z
            out[:, i + 1, 0] = x
        return out


def gbm_exact_from_points(prob, traj_i, tidx, n_samples=2000, seed=0,
                          pool_factor=8):
    """Exact per-design GBM posterior: conjugate in (b, sigma^2) on the
    log-increments over the observed gaps, importance-corrected to the box
    prior. ``tidx`` are fine-grid indices (the S0 = 1 anchor at index 0 is
    prepended automatically)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(traj_i, dtype=np.float64).reshape(-1)
    idx = np.unique(np.concatenate([[0], np.asarray(tidx, dtype=np.int64)]))
    vals = x[idx]
    tau = np.diff(idx).astype(np.float64) * float(prob.observer.dt_sim)
    r = np.diff(np.log(np.maximum(vals, 1e-12)))
    n = r.size
    T = float(tau.sum())
    R1 = float(r.sum())
    SSw = float((r ** 2 / tau).sum()) - R1 ** 2 / T
    low = prob.prior.low.numpy().astype(np.float64)
    high = prob.prior.high.numpy().astype(np.float64)

    # The conjugate draw is exact only up to the importance correction onto the
    # box prior, and that correction is not free: at dense designs the
    # posterior is narrow, and when the truth sits near a corner of the box
    # most candidates land outside it. One battery set had 45 usable draws out
    # of 32,000, so its "exact" sample was fifty distinct values with repeats
    # and two independent draws disagreed by 0.4 posterior sd. The pool is
    # therefore grown until the effective sample size is adequate, and the
    # caller is told when it cannot be.
    draws_all, w_all = [], []
    npool = n_samples * pool_factor
    for _ in range(6):
        chi = rng.chisquare(max(n - 1, 1), size=npool)
        v = SSw / np.maximum(chi, 1e-300)
        b = R1 / T + np.sqrt(v / T) * rng.standard_normal(npool)
        sigma = np.sqrt(v)
        d = np.stack([b + 0.5 * v, sigma], axis=1)
        draws_all.append(d)
        w_all.append(sigma * np.all((d >= low) & (d <= high), axis=1))
        w = np.concatenate(w_all)
        if w.sum() > 0 and w.sum() ** 2 / (w ** 2).sum() >= 10 * n_samples:
            break
        npool *= 4
    draws = np.concatenate(draws_all)
    if w.sum() <= 0:
        raise RuntimeError("no conjugate draws inside the prior box")
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    if ess < n_samples:
        raise RuntimeError(
            f"GBM conjugate reference is degenerate here: effective sample "
            f"size {ess:.0f} for {n_samples} requested draws (the posterior "
            f"lies mostly outside the prior box). Exclude this observation "
            f"set rather than trusting the reference.")
    pick = rng.choice(len(draws), size=n_samples, replace=True, p=w / w.sum())
    return draws[pick]


def ou_logpost_factory(prob, traj_i, tidx):
    """Exact log-posterior of the generative (Euler) OU chain on an arbitrary
    design, including the stationary density of the informative start."""
    from ..mcmc import log_likelihood_ou

    x = np.asarray(traj_i, dtype=np.float64).reshape(-1)
    idx = np.unique(np.concatenate([[0], np.asarray(tidx, dtype=np.int64)]))
    s = x[idx]
    dt = float(prob.observer.dt_sim)
    gaps = np.diff(idx).astype(np.float64) * dt

    def log_post(p):
        th, sg = float(p[0]), float(p[1])
        if th <= 0 or sg <= 0:
            return -np.inf
        base = log_likelihood_ou(s, p, gaps, scheme="euler", dt_fine=dt)
        var0 = sg ** 2 / (2.0 * th)
        return base - 0.5 * (s[0] ** 2 / var0 + math.log(2 * math.pi * var0))

    return log_post
