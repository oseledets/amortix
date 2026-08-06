"""Ornstein-Uhlenbeck process: a canonical SDE-recovery benchmark.

    dX = theta * (mu - X) dt + sigma dW

Recover m = (theta, mu, sigma) from observed paths. OU is special because it has
an exact Gaussian transition density, so a closed-form maximum-likelihood
estimator exists (see amortix.baselines) -- making it the ideal first case to
*validate* the amortized posterior against a known-optimal classical method.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..sde import SDEProblem, PathObserver, Channel


class OrnsteinUhlenbeck(SDEProblem):
    def __init__(self, n_paths: int = 1):
        # Canonical (centred) OU: two parameters. mu is a pure location parameter
        # that separates from the dynamics -- it added a third dimension without
        # adding a question, it was where the X0 = mu leak lived, and it polluted
        # the correlation measurement with a spurious mu-sigma pair. The
        # substantive dependence in OU is theta-sigma: both enter the transition
        # law through rho = exp(-theta dt) and sigma^2 (1 - rho^2) / (2 theta).
        self.prior = BoxUniform(
            low=[0.3, 0.2],
            high=[3.0, 1.5],
            names=["theta", "sigma"],
        )
        # internal integration grid + two observation channels:
        #   fast: every step over a short window  -> diffusion (sigma)
        #   slow: coarse over the full horizon    -> mean reversion (theta, mu)
        # n_paths independent replicate trajectories sharpen the posterior.
        self.observer = PathObserver(
            dt_sim=0.02, n_steps=500,
            channels=[
                Channel(every=1, count=50, label="fast"),
                Channel(every=20, count=24, label="slow"),
            ],
            n_paths=n_paths,
        )

    def drift(self, x, m):
        return -m[:, 0:1] * x                   # [B, 1]

    def diffusion(self, x, m):
        return m[:, 1:2].expand_as(x)           # constant additive diffusion [B, 1]

    def x0_sampler(self, m, generator=None):
        # Stationary start: X0 ~ N(0, sigma^2 / (2 theta)).
        theta, sigma = m[:, 0:1], m[:, 1:2]
        sd = sigma / torch.sqrt(2.0 * theta)
        return sd * torch.randn(sd.shape, generator=generator)


# --- gallery contract -----------------------------------------------------
SOTA_NAME = "exact MLE"


def make():
    return OrnsteinUhlenbeck()


def sota(tokens, traj, prob):
    from ..baselines import ou_mle
    path = np.asarray(traj)[:, 0]
    d = ou_mle(path, dt=prob.observer.dt_sim)
    return np.array([d["theta"], d["sigma"]])
