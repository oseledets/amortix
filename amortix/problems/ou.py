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
        self.prior = BoxUniform(
            low=[0.3, -1.0, 0.2],
            high=[3.0, 1.0, 1.5],
            names=["theta", "mu", "sigma"],
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
        theta, mu = m[:, 0:1], m[:, 1:2]
        return theta * (mu - x)                 # [B, 1]

    def diffusion(self, x, m):
        return m[:, 2:3].expand_as(x)           # constant additive diffusion [B, 1]

    def x0_sampler(self, m, generator=None):
        # start at the mean so paths are stationary-ish from the outset
        return m[:, 1:2].clone()                # [B, 1]


# --- gallery contract -----------------------------------------------------
SOTA_NAME = "exact MLE"


def make():
    return OrnsteinUhlenbeck()


def sota(tokens, traj, prob):
    from ..baselines import ou_mle
    path = np.asarray(traj)[:, 0]
    d = ou_mle(path, dt=prob.observer.dt_sim)
    return np.array([d["theta"], d["mu"], d["sigma"]])
