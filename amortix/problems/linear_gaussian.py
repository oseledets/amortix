"""Linear-Gaussian inverse problem -- the reference testbed with an EXACT posterior.

    m ~ U([-3, 3]^4),   y = A m + eps,   eps ~ N(0, s^2 I_6)

With a flat prior inside the box the posterior is a Gaussian restricted to the box,

    Sigma = (A^T A / s^2)^-1,     mu = Sigma A^T y / s^2,

so `exact_posterior(y, prob, n)` returns *ground-truth* posterior samples by
rejection. `A` is deliberately built with correlated columns, so the true
posterior has strong off-diagonal covariance.

Why this case exists: every other case can only be checked against SBC
(self-consistency) or a classical point estimator. Here we know the answer, so a
mismatch is unambiguously the fault of the inference machinery -- not of the
simulator, the summary statistics or the budget. It is the instrument for
debugging conditional flow matching itself.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform

D_PARAM = 4
N_OBS = 6
NOISE = 0.5

# fixed, correlated design matrix (seeded once, not learned)
_g = torch.Generator().manual_seed(20240630)
A = (torch.randn(N_OBS, D_PARAM, generator=_g) * 0.6
     + torch.linspace(0.4, 1.0, D_PARAM)[None, :])


class _VectorObserver:
    """Each observation component becomes one token: [value, index]."""
    N_FEATURES = 2

    def __init__(self, n_obs):
        self.n_tokens = n_obs

    def tokens(self, y):                       # y [B, N_OBS] -> [B, N_OBS, 2]
        B, T = y.shape
        idx = (torch.arange(T, dtype=torch.float32) / T).expand(B, T)
        return torch.stack([y, idx], dim=-1)


class LinearGaussian:
    """Reference problem: exact posterior known in closed form."""

    state_dim = 1

    def __init__(self):
        self.prior = BoxUniform(low=[-3.0] * D_PARAM, high=[3.0] * D_PARAM,
                                names=[f"m{i+1}" for i in range(D_PARAM)])
        self.observer = _VectorObserver(N_OBS)
        self.A = A
        self.noise = NOISE
        # posterior covariance is data-independent for a linear-Gaussian model
        self.Sigma = torch.linalg.inv(A.T @ A / NOISE ** 2)

    def _forward(self, m, generator=None):
        y = m @ self.A.T
        return y + self.noise * torch.randn(y.shape, generator=generator)

    def simulate(self, n, generator=None):
        m = self.prior.sample(n, generator)
        return m, self.observer.tokens(self._forward(m, generator))

    def observe(self, m, generator=None):
        y = self._forward(m, generator)
        return self.observer.tokens(y), y            # "traj" is the raw observation

    # --- ground truth -----------------------------------------------------
    def posterior_moments(self, y):
        """Unconstrained Gaussian posterior (before the box restriction)."""
        mu = (self.Sigma @ (self.A.T @ y.reshape(-1, 1) / self.noise ** 2)).squeeze(-1)
        return mu, self.Sigma


def exact_posterior(y, prob, n=2000, seed=0, max_tries=200):
    """Exact posterior samples for one observation y, by rejection inside the box."""
    g = torch.Generator().manual_seed(seed)
    mu, Sigma = prob.posterior_moments(torch.as_tensor(y, dtype=torch.float32))
    L = torch.linalg.cholesky(Sigma)
    out, tries = [], 0
    while sum(x.shape[0] for x in out) < n and tries < max_tries:
        z = torch.randn(4 * n, D_PARAM, generator=g)
        cand = mu[None] + z @ L.T
        keep = ((cand >= prob.prior.low) & (cand <= prob.prior.high)).all(-1)
        out.append(cand[keep])
        tries += 1
    return torch.cat(out)[:n]


# --- gallery contract -----------------------------------------------------
SOTA_NAME = "exact posterior mean"


def make():
    return LinearGaussian()


def sota(tokens, traj, prob):
    """The Bayes-optimal point estimate: the mean of the exact posterior."""
    s = exact_posterior(np.asarray(traj), prob, n=800)
    return s.mean(0).numpy()
