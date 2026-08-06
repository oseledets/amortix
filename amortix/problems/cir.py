"""Cox-Ingersoll-Ross (CIR) process: a square-root mean-reverting SDE.

    dX = a (b - X) dt + sigma sqrt(X) dW,   X > 0,   X0 = b

CIR is the workhorse short-rate / stochastic-variance model (Heston's variance
follows a CIR). Recover m = (a, b, sigma) from observed paths:
  * a     -- mean-reversion speed (drift / slow-timescale info),
  * b     -- long-run mean        (drift / level info),
  * sigma -- volatility of vol    (diffusion / fast-timescale info).

Unlike OU, CIR has *state-dependent* diffusion sigma*sqrt(X), so the path's
local quadratic variation scales with the level X -- the fast channel still
exposes sigma, but now conditioned on X. The transition density is noncentral
chi-square (no simple closed-form MLE); the standard fast-data estimator is an
Euler pseudo-MLE: OLS for the drift and conditional quadratic variation for
sigma. We keep parameters in the Feller-stable region (2 a b >= sigma^2 mostly
holds over these ranges) so X stays positive; the simulator clamps X and uses
sqrt(clamp_min(X, 0)) for safety.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..sde import SDEProblem, PathObserver, Channel


class CIR(SDEProblem):
    state_dim = 1

    def __init__(self, n_paths: int = 1):
        self.prior = BoxUniform(
            low=[0.3, 0.3, 0.10],
            high=[3.0, 1.5, 0.50],
            names=["a", "b", "sigma"],
        )
        # internal integration grid + two observation channels:
        #   fast: every step over a short window  -> diffusion (sigma)
        #   slow: coarse over the full horizon    -> mean reversion (a, b)
        self.observer = PathObserver(
            dt_sim=0.02, n_steps=500,
            channels=[
                Channel(every=1, count=50, label="fast"),
                Channel(every=20, count=24, label="slow"),
            ],
            n_paths=n_paths,
            obs_dims=(0,),
        )

    def drift(self, x, m):
        a, b = m[:, 0:1], m[:, 1:2]
        return a * (b - x)                       # [B, 1]

    def diffusion(self, x, m):
        sigma = m[:, 2:3]
        return sigma * torch.sqrt(x.clamp_min(0.0))   # state-dependent [B, 1]

    def x0_sampler(self, m, generator=None):
        # Draw X0 from the CIR stationary law, Gamma(2ab/sigma^2, rate 2a/sigma^2),
        # whose mean is b. Sampled by inverse-CDF from uniforms taken off the
        # supplied generator, so runs stay reproducible.
        #
        # Do NOT start at X0 = b exactly: that hands the long-run mean to any
        # estimator that looks at the first observation, so b stops being inferred
        # and starts being read off (it was the best-recovered parameter for
        # exactly that reason).
        from scipy.stats import gamma as _gamma
        a, b, sigma = m[:, 0], m[:, 1], m[:, 2]
        conc = (2.0 * a * b / sigma ** 2).clamp(min=1e-3)
        scale = (sigma ** 2 / (2.0 * a))
        u = torch.rand(m.shape[0], generator=generator).clamp(1e-6, 1 - 1e-6)
        x0 = _gamma.ppf(u.numpy(), a=conc.numpy(), scale=scale.numpy())
        return torch.as_tensor(x0, dtype=m.dtype).clamp_min(1e-6).unsqueeze(-1)

    def simulate_paths(self, m, generator=None):
        # clamp the state positive after each Euler step (CIR positivity)
        x0 = self.x0_sampler(m, generator)
        dt = self.observer.dt_sim
        n_steps = self.observer.n_steps
        B, S = x0.shape
        traj = torch.empty(B, n_steps + 1, S, dtype=x0.dtype)
        x = x0
        traj[:, 0] = x
        sqrt_dt = dt ** 0.5
        for i in range(n_steps):
            z = torch.randn(B, S, generator=generator)
            x = x + self.drift(x, m) * dt + self.diffusion(x, m) * (z * sqrt_dt)
            x = x.clamp_min(0.0)
            traj[:, i + 1] = x
        return traj


def make():
    return CIR()


SOTA_NAME = "Euler pseudo-MLE"


def sota(tokens, traj, prob):
    """Euler-discretized pseudo-MLE on the fine path.

    The Euler transition is
        dX ~= a*b*dt - a*X*dt + sigma*sqrt(X)*sqrt(dt)*Z.
    Regress dX on [1, X] by OLS: dX ~= c0 + c1*X with c0 = a*b*dt, c1 = -a*dt
        => a_hat = -c1/dt,  b_hat = c0/(a_hat*dt).
    Estimate sigma from the conditional quadratic variation of the residual:
        eps = dX - (c0 + c1*X),  sigma_hat^2 = mean(eps^2 / (X*dt))  over X>tiny.
    """
    x = np.asarray(traj, dtype=np.float64).reshape(-1)      # [n_steps+1]
    dt = float(prob.observer.dt_sim)

    X = x[:-1]
    dX = x[1:] - x[:-1]

    # OLS: dX ~ c0 + c1 * X
    A = np.column_stack([np.ones_like(X), X])
    coef, *_ = np.linalg.lstsq(A, dX, rcond=None)
    c0, c1 = float(coef[0]), float(coef[1])

    a_hat = -c1 / dt
    # guard against degenerate / non-mean-reverting fits
    if not np.isfinite(a_hat) or abs(a_hat) < 1e-8:
        a_hat = 1e-3
    b_hat = c0 / (a_hat * dt)

    # sigma from conditional quadratic variation of the OLS residual
    eps = dX - (c0 + c1 * X)
    tiny = 1e-6
    mask = X > tiny
    if mask.any():
        sig2 = np.mean(eps[mask] ** 2 / (X[mask] * dt))
    else:
        sig2 = 0.0
    sigma_hat = float(np.sqrt(max(sig2, 0.0)))

    return np.array([a_hat, b_hat, sigma_hat], dtype=np.float64)
