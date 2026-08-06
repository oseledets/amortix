"""Stochastic Lotka-Volterra predator-prey: a 2D ecological SDE-recovery case.

State x = prey, y = predator, with fixed small multiplicative noise:

    dx = (alpha * x - beta  * x * y) dt + s1 * x dW1
    dy = (delta * x * y - gamma * y) dt + s2 * y dW2

Recover m = (alpha, beta, delta, gamma) from a single observed trajectory of
BOTH components. The diffusion is *known* (s1 = s2 = 0.05, small multiplicative
noise) and not recovered -- the unknowns are the four drift rates that set the
period and amplitude of the predator-prey oscillations.

This is a genuinely nonlinear, multi-output, oscillatory SDE: the same drift
identifies as the deterministic LV ODE, so the classical SOTA baseline here is a
deterministic nonlinear-least-squares fit of that ODE to the noisy path. The
amortized posterior must match (and quantify the uncertainty of) that fit while
seeing only the raw tokenized trajectory.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..sde import SDEProblem, PathObserver, Channel

# fixed, known multiplicative noise levels (NOT recovered)
S1 = 0.05
S2 = 0.05

# initial state (prey, predator)
X0 = [1.0, 1.0]


class StochasticLotkaVolterra(SDEProblem):
    state_dim = 2  # prey, predator

    def __init__(self):
        self.prior = BoxUniform(
            low=[0.8, 0.4, 0.4, 0.8],
            high=[1.5, 1.2, 1.2, 1.5],
            names=["alpha", "beta", "delta", "gamma"],
        )
        # horizon 6 = dt_sim * n_steps -> a few predator-prey oscillations.
        # fast channel: every step over a short window -> diffusion / fine drift.
        # slow channel: coarse over the full horizon  -> oscillation rates.
        self.observer = PathObserver(
            dt_sim=0.01, n_steps=600,
            channels=[
                Channel(every=1, count=50, label="fast"),
                Channel(every=15, count=40, label="slow"),   # spans the full horizon
            ],
            obs_dims=(0, 1),          # observe both prey and predator
        )

    def drift(self, x, m):
        xc = x.clamp_min(1e-6)
        prey, pred = xc[:, 0:1], xc[:, 1:2]
        alpha, beta = m[:, 0:1], m[:, 1:2]
        delta, gamma = m[:, 2:3], m[:, 3:4]
        dprey = alpha * prey - beta * prey * pred
        dpred = delta * prey * pred - gamma * pred
        return torch.cat([dprey, dpred], dim=1)          # [B, 2]

    def diffusion(self, x, m):
        xc = x.clamp_min(1e-6)
        return torch.cat([S1 * xc[:, 0:1], S2 * xc[:, 1:2]], dim=1)  # [B, 2]

    def x0_sampler(self, m, generator=None):
        B = m.shape[0]
        x0 = torch.tensor(X0, dtype=m.dtype).unsqueeze(0).expand(B, -1).clone()
        return x0                                        # [B, 2]


def make() -> StochasticLotkaVolterra:
    return StochasticLotkaVolterra()


# --------------------------------------------------------------------------- #
# SOTA / classical baseline: deterministic LV ODE fit by nonlinear least squares
# --------------------------------------------------------------------------- #
SOTA_NAME = "deterministic NLS"


def _lv_rk4(params, x0, dt, n_steps):
    """Pure-numpy RK4 of the deterministic Lotka-Volterra ODE.

    Returns the trajectory [n_steps + 1, 2] of (prey, predator).
    """
    alpha, beta, delta, gamma = params

    def f(state):
        prey = max(state[0], 1e-6)
        pred = max(state[1], 1e-6)
        return np.array([
            alpha * prey - beta * prey * pred,
            delta * prey * pred - gamma * pred,
        ])

    traj = np.empty((n_steps + 1, 2))
    x = np.asarray(x0, dtype=float)
    traj[0] = x
    for i in range(n_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = x
    return traj


def sota(tokens, traj, prob) -> np.ndarray:
    """Deterministic Lotka-Volterra ODE fit to the observed path by NLS.

    Fits (alpha, beta, delta, gamma) of the noise-free LV ODE so that an RK4
    simulation from x0 = [1, 1] over the same fine grid matches the observed
    (noisy) trajectory at a subsampled set of times, for BOTH components.
    Bounded to the prior box and started at the prior mean.
    """
    from scipy.optimize import least_squares

    obs = prob.observer
    dt = obs.dt_sim
    n_steps = obs.n_steps

    traj = np.asarray(traj, dtype=float)                 # [n_steps+1, 2]
    obs_idx = np.arange(0, n_steps + 1, 6)               # subsample every 6 steps
    target = traj[obs_idx]                               # [K, 2] both components

    low = prob.prior.low.numpy().astype(float)
    high = prob.prior.high.numpy().astype(float)
    p0 = 0.5 * (low + high)

    def residual(params):
        sim = _lv_rk4(params, X0, dt, n_steps)
        return (sim[obs_idx] - target).ravel()

    res = least_squares(
        residual, p0, bounds=(low, high), max_nfev=200,
    )
    return np.asarray(res.x, dtype=float)
