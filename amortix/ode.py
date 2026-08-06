"""Deterministic ODE problems: simulation and time-series observation.

Parallel to sde.py but for ordinary differential equations dX/dt = f(X, m, t).
The observation is a (possibly noisy, multi-output) time series, turned into a
token set [t, value, channel] by TimeSeriesObserver. This is the abstraction the
paper's SEIR / disease-dynamics case lives in.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from .prior import BoxUniform


def rk4(rhs, x0: torch.Tensor, m: torch.Tensor, dt: float, n_steps: int) -> torch.Tensor:
    """Batched 4th-order Runge-Kutta.

    rhs(x[B, state], m[B, d], t: float) -> [B, state].
    Returns solution [B, n_steps + 1, state].
    """
    B, S = x0.shape
    sol = torch.empty(B, n_steps + 1, S, dtype=x0.dtype)
    x = x0
    sol[:, 0] = x
    for i in range(n_steps):
        t = i * dt
        k1 = rhs(x, m, t)
        k2 = rhs(x + 0.5 * dt * k1, m, t + 0.5 * dt)
        k3 = rhs(x + 0.5 * dt * k2, m, t + 0.5 * dt)
        k4 = rhs(x + dt * k3, m, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sol[:, i + 1] = x
    return sol


class TimeSeriesObserver:
    """Observe selected state components at selected times -> token set.

    Token features: [t / horizon, value (+noise), channel_id]. With several
    observed components (e.g. infected & deceased) the channel feature lets the
    transformer tell the series apart while staying one permutation-invariant set.
    """

    N_FEATURES = 3

    def __init__(self, dt_sim: float, n_steps: int, obs_steps: List[int],
                 obs_indices: List[int], noise_std: float = 0.0):
        self.dt_sim = float(dt_sim)
        self.n_steps = int(n_steps)
        self.obs_steps = torch.as_tensor(obs_steps, dtype=torch.long)
        self.obs_indices = list(obs_indices)
        self.noise_std = float(noise_std)
        self.horizon = dt_sim * n_steps

    @property
    def n_tokens(self) -> int:
        return len(self.obs_steps) * len(self.obs_indices)

    def tokens_from_solution(self, sol: torch.Tensor,
                             generator: torch.Generator = None) -> torch.Tensor:
        B = sol.shape[0]
        t = (self.obs_steps.float() * self.dt_sim) / self.horizon  # [n_obs]
        feats = []
        for ch, si in enumerate(self.obs_indices):
            val = sol[:, self.obs_steps, si]                       # [B, n_obs]
            if self.noise_std > 0:
                val = val + self.noise_std * torch.randn(val.shape, generator=generator)
            chan = torch.full((self.obs_steps.numel(),), float(ch))
            tok = torch.stack([t.expand(B, -1), val, chan.expand(B, -1)], dim=-1)
            feats.append(tok)
        return torch.cat(feats, dim=1)


class ODEProblem:
    prior: BoxUniform
    observer: TimeSeriesObserver
    state_dim: int

    def rhs(self, x, m, t):
        raise NotImplementedError

    def x0(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def simulate_solution(self, m: torch.Tensor) -> torch.Tensor:
        return rk4(self.rhs, self.x0(m), m, self.observer.dt_sim, self.observer.n_steps)

    def simulate(self, n: int,
                 generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.prior.sample(n, generator)
        sol = self.simulate_solution(m)
        tokens = self.observer.tokens_from_solution(sol, generator)
        return m, tokens

    def observe(self, m: torch.Tensor,
                generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sol = self.simulate_solution(m)
        tokens = self.observer.tokens_from_solution(sol, generator)
        return tokens, sol
