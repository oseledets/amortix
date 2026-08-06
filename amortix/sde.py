"""Stochastic differential equations: simulation and observation.

The forward model for an SDE problem is

    dX = drift(X, m) dt + diffusion(X, m) . dW          (X in R^S)

We simulate trajectories with a vectorized Euler-Maruyama integrator (fully
batched over the parameter draws -- this is what makes amortized training cheap)
and turn each trajectory into a *set of tokens* via a PathObserver. State may be
vector-valued (S >= 1) with optionally correlated Brownian noise, so multi-
dimensional systems (Heston, stochastic Lotka-Volterra) use the same core.

For stiff / production SDEs, `simulate_paths` is the single place to swap in
torchsde; the rest of the package only depends on the token tensors it returns.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch

from .prior import BoxUniform


def euler_maruyama(drift: Callable, diffusion: Callable, x0: torch.Tensor,
                   m: torch.Tensor, dt: float, n_steps: int,
                   generator: torch.Generator = None,
                   corr_chol: torch.Tensor = None) -> torch.Tensor:
    """Batched Euler-Maruyama for vector states.

    Args:
        drift, diffusion: callables (x[B, S], m[B, d]) -> [B, S] (diagonal noise).
        x0: [B, S] initial states.
        m:  [B, d] parameter batch.
        dt: integration step; n_steps -> returns trajectory [B, n_steps + 1, S].
        corr_chol: optional [S, S] lower-Cholesky of the Brownian correlation
            matrix (for correlated components, e.g. Heston price/vol).
    """
    B, S = x0.shape
    traj = torch.empty(B, n_steps + 1, S, dtype=x0.dtype)
    x = x0
    traj[:, 0] = x
    sqrt_dt = dt ** 0.5
    for i in range(n_steps):
        z = torch.randn(B, S, generator=generator)
        if corr_chol is not None:
            z = z @ corr_chol.T
        dw = z * sqrt_dt
        x = x + drift(x, m) * dt + diffusion(x, m) * dw
        traj[:, i + 1] = x
    return traj


class Channel:
    """One observation channel: a (resolution, horizon) view of a trajectory.

    `every` subsamples the fine integration grid; `count` increments are kept.
    A "fast" channel (every=1, short) carries diffusion information; a "slow"
    channel (every>>1, long) carries drift / mean-reversion information. Feeding
    both is what lets a single amortized network identify drift and diffusion
    jointly.
    """

    def __init__(self, every: int, count: int, label: str = ""):
        self.every = int(every)
        self.count = int(count)
        self.label = label


class PathObserver:
    """Turns trajectories into token sets for the transformer encoder.

    Each token describes one observed increment of one observed component:
        [t/horizon, x, dx, dx^2, log10(dt_obs), component_id]
    dx^2 is the per-step quadratic-variation cue (-> diffusion), the resolution
    feature tells the encoder which timescale the token is from, and component_id
    distinguishes observed state components (0 for scalar SDEs).
    """

    N_FEATURES = 6

    def __init__(self, dt_sim: float, n_steps: int, channels: List[Channel],
                 horizon: float = None, n_paths: int = 1,
                 obs_dims: Sequence[int] = (0,)):
        self.dt_sim = float(dt_sim)
        self.n_steps = int(n_steps)
        self.channels = channels
        self.horizon = horizon if horizon is not None else dt_sim * n_steps
        self.n_paths = int(n_paths)
        self.obs_dims = list(obs_dims)

    @property
    def tokens_per_path(self) -> int:
        return sum(c.count for c in self.channels) * len(self.obs_dims)

    @property
    def n_tokens(self) -> int:
        return self.n_paths * self.tokens_per_path

    def tokens_from_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """traj [B, n_steps+1, S] (or [B, n_steps+1]) -> tokens [B, T, N_FEATURES]."""
        if traj.dim() == 2:
            traj = traj.unsqueeze(-1)
        B = traj.shape[0]
        feats = []
        for ci, od in enumerate(self.obs_dims):
            comp = traj[:, :, od]                            # [B, n_steps+1]
            for c in self.channels:
                idx = torch.arange(0, c.count + 1) * c.every
                idx = idx.clamp(max=self.n_steps)
                seg = comp[:, idx]                           # [B, count+1]
                x = seg[:, :-1]
                dx = seg[:, 1:] - seg[:, :-1]
                dt_obs = c.every * self.dt_sim
                t = (idx[:-1].float() * self.dt_sim) / self.horizon
                res = torch.full((c.count,), torch.log10(torch.tensor(dt_obs)).item())
                cid = torch.full((c.count,), float(ci))
                tok = torch.stack([
                    t.expand(B, -1),
                    x,
                    dx,
                    dx * dx,
                    res.expand(B, -1),
                    cid.expand(B, -1),
                ], dim=-1)                                    # [B, count, F]
                feats.append(tok)
        return torch.cat(feats, dim=1)


class SDEProblem:
    """Base class for SDE parameter-recovery problems.

    Subclasses define `prior`, `drift`, `diffusion`, `x0_sampler` and an
    `observer` (PathObserver); drift/diffusion return [B, S]. Set `corr_chol`
    (a [S, S] lower-Cholesky) for correlated noise. `simulate` returns (m, tokens).
    """

    prior: BoxUniform
    observer: PathObserver
    corr_chol: torch.Tensor = None
    state_dim: int = 1

    def drift(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diffusion(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def x0_sampler(self, m: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        return torch.zeros(m.shape[0], self.state_dim)

    def simulate_paths(self, m: torch.Tensor,
                       generator: torch.Generator = None) -> torch.Tensor:
        x0 = self.x0_sampler(m, generator)
        return euler_maruyama(self.drift, self.diffusion, x0, m,
                              self.observer.dt_sim, self.observer.n_steps,
                              generator, corr_chol=self.corr_chol)

    def _tokens_for(self, m: torch.Tensor, generator=None):
        """Simulate observer.n_paths independent trajectories per parameter and
        concatenate their tokens into one set [B, n_paths * tokens_per_path, F]."""
        toks = []
        last = None
        for _ in range(self.observer.n_paths):
            last = self.simulate_paths(m, generator)
            toks.append(self.observer.tokens_from_traj(last))
        return torch.cat(toks, dim=1), last

    def simulate(self, n: int,
                 generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw n parameter sets and return (m [n,d], tokens [n,T,F])."""
        m = self.prior.sample(n, generator)
        tokens, _ = self._tokens_for(m, generator)
        return m, tokens

    def observe(self, m: torch.Tensor,
                generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Like simulate but also returns the last raw trajectory (for baselines)."""
        tokens, traj = self._tokens_for(m, generator)
        return tokens, traj
