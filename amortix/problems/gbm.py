"""Geometric Brownian Motion -- the canonical finance SDE-recovery case.

    dS = mu * S dt + sigma * S dW,    S > 0,   S0 = 1.

GBM is the Black-Scholes price process. We recover m = (mu, sigma) -- the drift
(expected log-growth) and the volatility -- from a single observed price path.

It is the ideal *finance* counterpart to the OU benchmark: like OU it has a
tractable likelihood, because log-returns r_i = log(S_{i+1}/S_i) are i.i.d.
Gaussian with mean (mu - sigma^2/2) dt and variance sigma^2 dt. That yields a
closed-form, near-optimal estimator (`sota` below) to validate the amortized
posterior against. The catch finance practitioners know well: volatility is
estimated very sharply from the quadratic variation, but the drift mu is
notoriously *hard* to pin down over any finite horizon -- so it is the slow
channel and the post.std that should reveal mu's weak identifiability.

Stability: under Euler-Maruyama a large-sigma GBM step can push S <= 0, after
which mu*S and sigma*S would carry the path negative. We clamp the state to
1e-8 inside drift/diffusion so the multiplicative dynamics stay positive,
matching the contract's CIR/GBM positivity guidance.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..sde import SDEProblem, PathObserver, Channel


S0 = 1.0
DT_SIM = 0.01
N_STEPS = 500          # horizon = 5 "years"
CLAMP = 1e-8


class LogPathObserver(PathObserver):
    """Reference observer that tokenizes log(S) rather than S.

    Kept as the exact-sufficient-statistic BENCHMARK, not the default: dx
    becomes the exact log-return and dx^2 the exact per-step quadratic
    variation. The default pipeline instead keeps raw-price tokens and relies
    on the universal learnable warped-increment embedding (embed="wdiff",
    resolved automatically for PathObserver problems), which reaches the same
    calibration (sigma SBC p=0.876 vs 0.812 for this observer at production
    budget) with no hand-coded features. Use `make_logprice()` when you want
    the analytic reference to compare a learned representation against.
    """

    def tokens_from_traj(self, traj: torch.Tensor) -> torch.Tensor:
        if traj.dim() == 2:
            traj = traj.unsqueeze(-1)
        return super().tokens_from_traj(traj.clamp_min(CLAMP).log())


class GeometricBrownianMotion(SDEProblem):
    """dS = mu S dt + sigma S dW with S0 = 1, recover (mu, sigma).

    Tokens are RAW price increments; the floating multiplicative scale of S is
    handled by the default universal embedding (embed="auto" -> "wdiff"), not
    by a hand-crafted observer transform. Measured (K=96, vs the exact
    closed-form posterior): raw tokens + linear embed biased sigma by +0.483
    posterior-sd (SBC p=0.002 at production budget); raw tokens + wdiff is
    -0.010 sd, SBC p=0.876.
    """

    state_dim = 1

    def __init__(self, n_paths: int = 1):
        self.prior = BoxUniform(
            low=[-0.20, 0.10],
            high=[0.40, 0.60],
            names=["mu", "sigma"],
        )
        # Two observation channels on a 5-year, dt=0.01 grid:
        #   fast (every=1): dense increments -> quadratic variation -> sigma
        #   slow (every=10): coarse long-horizon view -> the (weak) drift mu
        self.observer = PathObserver(
            dt_sim=DT_SIM, n_steps=N_STEPS,
            channels=[
                Channel(every=1, count=60, label="fast"),
                Channel(every=10, count=40, label="slow"),
            ],
            n_paths=n_paths,
            obs_dims=(0,),
        )

    def drift(self, x, m):
        mu = m[:, 0:1]
        return mu * x.clamp_min(CLAMP)              # [B, 1]

    def diffusion(self, x, m):
        sigma = m[:, 1:2]
        return sigma * x.clamp_min(CLAMP)           # [B, 1] multiplicative noise

    def x0_sampler(self, m, generator=None):
        return torch.full((m.shape[0], self.state_dim), S0)


def make() -> GeometricBrownianMotion:
    """Fresh GBM problem instance."""
    return GeometricBrownianMotion()


SOTA_NAME = "exact MLE"


def sota(tokens, traj, prob) -> np.ndarray:
    """Closed-form maximum-likelihood estimator for GBM (textbook, near-optimal).

    Log-returns r_i = log(S_{i+1}/S_i) on the fine grid are i.i.d. Normal with
        mean = (mu - sigma^2/2) * dt,   variance = sigma^2 * dt.
    Hence the MLE is
        sigma_hat^2 = var(r) / dt,
        mu_hat      = mean(r) / dt + sigma_hat^2 / 2.

    Uses the raw fine path `traj` ([n_steps+1, 1] numpy), not the sparse tokens.
    Returns np.array([mu_hat, sigma_hat]) aligned to prob.prior.names.
    """
    dt = prob.observer.dt_sim
    s = np.asarray(traj, dtype=np.float64).reshape(-1)
    s = np.clip(s, CLAMP, None)                     # guard log of a clamped step
    r = np.diff(np.log(s))                          # log-returns, length n_steps
    var_r = r.var(ddof=1) if r.size > 1 else 0.0
    sigma2_hat = var_r / dt
    sigma_hat = float(np.sqrt(max(sigma2_hat, 0.0)))
    mu_hat = float(r.mean() / dt + sigma2_hat / 2.0)
    return np.array([mu_hat, sigma_hat])
