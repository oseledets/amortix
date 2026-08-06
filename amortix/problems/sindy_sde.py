"""Nonparametric drift discovery for a 1D SDE -- the SINDy-for-SDE bridge.

This is amortix's flagship differentiator: instead of recovering a handful of
named physical rates, we recover an *unknown drift function* expanded in a
polynomial library, exactly as SINDy does -- but with a full, calibrated
posterior over every library coefficient.

    dX = (c0 + c1 X + c2 X^2 + c3 X^3) dt + sigma dW

The unknown is m = (c0, c1, c2, c3, sigma): the four coefficients of the drift
in the basis {1, X, X^2, X^3} plus the noise level. Treating drift discovery as
parameter recovery in a fixed library is the bridge between data-driven model
discovery (SINDy / Kramers-Moyal) and amortized Bayesian inference.

To keep the SDE confining/stable the prior pins the leading cubic coefficient
negative (c3 < 0, a double-well-style restoring force) and c1 <= 0, so paths
explore but do not run away. The noise is large enough that X visits a wide
range, which is what makes the higher-order coefficients (c2, c3) identifiable
at all -- a coefficient of X^3 can only be seen where |X| is large.

Classical comparison: the first/second Kramers-Moyal coefficients estimated from
the fine path and fit in the same basis by ordinary least squares (= SINDy for
SDEs). The amortized posterior targets the same coefficients but additionally
reports per-coefficient uncertainty.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..sde import SDEProblem, PathObserver, Channel


class PolynomialDriftSDE(SDEProblem):
    state_dim = 1

    def __init__(self, n_paths: int = 1):
        # Confining cubic: c3 < 0 keeps the drift restoring at large |X| so the
        # Euler-Maruyama integrator stays bounded; c1 <= 0 adds linear damping.
        self.prior = BoxUniform(
            low=[-0.5, -2.0, -0.5, -1.0, 0.3],
            high=[0.5, 0.0, 0.5, -0.1, 0.8],
            names=["c0", "c1", "c2", "c3", "sigma"],
        )
        # dt=0.01, horizon 10 (1000 steps). Two observation views:
        #   fast (every=1)   -> per-step quadratic variation -> sigma & local drift
        #   slow (every=20)  -> the drift *shape* across the explored range of X,
        #                       which is what pins the higher-order coefficients.
        self.observer = PathObserver(
            dt_sim=0.01, n_steps=1000,
            channels=[
                Channel(every=1, count=80, label="fast"),
                Channel(every=20, count=45, label="slow"),
            ],
            n_paths=n_paths,
            obs_dims=(0,),
        )

    def drift(self, x, m):
        c0, c1, c2, c3 = m[:, 0:1], m[:, 1:2], m[:, 2:3], m[:, 3:4]
        x = x.clamp(-1e3, 1e3)  # defensive: keep the polynomial finite
        return c0 + c1 * x + c2 * x * x + c3 * x * x * x   # [B, 1]

    def diffusion(self, x, m):
        return m[:, 4:5].expand_as(x)                      # constant additive sigma

    def x0_sampler(self, m, generator=None):
        return torch.zeros(m.shape[0], self.state_dim)     # X0 = 0


def make() -> PolynomialDriftSDE:
    """Fresh problem instance."""
    return PolynomialDriftSDE()


SOTA_NAME = "SINDy / Kramers-Moyal"


def sota(tokens, traj, prob) -> np.ndarray:
    """Classical data-driven drift/diffusion estimator (SINDy for SDEs).

    From the fine path, estimate the drift via the first Kramers-Moyal
    coefficient D1(x) = E[dX | X=x] / dt and fit it in the polynomial basis
    {1, X, X^2, X^3} by ordinary least squares: regress dX/dt on the design
    matrix [1, X, X^2, X^3] -> (c0, c1, c2, c3). The diffusion is recovered from
    the second Kramers-Moyal coefficient: sigma^2 = E[dX^2] / dt.

    Args:
        tokens: numpy [T, F] for one instance (unused; we use the raw path).
        traj:   numpy [n_steps+1, 1] fine path.
        prob:   the PolynomialDriftSDE instance (for dt and prior bounds).

    Returns:
        np.ndarray [5] = (c0, c1, c2, c3, sigma), aligned to prob.prior.names.
    """
    dt = prob.observer.dt_sim
    x = np.asarray(traj, dtype=np.float64).reshape(-1)     # [n_steps+1]
    x_cur = x[:-1]
    dx = x[1:] - x[:-1]

    # --- drift: OLS of dX/dt on the polynomial library --------------------
    y = dx / dt
    design = np.stack([np.ones_like(x_cur), x_cur, x_cur ** 2, x_cur ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)      # (c0, c1, c2, c3)

    # --- diffusion: second Kramers-Moyal coefficient ----------------------
    sigma2 = np.mean(dx * dx) / dt
    sigma = float(np.sqrt(max(sigma2, 0.0)))

    out = np.array([coef[0], coef[1], coef[2], coef[3], sigma], dtype=np.float64)

    # clip into the prior box so the baseline stays a sane competitor
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    return np.clip(out, lo, hi)
