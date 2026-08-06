"""Double-well bistable SDE -- the canonical metastable-system recovery case.

    dX = (theta1 X - theta2 X^3) dt + sigma dW

The drift is -V'(X) for the double-well potential V(X) = -theta1 X^2/2 + theta2 X^4/4,
with two stable wells at X = +/- sqrt(theta1/theta2) and an unstable barrier at 0.
With enough noise and a long horizon the path *hops* between wells (a Kramers
escape problem); those rare crossings are what pin down the full cubic drift
shape -- if a trajectory only ever explores one well, the curvature theta2 (and
hence the well location) is only weakly identified, which is exactly the regime
where a calibrated amortized posterior earns its keep.

Recover m = (theta1, theta2, sigma) from observed paths. There is no closed-form
transition density, so the classical reference is the Kramers-Moyal / SINDy-style
estimator: regress the empirical drift onto the [X, X^3] basis and read sigma off
the quadratic variation (see `sota`).
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..sde import SDEProblem, PathObserver, Channel


class DoubleWell(SDEProblem):
    state_dim = 1

    def __init__(self, n_paths: int = 1):
        self.prior = BoxUniform(
            low=[0.5, 0.5, 0.4],
            high=[3.0, 3.0, 1.2],
            names=["theta1", "theta2", "sigma"],
        )
        # Long horizon (10 time units, dt=0.01) with strong noise so trajectories
        # actually cross the barrier between wells -- needed to see the cubic term.
        #   fast channel (every=1)  -> per-step quadratic variation -> sigma
        #   slow channel (every=25) -> coarse view of well structure -> drift
        self.observer = PathObserver(
            dt_sim=0.01, n_steps=1000,
            channels=[
                Channel(every=1, count=80, label="fast"),
                Channel(every=25, count=36, label="slow"),
            ],
            n_paths=n_paths,
            obs_dims=(0,),
        )

    def drift(self, x, m):
        theta1, theta2 = m[:, 0:1], m[:, 1:2]
        return theta1 * x - theta2 * x ** 3        # [B, 1]

    def diffusion(self, x, m):
        return m[:, 2:3].expand_as(x)              # constant additive noise [B, 1]

    def x0_sampler(self, m, generator=None):
        # start on the barrier so each path can fall into either well
        return torch.zeros(m.shape[0], self.state_dim)


def make() -> DoubleWell:
    """Fresh problem instance."""
    return DoubleWell()


SOTA_NAME = "Kramers-Moyal LS"


def sota(tokens, traj, prob) -> np.ndarray:
    """Classical Kramers-Moyal / SINDy drift-diffusion estimator.

    From the fine path estimate the drift by OLS regression of the empirical
    increment rate dX/dt on the basis [X, X^3]:

        dX/dt ~= theta1 * X - theta2 * X^3

    so the regression coefficients are (theta1, -theta2). sigma comes from the
    quadratic variation: sigma_hat^2 = mean(dX^2)/dt. Returns
    [theta1_hat, theta2_hat, sigma_hat] aligned to prob.prior.names.
    """
    x = np.asarray(traj, dtype=np.float64).reshape(-1)
    dt = float(prob.observer.dt_sim)

    x0 = x[:-1]
    dx = x[1:] - x[:-1]

    # drift: OLS of dx/dt on [x, x^3]
    A = np.stack([x0, x0 ** 3], axis=1)             # [n, 2]
    y = dx / dt                                      # empirical drift rate
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    theta1_hat = coef[0]
    theta2_hat = -coef[1]

    # diffusion: quadratic variation
    sigma2_hat = np.mean(dx ** 2) / dt
    sigma_hat = np.sqrt(max(sigma2_hat, 1e-12))

    return np.array([theta1_hat, theta2_hat, sigma_hat], dtype=np.float64)
