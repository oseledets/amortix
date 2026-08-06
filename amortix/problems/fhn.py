"""FitzHugh-Nagumo neuron model: a 2D excitable ODE from computational neuroscience.

    dv/dt = v - v^3/3 - w + I        (fast membrane potential)
    dw/dt = eps * (v + a - b * w)    (slow recovery variable)

This is the textbook reduction of the Hodgkin-Huxley dynamics: a fast cubic
"activator" v (the membrane potential) coupled to a slow linear "inhibitor" w.
For parameters in the chosen prior box the system sits in its relaxation /
oscillatory regime, producing a train of spikes followed by slow recovery over
the horizon -- a rich, non-stationary signal.

Recover m = (a, b, eps, I) from a noisy, *partially observed* trajectory: only
the membrane potential v(t) is measured (obs_indices=[0]), as in a real
electrophysiology recording. The recovery variable w is hidden, which makes the
slow-timescale parameters (eps, and the I/a offset that sets the resting level)
only weakly identifiable from v alone -- a realistic, interesting test case.

SOTA baseline: nonlinear least squares -- forward-simulate the FHN ODE and fit
its v(t) to the noisy observed potential (scipy.optimize.least_squares over the
prior box). This is the standard classical estimator for mechanistic ODE models.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..ode import ODEProblem, TimeSeriesObserver


class FitzHughNagumo(ODEProblem):
    state_dim = 2  # v (membrane potential), w (recovery)

    def __init__(self):
        self.prior = BoxUniform(
            low=[0.5, 0.5, 0.05, 0.0],
            high=[0.9, 0.9, 0.30, 0.5],
            names=["a", "b", "eps", "I"],
        )
        # horizon 40 (= dt_sim 0.05 * 800 steps): several spike/relaxation cycles.
        # observe ONLY v at ~25 evenly spaced times.
        n_steps = 800
        obs_steps = torch.linspace(0, n_steps, 25).round().long().tolist()
        self.observer = TimeSeriesObserver(
            dt_sim=0.05, n_steps=n_steps, obs_steps=obs_steps,
            obs_indices=[0],             # measure v only
            noise_std=0.05,
        )

    def x0(self, m):
        B = m.shape[0]
        x = torch.empty(B, self.state_dim, dtype=m.dtype)
        x[:, 0] = -1.0   # v
        x[:, 1] = 1.0    # w
        return x

    def rhs(self, x, m, t):
        v, w = x[:, 0], x[:, 1]
        a, b, eps, I = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
        dv = v - v.pow(3) / 3.0 - w + I
        dw = eps * (v + a - b * w)
        return torch.stack([dv, dw], dim=1)


def make() -> FitzHughNagumo:
    """Fresh FitzHugh-Nagumo problem instance."""
    return FitzHughNagumo()


SOTA_NAME = "nonlinear least squares"


def _fhn_solve_np(m, dt, n_steps):
    """Numpy RK4 forward solve of FHN from x0 = [-1, 1] (matches FitzHughNagumo)."""
    a, b, eps, I = m

    def rhs(x):
        v, w = x
        return np.array([
            v - v ** 3 / 3.0 - w + I,
            eps * (v + a - b * w),
        ])

    sol = np.empty((n_steps + 1, 2))
    x = np.array([-1.0, 1.0], dtype=np.float64)
    sol[0] = x
    for i in range(n_steps):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sol[i + 1] = x
    return sol


def sota(tokens, traj, prob) -> np.ndarray:
    """Nonlinear least-squares fit of (a, b, eps, I) to the noisy observed v(t).

    Forward-simulates the FHN ODE with a numpy RK4 solver (same equations / x0 /
    grid as the problem) and minimizes the residual between simulated v and the
    observed noisy v at the observation steps, over the prior box. This is the
    standard classical estimator for a mechanistic ODE model.
    """
    obs = prob.observer
    obs_steps = obs.obs_steps.numpy()
    dt, n_steps = obs.dt_sim, obs.n_steps
    low = prob.prior.low.numpy().astype(np.float64)
    high = prob.prior.high.numpy().astype(np.float64)

    # only v is observed -> all tokens are v; value column is index 1.
    v_obs = np.asarray(tokens[:, 1], dtype=np.float64)

    from scipy.optimize import least_squares

    def resid(m):
        sol = _fhn_solve_np(m, dt, n_steps)
        return sol[obs_steps, 0] - v_obs

    m0 = 0.5 * (low + high)
    res = least_squares(resid, m0, bounds=(low, high), method="trf", max_nfev=200)
    return np.asarray(res.x, dtype=np.float64)
