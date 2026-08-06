"""SEIRD epidemic model -- the package's port of the paper's disease-dynamics case.

Compartments S, E, I, R, D (population normalized to 1) with a two-phase
transmission rate (an intervention switches beta1 -> beta2 at a known day):

    dS = -beta(t) S I
    dE =  beta(t) S I - alpha E
    dI =  alpha E - (gamma_r + gamma_d) I
    dR =  gamma_r I
    dD =  gamma_d I

Recover m = (beta1, beta2, alpha, gamma_r, gamma_d) from noisy observations of
the infected I(t) and deceased D(t) curves. Mirrors arXiv:2503.01375's SEIR
setup (multi-output epidemic time series, ~5-6 unknown rates); not bit-exact.
"""
from __future__ import annotations

import numpy as np
import torch

from ..prior import BoxUniform
from ..ode import ODEProblem, TimeSeriesObserver


class SEIRD(ODEProblem):
    state_dim = 5  # S, E, I, R, D

    def __init__(self, t_switch: float = 20.0):
        self.t_switch = float(t_switch)
        self.prior = BoxUniform(
            low=[0.20, 0.05, 0.20, 0.05, 0.005],
            high=[0.60, 0.30, 0.50, 0.20, 0.050],
            names=["beta1", "beta2", "alpha", "gamma_r", "gamma_d"],
        )
        # 60 "days" at dt=0.1; observe I and D at 20 evenly spaced days
        n_steps = 600
        obs_steps = torch.linspace(20, n_steps, 20).long().tolist()
        self.observer = TimeSeriesObserver(
            dt_sim=0.1, n_steps=n_steps, obs_steps=obs_steps,
            obs_indices=[2, 4],          # I, D
            noise_std=0.004,
        )

    def x0(self, m):
        B = m.shape[0]
        x = torch.zeros(B, self.state_dim)
        x[:, 0] = 0.999   # S
        x[:, 2] = 0.001   # I
        return x

    def rhs(self, x, m, t):
        S, E, I, R, D = x.unbind(dim=1)
        beta = m[:, 0] if t < self.t_switch else m[:, 1]
        alpha, gr, gd = m[:, 2], m[:, 3], m[:, 4]
        dS = -beta * S * I
        dE = beta * S * I - alpha * E
        dI = alpha * E - (gr + gd) * I
        dR = gr * I
        dD = gd * I
        return torch.stack([dS, dE, dI, dR, dD], dim=1)


# --- gallery contract -----------------------------------------------------
SOTA_NAME = "nonlinear LS"


def make():
    return SEIRD()


def sota(tokens, traj, prob):
    from ..baselines import seird_nls
    tk = np.asarray(tokens)
    n_obs = len(prob.observer.obs_steps)
    I_obs = tk[:n_obs, 1]
    D_obs = tk[n_obs:2 * n_obs, 1]
    d = seird_nls(I_obs, D_obs, prob.observer.obs_steps.numpy(), prob.observer.dt_sim,
                  prob.observer.n_steps, prob.t_switch,
                  prob.prior.low.numpy(), prob.prior.high.numpy())
    return np.array([d[k] for k in prob.prior.names])
