"""Classical baselines for SDE recovery.

For the OU process the discrete-time observations form an AR(1) process

    X_{k+1} = mu (1 - rho) + rho X_k + eps,   rho = exp(-theta dt),
    Var(eps) = sigma^2 / (2 theta) * (1 - rho^2),

whose exact conditional maximum-likelihood estimator is available in closed form
(linear regression of X_{k+1} on X_k plus residual variance). This is the
near-optimal classical reference the amortized posterior is validated against.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def _seird_solve_np(m, dt, n_steps, t_switch, x0):
    """Numpy RK4 forward solve of the SEIRD model (matches problems/seir.py)."""
    b1, b2, alpha, gr, gd = m

    def rhs(x, t):
        S, E, I, R, D = x
        beta = b1 if t < t_switch else b2
        return np.array([
            -beta * S * I,
            beta * S * I - alpha * E,
            alpha * E - (gr + gd) * I,
            gr * I,
            gd * I,
        ])

    sol = np.empty((n_steps + 1, 5))
    x = np.asarray(x0, dtype=np.float64)
    sol[0] = x
    for i in range(n_steps):
        t = i * dt
        k1 = rhs(x, t)
        k2 = rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sol[i + 1] = x
    return sol


def seird_nls(I_obs, D_obs, obs_steps, dt, n_steps, t_switch, low, high) -> dict:
    """Nonlinear least-squares fit of SEIRD rates -- the classical baseline
    (the same family the bioprocess paper, arXiv:2604.22496, compares CFM to)."""
    obs_steps = np.asarray(obs_steps)
    x0 = [0.999, 0.0, 0.001, 0.0, 0.0]
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)

    def resid(m):
        sol = _seird_solve_np(m, dt, n_steps, t_switch, x0)
        I = sol[obs_steps, 2]
        D = sol[obs_steps, 4]
        return np.concatenate([I - I_obs, D - D_obs])

    m0 = 0.5 * (low + high)
    res = least_squares(resid, m0, bounds=(low, high), method="trf", max_nfev=200)
    names = ["beta1", "beta2", "alpha", "gamma_r", "gamma_d"]
    return {n: float(v) for n, v in zip(names, res.x)}


def ou_mle(path: np.ndarray, dt: float) -> dict:
    """Closed-form conditional MLE of (theta, mu, sigma) from a uniform OU path."""
    x = np.asarray(path, dtype=np.float64)
    x0, x1 = x[:-1], x[1:]
    n = x0.size
    # OLS  x1 = a + b x0
    sx, sy = x0.mean(), x1.mean()
    sxx = ((x0 - sx) ** 2).mean()
    sxy = ((x0 - sx) * (x1 - sy)).mean()
    b = sxy / sxx
    a = sy - b * sx
    rho = float(np.clip(b, 1e-6, 1 - 1e-9))
    resid = x1 - (a + b * x0)
    s2 = (resid ** 2).sum() / n
    theta = -np.log(rho) / dt
    mu = a / (1.0 - rho)
    sigma2 = s2 * 2.0 * theta / (1.0 - rho ** 2)
    return {"theta": float(theta), "mu": float(mu), "sigma": float(np.sqrt(max(sigma2, 1e-12)))}
