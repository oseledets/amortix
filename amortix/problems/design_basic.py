"""Random-design versions of the two calibration workhorses, GBM and OU.

These exist so that every claim of the paper can be made in the package's
default mode -- one network answering p(m | any K observation points) --
with exact or exact-likelihood references available *per design*:

* GBM: the conjugate normal--inverse-chi^2 posterior conditions on log-price
  increments over arbitrary gaps (``gbm_exact_from_points``);
* OU: the per-gap Euler--Maruyama transition density is Gaussian for any
  gap pattern, so adaptive Metropolis gives a reference for any design.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..prior import BoxUniform
from ..designs import DesignObserver, DesignProblem


class GBMDesign(DesignProblem):
    """dS = mu S dt + sigma S dW, S0 = 1, observed at arbitrary times."""

    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(low=[-0.20, 0.10], high=[0.40, 0.60],
                                names=["mu", "sigma"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        dt = self.observer.dt_sim
        n = self.observer.n_steps
        mu, sigma = m[:, 0], m[:, 1]
        logS = torch.zeros(B)
        out = torch.zeros(B, n + 1, 1)
        out[:, 0, 0] = 1.0
        sq = math.sqrt(dt)
        for i in range(n):
            z = torch.randn(B, generator=generator)
            logS = logS + (mu - 0.5 * sigma ** 2) * dt + sigma * sq * z
            out[:, i + 1, 0] = torch.exp(logS)
        return out


class OUDesign(DesignProblem):
    """dX = -theta X dt + sigma dW, stationary start, arbitrary times."""

    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(low=[0.3, 0.2], high=[3.0, 1.5],
                                names=["theta", "sigma"])
        self.observer = DesignObserver(dt_sim=0.02, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        dt = self.observer.dt_sim
        n = self.observer.n_steps
        theta, sigma = m[:, 0], m[:, 1]
        x = (sigma / torch.sqrt(2.0 * theta)) * torch.randn(B, generator=generator)
        out = torch.zeros(B, n + 1, 1)
        out[:, 0, 0] = x
        sq = math.sqrt(dt)
        for i in range(n):
            z = torch.randn(B, generator=generator)
            x = x + (-theta * x) * dt + sigma * sq * z
            out[:, i + 1, 0] = x
        return out


def gbm_exact_from_points(prob, traj_i, tidx, n_samples=2000, seed=0,
                          pool_factor=8):
    """Exact per-design GBM posterior: conjugate in (b, sigma^2) on the
    log-increments over the observed gaps, importance-corrected to the box
    prior. ``tidx`` are fine-grid indices (the S0 = 1 anchor at index 0 is
    prepended automatically)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(traj_i, dtype=np.float64).reshape(-1)
    idx = np.unique(np.concatenate([[0], np.asarray(tidx, dtype=np.int64)]))
    vals = x[idx]
    tau = np.diff(idx).astype(np.float64) * float(prob.observer.dt_sim)
    r = np.diff(np.log(np.maximum(vals, 1e-12)))
    n = r.size
    T = float(tau.sum())
    R1 = float(r.sum())
    SSw = float((r ** 2 / tau).sum()) - R1 ** 2 / T
    low = prob.prior.low.numpy().astype(np.float64)
    high = prob.prior.high.numpy().astype(np.float64)

    # The conjugate draw is exact only up to the importance correction onto the
    # box prior, and that correction is not free: at dense designs the
    # posterior is narrow, and when the truth sits near a corner of the box
    # most candidates land outside it. One battery set had 45 usable draws out
    # of 32,000, so its "exact" sample was fifty distinct values with repeats
    # and two independent draws disagreed by 0.4 posterior sd. The pool is
    # therefore grown until the effective sample size is adequate, and the
    # caller is told when it cannot be.
    draws_all, w_all = [], []
    npool = n_samples * pool_factor
    for _ in range(6):
        chi = rng.chisquare(max(n - 1, 1), size=npool)
        v = SSw / np.maximum(chi, 1e-300)
        b = R1 / T + np.sqrt(v / T) * rng.standard_normal(npool)
        sigma = np.sqrt(v)
        d = np.stack([b + 0.5 * v, sigma], axis=1)
        draws_all.append(d)
        w_all.append(sigma * np.all((d >= low) & (d <= high), axis=1))
        w = np.concatenate(w_all)
        if w.sum() > 0 and w.sum() ** 2 / (w ** 2).sum() >= 10 * n_samples:
            break
        npool *= 4
    draws = np.concatenate(draws_all)
    if w.sum() <= 0:
        raise RuntimeError("no conjugate draws inside the prior box")
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    if ess < n_samples:
        raise RuntimeError(
            f"GBM conjugate reference is degenerate here: effective sample "
            f"size {ess:.0f} for {n_samples} requested draws (the posterior "
            f"lies mostly outside the prior box). Exclude this observation "
            f"set rather than trusting the reference.")
    pick = rng.choice(len(draws), size=n_samples, replace=True, p=w / w.sum())
    return draws[pick]


def ou_logpost_factory(prob, traj_i, tidx):
    """Exact log-posterior of the generative (Euler) OU chain on an arbitrary
    design, including the stationary density of the informative start."""
    from ..mcmc import log_likelihood_ou

    x = np.asarray(traj_i, dtype=np.float64).reshape(-1)
    idx = np.unique(np.concatenate([[0], np.asarray(tidx, dtype=np.int64)]))
    s = x[idx]
    dt = float(prob.observer.dt_sim)
    gaps = np.diff(idx).astype(np.float64) * dt

    def log_post(p):
        th, sg = float(p[0]), float(p[1])
        if th <= 0 or sg <= 0:
            return -np.inf
        base = log_likelihood_ou(s, p, gaps, scheme="euler", dt_fine=dt)
        var0 = sg ** 2 / (2.0 * th)
        return base - 0.5 * (s[0] ** 2 / var0 + math.log(2 * math.pi * var0))

    return log_post

class CIRDesign(DesignProblem):
    """dX = a(b - X)dt + sigma sqrt(X) dW, stationary start, arbitrary times.

    Simulation is EXACT, not Euler: each fine step draws the noncentral
    chi-square transition through its Poisson--Gamma representation, and the
    start is the stationary Gamma(2ab/sigma^2, sigma^2/2a). That choice is what
    makes an exact-likelihood reference possible at all -- the reference
    describes the same chain the simulator generates, gap by gap, with no
    discretization slack between them (the OU reference achieves this by
    composing its Euler chain in closed form; CIR's Euler chain has no such
    form, so the simulator is lifted to the exact transition instead).

    Randomness runs through a numpy generator seeded from the torch one:
    torch has no generator-threaded Poisson/Gamma sampling, and the CPU numpy
    path keeps runs bit-reproducible across devices, like every other
    simulator in the package.
    """

    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(low=[0.3, 0.3, 0.10], high=[3.0, 1.5, 0.50],
                                names=["a", "b", "sigma"])
        self.observer = DesignObserver(dt_sim=0.02, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        seed = int(torch.randint(2 ** 31 - 1, (1,), generator=generator))
        rng = np.random.default_rng(seed)
        a = m[:, 0].double().numpy(); b = m[:, 1].double().numpy()
        sg = m[:, 2].double().numpy()
        dt = self.observer.dt_sim; n = self.observer.n_steps
        B = len(a)
        edt = np.exp(-a * dt)
        c = sg * sg * (1.0 - edt) / (4.0 * a)
        df = 4.0 * a * b / (sg * sg)
        x = rng.gamma(shape=df / 2.0, scale=sg * sg / (2.0 * a))   # stationary
        out = np.empty((B, n + 1), dtype=np.float64)
        out[:, 0] = x
        for i in range(n):
            lam = x * edt / c
            k = rng.poisson(lam / 2.0)
            x = c * rng.gamma(shape=df / 2.0 + k, scale=2.0)       # ncx2 exact
            out[:, i + 1] = x
        return torch.from_numpy(out).float()[:, :, None]


def cir_logpost_factory(prob, traj_i, tidx):
    """Exact log-posterior of the CIR chain on an arbitrary design: noncentral
    chi-square gap transitions plus the stationary Gamma density of the start."""
    from scipy.stats import ncx2, gamma as sp_gamma

    x = np.asarray(traj_i, dtype=np.float64).reshape(-1)
    idx = np.unique(np.concatenate([[0], np.asarray(tidx, dtype=np.int64)]))
    s = x[idx]
    dt = float(prob.observer.dt_sim)
    gaps = np.diff(idx).astype(np.float64) * dt

    def log_post(p):
        a, b, sg = float(p[0]), float(p[1]), float(p[2])
        if a <= 0 or b <= 0 or sg <= 0:
            return -np.inf
        df = 4.0 * a * b / (sg * sg)
        edt = np.exp(-a * gaps)
        c = sg * sg * (1.0 - edt) / (4.0 * a)
        lam = s[:-1] * edt / c
        ll = float(np.sum(ncx2.logpdf(s[1:] / c, df, lam) - np.log(c)))
        ll += float(sp_gamma.logpdf(s[0], a=df / 2.0,
                                    scale=sg * sg / (2.0 * a)))
        return ll if np.isfinite(ll) else -np.inf

    return log_post

class LinGaussDesign(DesignProblem):
    """The linear-Gaussian check as a design problem: y = A m + eps, and a
    design is a SUBSET of the six observation components. The posterior for
    any subset is the conjugate Gaussian of the selected rows of A restricted
    to the prior box, so references are exact draws -- the only system whose
    evaluation set involves no MCMC at all.
    """

    obs_noise = 0.5          # NOISE of the fixed-design instrument, unchanged

    def __init__(self):
        from .linear_gaussian import A, D_PARAM, N_OBS
        self.A = A
        self.prior = BoxUniform(low=[-3.0] * D_PARAM, high=[3.0] * D_PARAM,
                                names=[f"m{i+1}" for i in range(D_PARAM)])
        # n_steps = number of observation components; "time" is component id
        self.observer = DesignObserver(dt_sim=1.0, n_steps=N_OBS, k_max=N_OBS)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        # noiseless A m as a 6-point "path"; observation noise enters at
        # tokenization, like every other design problem
        clean = m @ self.A.T                          # [B, 6]
        B = m.shape[0]
        out = torch.zeros(B, self.observer.n_steps + 1, 1)
        out[:, 1:, 0] = clean
        return out

    def exact_from_points(self, tidx, y, n_samples=4000, seed=0,
                          n_sweeps=400):
        """Exact posterior draws for the observed subset.

        The subset posterior is a box-truncated Gaussian, and for sparse
        subsets it is underdetermined (rank of the selected rows < d), so
        rejection from the unconstrained Gaussian starves. Sampling instead
        runs ``n_samples`` INDEPENDENT Gibbs chains in parallel on the
        canonical form (precision P = A_S^T A_S / s^2, shift b = A_S^T y /
        s^2): each coordinate conditional is a one-dimensional truncated
        normal, exact and vectorized across all chains, and no matrix is ever
        inverted -- the degenerate directions are simply flat conditionals
        clipped by the box. Log-concave target, so the sweeps mix fast;
        chains start uniform in the box and only the final state is kept,
        giving independent draws.
        """
        from scipy.stats import truncnorm

        rng = np.random.default_rng(seed)
        As = self.A[tidx - 1].double().numpy()
        s2 = float(self.obs_noise) ** 2
        P = As.T @ As / s2
        b = As.T @ np.asarray(y, dtype=np.float64) / s2
        lo = self.prior.low.double().numpy(); hi = self.prior.high.double().numpy()
        d = len(b)
        m = rng.uniform(lo, hi, size=(n_samples, d))
        sd = 1.0 / np.sqrt(np.diag(P))
        for _ in range(n_sweeps):
            for i in range(d):
                mean = (b[i] - m @ P[:, i] + m[:, i] * P[i, i]) / P[i, i]
                a_, b_ = (lo[i] - mean) / sd[i], (hi[i] - mean) / sd[i]
                m[:, i] = truncnorm.rvs(a_, b_, loc=mean, scale=sd[i],
                                        random_state=rng)
        return torch.from_numpy(m).float()

class DoubleWellDesign(DesignProblem):
    """dX = (theta1 X - theta2 X^3) dt + sigma dW, arbitrary observation times.

    Reference likelihood: exact density transport of the generative Euler
    chain (amortix.transfer) -- a banded transition matrix per step, exact
    conditioning at observed points. No latent variables, no bridges.
    """

    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(low=[0.5, 0.5, 0.4], high=[3.0, 3.0, 1.2],
                                names=["theta1", "theta2", "sigma"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=1000, k_max=128)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        t1, t2, sg = m[:, 0], m[:, 1], m[:, 2]
        dt = self.observer.dt_sim
        x = torch.ones(B)
        out = torch.zeros(B, self.observer.n_steps + 1, 1)
        out[:, 0, 0] = x
        sq = dt ** 0.5
        for i in range(self.observer.n_steps):
            x = x + (t1 * x - t2 * x ** 3) * dt                 + sg * sq * torch.randn(B, generator=generator)
            out[:, i + 1, 0] = x
        return out


def dw_logpost_factory(prob, traj_i, tidx, n_part=256):
    """Unbiased bridge likelihood for the double well (amortix.bridgelik).

    Grid-free, for the reason given in poly_logpost_factory: a state-space
    grid makes the reference's accuracy a tuning parameter, and a reference
    the model can out-resolve is worse than useless -- it turns improvement
    into apparent degradation.
    """
    from ..bridgelik import gap_loglik
    x = np.asarray(traj_i, dtype=np.float64).reshape(-1, 1)
    idx = np.unique(np.concatenate([[0], np.asarray(tidx, dtype=np.int64)]))
    gaps = np.diff(idx)
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)

    def lp(m):
        if np.any(m <= lo) or np.any(m >= hi):
            return -np.inf
        t1, t2, sg = (float(v) for v in m)
        rng = np.random.default_rng(0)
        drift = lambda X: (t1 * X[:, 0] - t2 * X[:, 0] ** 3)[:, None]
        diff = lambda X: np.full_like(X, sg)
        ll = 0.0
        for k, g in enumerate(gaps):
            v = gap_loglik(x[idx[k]], x[idx[k + 1]], int(g), drift, diff,
                           prob.observer.dt_sim, n_part, rng)
            if not np.isfinite(v):
                return -np.inf
            ll += v
        return ll

    return lp


class PolyDriftDesign(DesignProblem):
    """dX = (c0 + c1 X + c2 X^2 + c3 X^3) dt + sigma dW, observed anywhere.

    Same transport reference as the double well (amortix.transfer) -- the
    drift is a different lambda, nothing else changes.
    """

    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(low=[-0.5, -2.0, -0.5, -1.0, 0.3],
                                high=[0.5, 0.0, 0.5, -0.1, 0.8],
                                names=["c0", "c1", "c2", "c3", "sigma"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=1000, k_max=128)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        c0, c1, c2, c3, sg = (m[:, i] for i in range(5))
        dt = self.observer.dt_sim
        x = torch.zeros(B)
        out = torch.zeros(B, self.observer.n_steps + 1, 1)
        sq = dt ** 0.5
        for i in range(self.observer.n_steps):
            f = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
            x = (x + f * dt + sg * sq * torch.randn(B, generator=generator)).clamp(-8, 8)
            out[:, i + 1, 0] = x
        return out


def poly_logpost_factory(prob, traj_i, tidx, n_part=256):
    """Unbiased bridge likelihood for the polynomial-drift SDE.

    Uses the same grid-free estimator as every other diffusion in the suite
    (amortix.bridgelik). The earlier density-transport version is retired: its
    accuracy is set by a state-space grid, and on this system the affordable
    grid put only ~150 cells across the region the state actually visits,
    which biases the reference the model is scored against -- the failure mode
    where a more accurate model reads as worse because it has moved away from
    a mis-stated target.
    """
    from ..bridgelik import path_loglik_factory
    x = np.asarray(traj_i, dtype=np.float64).reshape(-1, 1)
    idx = np.unique(np.concatenate([[0], np.asarray(tidx, dtype=np.int64)]))

    def drift_of(m):
        c0, c1, c2, c3, sg = (float(v) for v in m)
        def f(X):
            g = X[:, 0]
            return (c0 + c1 * g + c2 * g ** 2 + c3 * g ** 3)[:, None]
        return f

    def make(m):
        sg = float(m[-1])
        return lambda X: np.full_like(X, sg)

    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    from ..bridgelik import gap_loglik
    gaps = np.diff(idx)

    def lp(m):
        if np.any(m <= lo) or np.any(m >= hi):
            return -np.inf
        rng = np.random.default_rng(0)
        drift = drift_of(m); diff = make(m)
        ll = 0.0
        for k, g in enumerate(gaps):
            v = gap_loglik(x[idx[k]], x[idx[k + 1]], int(g), drift, diff,
                           prob.observer.dt_sim, n_part, rng)
            if not np.isfinite(v):
                return -np.inf
            ll += v
        return ll

    return lp


class LotkaVolterraDesign(DesignProblem):
    """Stochastic Lotka--Volterra, both species observed at arbitrary times.

    Multiplicative noise on each species. The reference likelihood is the
    unbiased bridge estimate of amortix.bridgelik inside adaptive tempered
    SMC (amortix.smc); the density-transport reference this replaced made
    grid resolution a tuning parameter of the reference itself.
    """

    markov_observed = True
    S1 = S2 = 0.05

    def __init__(self):
        self.prior = BoxUniform(low=[0.8, 0.4, 0.4, 0.8],
                                high=[1.5, 1.2, 1.2, 1.5],
                                names=["alpha", "beta", "delta", "gamma"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=600, k_max=128,
                                       n_channels=2)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        a, b, d_, g = (m[:, i] for i in range(4))
        dt = self.observer.dt_sim
        x = torch.ones(B); y = torch.ones(B)
        out = torch.zeros(B, self.observer.n_steps + 1, 2)
        out[:, 0, 0] = x; out[:, 0, 1] = y
        sq = dt ** 0.5
        for i in range(self.observer.n_steps):
            xc = x.clamp_min(1e-6); yc = y.clamp_min(1e-6)
            x = (x + (a * xc - b * xc * yc) * dt
                 + self.S1 * xc * sq * torch.randn(B, generator=generator)).clamp_min(1e-6)
            y = (y + (d_ * xc * yc - g * yc) * dt
                 + self.S2 * yc * sq * torch.randn(B, generator=generator)).clamp_min(1e-6)
            out[:, i + 1, 0] = x; out[:, i + 1, 1] = y
        return out


def lv_logpost_factory(prob, traj_i, tidx, cidx=None, n_part=512):
    """Unbiased bridge likelihood for Lotka--Volterra, conditioning on exactly
    what the network is shown.

    Each design point observes ONE species, so the reference must pin one
    coordinate and leave the other latent -- the partially observed case of
    amortix.bridgelik. A first version conditioned on both species at every
    point; its posterior was four times narrower than the network could
    possibly match, and the resulting score (FID 57--152 against a floor of
    0.004) measured the mismatch, not the model.
    """
    from ..bridgelik import gap_loglik_partial
    xy = np.asarray(traj_i, dtype=np.float64)
    t = np.asarray(tidx, dtype=np.int64)
    c = (np.zeros_like(t) if cidx is None else np.asarray(cidx, dtype=np.int64))
    order = np.argsort(t)
    t, c = t[order], c[order]
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    S1, S2 = prob.S1, prob.S2
    y_obs = xy[t, c]
    dt = prob.observer.dt_sim

    def lp(m):
        if np.any(m <= lo) or np.any(m >= hi):
            return -np.inf
        a, b, d_, g = (float(v) for v in m)

        def drift(X):
            x = np.maximum(X[:, 0], 1e-6); y = np.maximum(X[:, 1], 1e-6)
            return np.stack([a * x - b * x * y, d_ * x * y - g * y], 1)

        def diff(X):
            return np.stack([S1 * np.maximum(X[:, 0], 1e-6),
                             S2 * np.maximum(X[:, 1], 1e-6)], 1)

        return gap_loglik_partial(xy[0], t, c, y_obs, drift, diff, dt,
                                  n_part, np.random.default_rng(0))

    return lp
