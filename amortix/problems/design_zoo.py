"""Design-amortized problem zoo: p(m | any K observation points).

Every case here follows the DesignProblem protocol (amortix.designs): full
raw trajectories + fresh random designs every optimizer step. Calibration
status and the full experimental chronicle are in CALIBRATION.md; the
recommended configuration is simply the defaults --

    post = FlowPosterior(prob)          # embed/rope resolve by class rule
    post.fit(n_train=..., steps=..., retokenize=prob.make_retokenizer())

(embed="auto" picks the set-conditioned pair embedding for Markov-observed
cases and bare points otherwise; rope="auto" picks continuous time-RoPE for
any DesignProblem); budget per the price curves in CALIBRATION.md.

Cases:
  HestonDesign     hidden stochastic volatility, correlated noises,
                   price-only observations (5 parameters)
  MertonDesign     jump-diffusion in log-price (5 parameters); near-exact
                   Poisson-mixture likelihood -> merton_logpost_factory
  HenonHeilesDesign classical Hamiltonian with the Lubich-Oseledets-
                   Vandereycken potential; noisy q1 only (3 parameters)
  HodgkinHuxleyDesign  sbibm's flagship spiking neuron (4 parameters)
  PharmacoKineticsDesign  oral one-compartment Bateman curve, log-normal
                   assay noise -- irregular blood draws (3 parameters);
                   exact likelihood -> pk_logpost_factory
  FisherKPPDesign  reaction-diffusion PDE, 3 point sensors, random
                   time x sensor designs (2 parameters); exact likelihood
                   via the deterministic solve -> kpp_logpost_factory
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..designs import DesignObserver, DesignProblem
from ..prior import BoxUniform


# --------------------------------------------------------------- Heston
class HestonDesign(DesignProblem):
    def __init__(self):
        self.prior = BoxUniform(
            low=[-0.10, 0.5, 0.02, 0.10, -0.90],
            high=[0.30, 5.0, 0.30, 1.00, 0.00],
            names=["mu", "kap", "th", "xi", "rho"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        dt = self.observer.dt_sim
        n = self.observer.n_steps
        mu, kap, th, xi, rho = (m[:, j] for j in range(5))
        S = torch.ones(B)
        v = th.clone()
        out = torch.zeros(B, n + 1, 1)
        out[:, 0, 0] = S
        sq = math.sqrt(dt)
        for i in range(n):
            z1 = torch.randn(B, generator=generator)
            z2 = torch.randn(B, generator=generator)
            w2 = rho * z1 + torch.sqrt(1.0 - rho ** 2) * z2
            vp = v.clamp_min(0.0)
            S = (S * (1.0 + mu * dt + torch.sqrt(vp) * sq * z1)).clamp_min(1e-8)
            v = v + kap * (th - vp) * dt + xi * torch.sqrt(vp) * sq * w2
            out[:, i + 1, 0] = S
        return out


# --------------------------------------------------------------- Merton
class MertonDesign(DesignProblem):
    markov_observed = True

    def __init__(self):
        self.prior = BoxUniform(
            low=[-0.10, 0.10, 0.2, -0.30, 0.05],
            high=[0.30, 0.50, 3.0, 0.30, 0.40],
            names=["mu", "sig", "lamJ", "muJ", "sJ"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        dt = self.observer.dt_sim
        n = self.observer.n_steps
        mu, sig, lamJ, muJ, sJ = (m[:, j] for j in range(5))
        logS = torch.zeros(B)
        out = torch.zeros(B, n + 1, 1)
        out[:, 0, 0] = 1.0
        sq = math.sqrt(dt)
        for i in range(n):
            z = torch.randn(B, generator=generator)
            nj = torch.poisson(lamJ * dt, generator=generator)
            jump = muJ * nj + sJ * torch.sqrt(nj.clamp_min(0.0)) \
                * torch.randn(B, generator=generator)
            logS = logS + (mu - 0.5 * sig ** 2) * dt + sig * sq * z + jump
            out[:, i + 1, 0] = torch.exp(logS)
        return out


def merton_logpost_factory(r, tau, low, high, nmax=40):
    """Near-exact Merton log-likelihood over arbitrary gaps: Poisson mixture
    truncated at nmax jumps. r: log-returns [K], tau: gap durations [K]."""
    n = np.arange(nmax + 1)
    logfact = np.concatenate([[0.0],
                              np.cumsum(np.log(np.arange(1, nmax + 1)))])

    def log_post(p):
        mu, sig, lamJ, muJ, sJ = p
        if sig <= 0 or lamJ <= 0 or sJ <= 0:
            return -np.inf
        lt = lamJ * tau
        logpois = (n[None, :] * np.log(lt[:, None] + 1e-300)
                   - lt[:, None] - logfact[None, :])
        mean = (mu - 0.5 * sig ** 2) * tau[:, None] + n[None, :] * muJ
        var = sig ** 2 * tau[:, None] + n[None, :] * sJ ** 2
        loggauss = -0.5 * (np.log(2 * np.pi * var)
                           + (r[:, None] - mean) ** 2 / var)
        m_ = logpois + loggauss
        mx = m_.max(axis=1, keepdims=True)
        return float((mx[:, 0] + np.log(np.exp(m_ - mx).sum(axis=1))).sum())

    return log_post


# --------------------------------------------------------------- Henon-Heiles
class HenonHeilesDesign(DesignProblem):
    """Classical HH (potential of Lubich-Oseledets-Vandereycken, SIAM 2015,
    lambda = 0.1118 at the prior centre); observe q1 + noise."""

    obs_noise = 0.05

    def __init__(self):
        self.prior = BoxUniform(low=[0.7, 0.7, 0.02], high=[1.3, 1.3, 0.22],
                                names=["om1", "om2", "lam"])
        self.observer = DesignObserver(dt_sim=0.05, n_steps=800, k_max=64)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        from ..ode import rk4

        def rhs(x, mm, t):
            q1, q2, p1, p2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            om1, om2, lam = mm[:, 0], mm[:, 1], mm[:, 2]
            return torch.stack(
                [p1, p2,
                 -(om1 ** 2) * q1 - 2.0 * lam * q1 * q2,
                 -(om2 ** 2) * q2 - lam * (q1 ** 2 - q2 ** 2)], dim=1)

        B = m.shape[0]
        x0 = torch.zeros(B, 4)
        x0[:, 0], x0[:, 1], x0[:, 2] = 0.4, -0.2, 0.3
        sol = rk4(rhs, x0, m, self.observer.dt_sim, self.observer.n_steps)
        return sol[:, :, 0:1]


# --------------------------------------------------------------- Hodgkin-Huxley
def _vtrap(x, y):
    z = x / y
    small = z.abs() < 1e-6
    return torch.where(small, y * (1.0 + z / 2.0), x / (1.0 - torch.exp(-z)))


class HodgkinHuxleyDesign(DesignProblem):
    """Classic HH neuron; V observed in units of 100 mV, noise 2 mV."""

    obs_noise = 0.02

    def __init__(self):
        self.prior = BoxUniform(low=[60.0, 10.0, 0.05, 4.0],
                                high=[180.0, 50.0, 0.50, 12.0],
                                names=["gNa", "gK", "gL", "I"])
        self.observer = DesignObserver(dt_sim=0.02 / 60.0, n_steps=3000,
                                       k_max=96)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        gNa, gK, gL, I = (m[:, j] for j in range(4))
        ENa, EK, EL = 50.0, -77.0, -54.4
        dt = 0.02
        V = torch.full((B,), -65.0)
        mm = torch.full((B,), 0.0529)
        h = torch.full((B,), 0.5961)
        n = torch.full((B,), 0.3177)
        out = torch.zeros(B, 3001, 1)
        out[:, 0, 0] = V / 100.0
        for i in range(3000):
            am = 0.1 * _vtrap(V + 40.0, 10.0)
            bm = 4.0 * torch.exp(-(V + 65.0) / 18.0)
            ah = 0.07 * torch.exp(-(V + 65.0) / 20.0)
            bh = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
            an = 0.01 * _vtrap(V + 55.0, 10.0)
            bn = 0.125 * torch.exp(-(V + 65.0) / 80.0)
            mm = (mm + dt * am) / (1.0 + dt * (am + bm))
            h = (h + dt * ah) / (1.0 + dt * (ah + bh))
            n = (n + dt * an) / (1.0 + dt * (an + bn))
            V = V + dt * (I - gNa * mm ** 3 * h * (V - ENa)
                          - gK * n ** 4 * (V - EK) - gL * (V - EL))
            out[:, i + 1, 0] = V / 100.0
        return out


# --------------------------------------------------------------- PK
class PharmacoKineticsDesign(DesignProblem):
    """Oral one-compartment Bateman curve; log-normal assay noise. The
    real-world archetype of irregular designs (blood draws)."""

    DOSE = 500.0
    LOGSD = 0.10

    def __init__(self):
        self.prior = BoxUniform(low=[0.5, 0.05, 20.0],
                                high=[4.0, 0.40, 100.0],
                                names=["ka", "ke", "V"])
        self.observer = DesignObserver(dt_sim=24.0 / 500.0, n_steps=500,
                                       k_max=64)
        self.k_min = 3

    def trajectories(self, m, generator=None):
        tg = torch.arange(self.observer.n_steps + 1,
                          dtype=torch.float32) * self.observer.dt_sim
        ka, ke, V = m[:, 0:1], m[:, 1:2], m[:, 2:3]
        c = (self.DOSE * ka / (V * (ka - ke))
             * (torch.exp(-ke * tg[None]) - torch.exp(-ka * tg[None])))
        return c[..., None]


def pk_logpost_factory(t_obs, y_obs, dose=500.0, logsd=0.10):
    """Exact PK log-likelihood: log-normal residuals on the Bateman curve."""
    logy = np.log(np.maximum(y_obs, 1e-9))

    def log_post(p):
        ka, ke, V = p
        if ka <= ke + 1e-4 or V <= 0:
            return -np.inf
        c = dose * ka / (V * (ka - ke)) * (np.exp(-ke * t_obs)
                                           - np.exp(-ka * t_obs))
        c = np.maximum(c, 1e-9)
        z = (logy - np.log(c)) / logsd
        return float(-0.5 * np.sum(z ** 2) - logy.size * math.log(logsd))

    return log_post


# --------------------------------------------------------------- Fisher-KPP
class FisherKPPDesign(DesignProblem):
    """Reaction-diffusion PDE theta_t = D theta_xx + r theta(1-theta) on
    [0,1], no-flux BC, bump IC; 3 point sensors, random time x sensor
    designs. The posterior concentrates on the D*r ridge (front speed) --
    see CALIBRATION.md for the measured budget curve of the along-ridge
    residual."""

    obs_noise = 0.02
    NX = 64
    SENSORS = (16, 32, 48)

    def __init__(self):
        self.prior = BoxUniform(low=[0.001, 2.0], high=[0.010, 10.0],
                                names=["D", "r"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=200, k_max=96,
                                       n_channels=3)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        D, r = m[:, 0:1], m[:, 1:2]
        nx = self.NX
        dx = 1.0 / (nx - 1)
        x = torch.linspace(0, 1, nx)
        th = 0.5 * torch.exp(-((x - 0.3) / 0.05) ** 2)[None, :] \
            .expand(B, -1).clone()
        sub = 4
        dt = self.observer.dt_sim / sub
        out = torch.zeros(B, self.observer.n_steps + 1, 3)
        out[:, 0] = th[:, list(self.SENSORS)]
        for i in range(self.observer.n_steps):
            for _ in range(sub):
                lap = torch.zeros_like(th)
                lap[:, 1:-1] = (th[:, 2:] - 2 * th[:, 1:-1]
                                + th[:, :-2]) / dx ** 2
                lap[:, 0] = 2 * (th[:, 1] - th[:, 0]) / dx ** 2
                lap[:, -1] = 2 * (th[:, -2] - th[:, -1]) / dx ** 2
                th = (th + dt * (D * lap + r * th * (1 - th))).clamp(0.0, 1.5)
            out[:, i + 1] = th[:, list(self.SENSORS)]
        return out


def kpp_logpost_factory(prob: FisherKPPDesign, tidx, cidx, y_obs):
    """Exact KPP log-likelihood: Gaussian obs noise around the deterministic
    solve at candidate (D, r) -- one PDE solve per evaluation."""
    s2 = prob.obs_noise ** 2

    def log_post(p):
        f = prob.trajectories(
            torch.tensor([[p[0], p[1]]], dtype=torch.float32))[0].numpy()
        return float(-0.5 * np.sum((y_obs - f[tidx, cidx]) ** 2) / s2)

    return log_post


DESIGN_ZOO = {
    "heston": HestonDesign,
    "merton": MertonDesign,
    "henon_heiles": HenonHeilesDesign,
    "hodgkin_huxley": HodgkinHuxleyDesign,
    "pk": PharmacoKineticsDesign,
    "kpp": FisherKPPDesign,
}
