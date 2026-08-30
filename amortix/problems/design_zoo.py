"""Design-amortized problem zoo: p(m | any K observation points).

Every case here follows the DesignProblem protocol (amortix.designs): full
raw trajectories + fresh random designs every optimizer step. Calibration
status and the measurements behind the defaults are documented in the
technical report (report/techreport.pdf); the recommended configuration is
simply the defaults --

    post = FlowPosterior(prob)          # embed/rope resolve by class rule
    post.fit(n_train=..., steps=..., retokenize=prob.make_retokenizer())

(embed="auto" picks the set-conditioned pair embedding for Markov-observed
cases and bare points otherwise; rope="auto" picks continuous time-RoPE for
any DesignProblem).

Cases, keyed by their DESIGN_ZOO registry names:
  gbm_rd           GBMDesign (design_basic): geometric Brownian motion,
                   exact conjugate reference (2 parameters)
  ou_rd            OUDesign (design_basic): Ornstein-Uhlenbeck, stationary
                   start, exact-likelihood reference (2 parameters)
  heston           HestonDesign: hidden stochastic volatility, correlated
                   noises, price-only observations (5 parameters)
  merton           MertonDesign: jump-diffusion in log-price (5 parameters);
                   near-exact Poisson-mixture likelihood ->
                   merton_logpost_factory
  henon_heiles     HenonHeilesDesign: classical Hamiltonian with the
                   Lubich-Oseledets-Vandereycken potential; noisy q1 only
                   (3 parameters)
  hodgkin_huxley   HodgkinHuxleyDesign: sbibm's flagship spiking neuron
                   (4 parameters)
  pk               PharmacoKineticsDesign: oral one-compartment Bateman
                   curve, log-normal assay noise -- irregular blood draws
                   (3 parameters); exact likelihood -> pk_logpost_factory
  kpp              FisherKPPDesign: reaction-diffusion PDE, 3 point sensors,
                   random time x sensor designs (2 parameters); exact
                   likelihood via the deterministic solve ->
                   kpp_logpost_factory
  fhn              FHNDesign: deterministic FitzHugh-Nagumo flow, membrane
                   potential observed with noise at arbitrary times
                   (4 parameters); exact likelihood -> fhn_logpost_factory
  seir             SEIRDesign: five-compartment SEIR-D epidemic, I and D
                   observed at arbitrary times (5 parameters); exact
                   likelihood -> seir_logpost_factory
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
    designs. The posterior concentrates on the D*r ridge (front speed)."""

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


from .design_basic import GBMDesign, OUDesign  # noqa: E402

DESIGN_ZOO = {
    "gbm_rd": GBMDesign,
    "ou_rd": OUDesign,
    "heston": HestonDesign,
    "merton": MertonDesign,
    "henon_heiles": HenonHeilesDesign,
    "hodgkin_huxley": HodgkinHuxleyDesign,
    "pk": PharmacoKineticsDesign,
    "kpp": FisherKPPDesign,
}

class FHNDesign(DesignProblem):
    """FitzHugh--Nagumo with the membrane potential observed at ARBITRARY
    times. The dynamics are the deterministic FHN flow (the suite's gallery
    instrument); randomness enters only as observation noise, so the
    reference likelihood is exact: one RK4 solve per evaluation."""

    obs_noise = 0.05

    def __init__(self):
        self.prior = BoxUniform(low=[0.5, 0.5, 0.05, 0.0],
                                high=[0.9, 0.9, 0.30, 0.5],
                                names=["a", "b", "eps", "I"])
        self.observer = DesignObserver(dt_sim=0.05, n_steps=800, k_max=64)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        from ..ode import rk4

        def rhs(x, mm, t):
            v, w = x[:, 0], x[:, 1]
            a, b, eps, I = mm[:, 0], mm[:, 1], mm[:, 2], mm[:, 3]
            return torch.stack([v - v ** 3 / 3.0 - w + I,
                                eps * (v + a - b * w)], dim=1)

        B = m.shape[0]
        x0 = torch.zeros(B, 2); x0[:, 0] = -1.0; x0[:, 1] = 1.0
        sol = rk4(rhs, x0, m, self.observer.dt_sim, self.observer.n_steps)
        return sol[:, :, 0:1]                       # v only


def fhn_logpost_factory(prob, tidx, y_obs):
    """Exact log-posterior: deterministic RK4 solve + Gaussian observation
    noise at the design's times. Numpy solver: one evaluation ~ milliseconds."""
    t_sel = np.asarray(tidx, dtype=int)
    y = np.asarray(y_obs, dtype=np.float64)
    dt = prob.observer.dt_sim
    n = prob.observer.n_steps
    sd = prob.obs_noise

    def lp(m):
        a, b, eps, I = m
        v, w = -1.0, 1.0
        vs = np.empty(n + 1); vs[0] = v

        def f(v, w):
            return v - v ** 3 / 3.0 - w + I, eps * (v + a - b * w)

        for i in range(n):
            k1v, k1w = f(v, w)
            k2v, k2w = f(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
            k3v, k3w = f(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
            k4v, k4w = f(v + dt * k3v, w + dt * k3w)
            v += dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
            w += dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            vs[i + 1] = v
        r = (y - vs[t_sel]) / sd
        return -0.5 * float(np.sum(r * r))

    return lp


DESIGN_ZOO["fhn"] = FHNDesign

class SEIRDesign(DesignProblem):
    """Five-compartment SEIR-D epidemic; I and D observed at arbitrary times.

    Deterministic compartment flow with observation noise, so the reference
    likelihood is exact: one RK4 solve per evaluation, like FitzHugh--Nagumo.
    Two observed channels, which the token contract carries natively.
    """

    obs_noise = 0.004

    def __init__(self, t_switch: float = 20.0):
        self.t_switch = float(t_switch)
        self.prior = BoxUniform(low=[0.20, 0.05, 0.20, 0.05, 0.005],
                                high=[0.60, 0.30, 0.50, 0.20, 0.050],
                                names=["beta1", "beta2", "alpha", "gamma_r",
                                       "gamma_d"])
        self.observer = DesignObserver(dt_sim=0.1, n_steps=600, k_max=64,
                                       n_channels=2)
        self.k_min = 4

    def _rhs(self, x, m, t):
        S, E, I, R, D = (x[:, i] for i in range(5))
        b1, b2, al, gr, gd = (m[:, i] for i in range(5))
        beta = torch.where(torch.as_tensor(t) < self.t_switch, b1, b2)
        inf = beta * S * I
        return torch.stack([-inf, inf - al * E, al * E - (gr + gd) * I,
                            gr * I, gd * I], dim=1)

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        x = torch.zeros(B, 5); x[:, 0] = 0.999; x[:, 2] = 0.001
        dt = self.observer.dt_sim
        out = torch.zeros(B, self.observer.n_steps + 1, 2)
        out[:, 0, 0], out[:, 0, 1] = x[:, 2], x[:, 4]
        for i in range(self.observer.n_steps):
            t = i * dt
            k1 = self._rhs(x, m, t)
            k2 = self._rhs(x + 0.5 * dt * k1, m, t + 0.5 * dt)
            k3 = self._rhs(x + 0.5 * dt * k2, m, t + 0.5 * dt)
            k4 = self._rhs(x + dt * k3, m, t + dt)
            x = (x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)).clamp(0.0, 1.0)
            out[:, i + 1, 0], out[:, i + 1, 1] = x[:, 2], x[:, 4]
        return out


def seir_logpost_factory(prob, tidx, cidx, y_obs):
    """Exact log-posterior: RK4 solve, Gaussian noise on the observed channel."""
    t_sel = np.asarray(tidx, dtype=int)
    c_sel = np.asarray(cidx, dtype=int)
    y = np.asarray(y_obs, dtype=np.float64)
    dt = prob.observer.dt_sim
    n = prob.observer.n_steps
    sd = prob.obs_noise
    tsw = prob.t_switch

    def lp(m):
        b1, b2, al, gr, gd = m
        S, E, I, R, D = 0.999, 0.0, 0.001, 0.0, 0.0
        obs = np.empty((n + 1, 2)); obs[0] = (I, D)

        def f(S, E, I, R, D, t):
            beta = b1 if t < tsw else b2
            inf = beta * S * I
            return (-inf, inf - al * E, al * E - (gr + gd) * I, gr * I, gd * I)

        for i in range(n):
            t = i * dt
            k1 = f(S, E, I, R, D, t)
            k2 = f(*[v + 0.5 * dt * k for v, k in zip((S, E, I, R, D), k1)], t + 0.5 * dt)
            k3 = f(*[v + 0.5 * dt * k for v, k in zip((S, E, I, R, D), k2)], t + 0.5 * dt)
            k4 = f(*[v + dt * k for v, k in zip((S, E, I, R, D), k3)], t + dt)
            S, E, I, R, D = [min(max(v + dt / 6.0 * (a + 2 * b + 2 * c + d), 0.0), 1.0)
                             for v, a, b, c, d in zip((S, E, I, R, D), k1, k2, k3, k4)]
            obs[i + 1] = (I, D)
        r = (y - obs[t_sel, c_sel]) / sd
        return -0.5 * float(np.sum(r * r))

    return lp


DESIGN_ZOO["seir"] = SEIRDesign

def hh_logpost_factory(prob, tidx, y_obs, backend="numpy"):
    """Exact log-posterior for Hodgkin--Huxley.

    ``backend="numpy"`` mirrors the package simulator step for step in plain
    numpy: the reference likelihood must describe exactly the chain the
    simulator generates, so the two implementations are kept in lockstep --
    an independently written version of the membrane equations can describe
    a measurably different neuron while every reference chain converges on
    it.

    Why a second implementation exists at all: a 3,000-step torch integration
    at batch one is bound by kernel launches, and nested sampling asks for
    points one at a time; the numpy path removes that penalty without
    changing the model.
    """
    t_sel = np.asarray(tidx, dtype=int)
    y = np.asarray(y_obs, dtype=np.float64)
    sd = float(prob.obs_noise)

    def _solve(m):
        gNa, gK, gL, I = (float(v) for v in m)
        ENa, EK, EL, dt = 50.0, -77.0, -54.4, 0.02
        V, mm, h, n = -65.0, 0.0529, 0.5961, 0.3177
        out = np.empty(3001); out[0] = V / 100.0
        for i in range(3000):
            x = V + 40.0
            am = 0.1 * (10.0 if abs(x / 10.0) < 1e-6 else x / (1.0 - np.exp(-x / 10.0)))
            bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
            ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
            bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            x2 = V + 55.0
            an = 0.01 * (10.0 if abs(x2 / 10.0) < 1e-6 else x2 / (1.0 - np.exp(-x2 / 10.0)))
            bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
            mm = (mm + dt * am) / (1.0 + dt * (am + bm))
            h = (h + dt * ah) / (1.0 + dt * (ah + bh))
            n = (n + dt * an) / (1.0 + dt * (an + bn))
            V = V + dt * (I - gNa * mm ** 3 * h * (V - ENa)
                          - gK * n ** 4 * (V - EK) - gL * (V - EL))
            out[i + 1] = V / 100.0
        return out

    def lp_np(m):
        r = (y - _solve(m)[t_sel]) / sd
        return -0.5 * float(np.sum(r * r))

    def lp_torch(m):
        mt = torch.as_tensor(np.asarray(m, np.float32))[None, :]
        with torch.no_grad():
            tr = prob.trajectories(mt, None)
        r = (y - tr[0, t_sel, 0].numpy().astype(np.float64)) / sd
        return -0.5 * float(np.sum(r * r))

    return lp_np if backend == "numpy" else lp_torch
