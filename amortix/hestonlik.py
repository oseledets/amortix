"""Reference likelihood for the Heston design problem.

A bootstrap particle filter that runs the simulator's own update rule.
Two properties make it the right instrument here:

  * it is the likelihood of the process that produced the data.  The
    simulator integrates the variance by clamped Euler, so whenever the
    Feller condition 2*kap*th >= xi^2 is violated the variance spends real
    time pinned at zero.  An idealised (noncentral chi-square) transition
    never sees that floor, and the gap between the two grows with xi;
  * prices are observed exactly, so the last sub-step of every gap pins the
    price innovation z1.  Feeding that pinned z1 into the variance update is
    the only channel through which rho enters the likelihood at all.

The estimator is unbiased for the likelihood (standard bootstrap filter),
so it may be used inside pseudo-marginal MCMC without further correction.
"""
from __future__ import annotations

import math

import numpy as np
import torch

LOG2PI = float(np.log(2.0 * np.pi))
_NEG = -1e30


# ------------------------------------------------------------------ numpy
def _systematic(w, u):
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, (u + np.arange(w.size)) / w.size)


def heston_loglik_np(theta, tidx, y, dt=0.01, n_part=2048, seed=0):
    """Single-theta filter in numpy: the readable definition of the target."""
    mu, kap, th, xi, rho = (float(x) for x in theta)
    rng = np.random.default_rng(seed)
    sq = math.sqrt(dt)
    rr = math.sqrt(max(1.0 - rho * rho, 0.0))
    S = np.ones(n_part)
    v = np.full(n_part, th)
    ll, prev = 0.0, 0
    for j, t in enumerate(tidx):
        g = int(t) - prev
        prev = int(t)
        for _ in range(g - 1):
            z1 = rng.standard_normal(n_part)
            z2 = rng.standard_normal(n_part)
            vp = np.maximum(v, 0.0)
            rt = np.sqrt(vp)
            S = np.maximum(S * (1.0 + mu * dt + rt * sq * z1), 1e-8)
            v = v + kap * (th - vp) * dt + xi * rt * sq * (rho * z1 + rr * z2)
        vp = np.maximum(v, 0.0)
        rt = np.sqrt(vp)
        scale = S * rt * sq
        ok = scale > 0.0
        z1 = (y[j] - S * (1.0 + mu * dt)) / np.where(ok, scale, 1.0)
        logw = np.where(ok, -0.5 * z1 * z1 - 0.5 * LOG2PI
                        - np.log(np.where(ok, scale, 1.0)), -np.inf)
        mx = logw.max()
        if not np.isfinite(mx):
            return -np.inf
        w = np.exp(logw - mx)
        sw = w.sum()
        ll += mx + math.log(sw / n_part)
        z2 = rng.standard_normal(n_part)
        v = v + kap * (th - vp) * dt + xi * rt * sq * (rho * z1 + rr * z2)
        S = np.full(n_part, float(y[j]))
        v = v[_systematic(w / sw, rng.random())]
    return float(ll)


# ------------------------------------------------------------------ torch
def heston_loglik_torch(theta, tidx, y, dt=0.01, n_part=1024, generator=None,
                        device=None):
    """The same filter, batched over theta.  Returns (T,) log-likelihoods."""
    dev = device or theta.device
    theta = theta.to(dev, torch.float32)
    T = theta.shape[0]
    mu, kap, th, xi, rho = (theta[:, j:j + 1] for j in range(5))
    rr = (1.0 - rho * rho).clamp_min(0.0).sqrt()
    sq = math.sqrt(dt)
    yv = torch.as_tensor(np.asarray(y, dtype=np.float32), device=dev)

    S = torch.ones(T, n_part, device=dev)
    v = th.expand(T, n_part).clone()
    ll = torch.zeros(T, device=dev)
    ar = torch.arange(n_part, device=dev, dtype=torch.float32)
    prev = 0
    for j, t in enumerate(tidx):
        g = int(t) - prev
        prev = int(t)
        for _ in range(g - 1):
            z1 = torch.randn(T, n_part, device=dev, generator=generator)
            z2 = torch.randn(T, n_part, device=dev, generator=generator)
            vp = v.clamp_min(0.0)
            rt = vp.sqrt()
            S = (S * (1.0 + mu * dt + rt * sq * z1)).clamp_min(1e-8)
            v = v + kap * (th - vp) * dt + xi * rt * sq * (rho * z1 + rr * z2)
        vp = v.clamp_min(0.0)
        rt = vp.sqrt()
        scale = S * rt * sq
        ok = scale > 0.0
        safe = torch.where(ok, scale, torch.ones_like(scale))
        z1 = (yv[j] - S * (1.0 + mu * dt)) / safe
        logw = torch.where(ok, -0.5 * z1 * z1 - 0.5 * LOG2PI - safe.log(),
                           torch.full_like(scale, -float("inf")))
        mx = logw.max(1, keepdim=True).values
        dead = ~torch.isfinite(mx)
        mx = torch.where(dead, torch.zeros_like(mx), mx)
        w = torch.where(dead.expand_as(logw), torch.ones_like(logw),
                        (logw - mx).exp())
        sw = w.sum(1, keepdim=True)
        ll = ll + torch.where(dead, torch.full_like(mx, _NEG),
                              mx + (sw / n_part).log()).squeeze(1)
        z2 = torch.randn(T, n_part, device=dev, generator=generator)
        z1 = torch.where(torch.isfinite(z1), z1, torch.zeros_like(z1))
        v = v + kap * (th - vp) * dt + xi * rt * sq * (rho * z1 + rr * z2)
        S = yv[j].expand(T, n_part).clone()
        u = (torch.rand(T, 1, device=dev, generator=generator) + ar) / n_part
        cum = (w / sw).cumsum(1)
        cum[:, -1] = 1.0
        idx = torch.searchsorted(cum.contiguous(), u.contiguous())
        v = v.gather(1, idx.clamp(max=n_part - 1))
    return ll.clamp_min(_NEG)


def heston_logpost_factory(problem, tidx, y, n_part=1024, device=None,
                           seed=0):
    """Batched log-posterior over the prior box, for the SMC/PMMH engines."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lo = problem.prior.low.to(dev)
    hi = problem.prior.high.to(dev)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    dt = problem.observer.dt_sim

    def logp(m):
        m = torch.as_tensor(m, dtype=torch.float32, device=dev)
        single = m.ndim == 1
        if single:
            m = m[None]
        inside = ((m >= lo) & (m <= hi)).all(1)
        out = torch.full((m.shape[0],), _NEG, device=dev)
        if inside.any():
            out[inside] = heston_loglik_torch(m[inside], tidx, y, dt=dt,
                                              n_part=n_part, generator=gen,
                                              device=dev)
        out = out.cpu()
        return out[0].item() if single else out

    return logp
