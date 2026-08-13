"""Paper-figure campaign: SBC ranks (gallery), budget grids, posterior pairs.

Modes:
  gallery <case>            train production config + SBC 500x200, save ranks
  budget <case> <n_train>   case in {gbm, ou, kpp40}; shared cached test refs
  pairs <case>              case in {merton, pk, kpp}; 3 datasets, samples+ref

Outputs into $OUT (default ~/amortix_exp/paper2).
"""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # examples/ copied next to this
sys.path.insert(0, os.path.expanduser("~/amortix_exp"))
OUT = os.environ.get("OUT", os.path.expanduser("~/amortix_exp/paper2"))
os.makedirs(OUT, exist_ok=True)

from amortix import FlowPosterior
from amortix.diagnostics import run_sbc, sbc_uniformity
from amortix.mcmc import (observed_indices, log_likelihood_ou,
                          log_likelihood_gbm, metropolis)


# ----------------------------------------------------------------- gallery
def mode_gallery(case):
    mod = importlib.import_module(f"amortix.problems.{case}")
    prob = mod.make()
    torch.manual_seed(0)
    post = FlowPosterior(prob)
    t0 = time.time()
    post.fit(n_train=40000, epochs=35, seed=0, verbose=False, device="cuda")
    t_train = time.time() - t0
    res = run_sbc(post, prob, n_sims=500, n_post=200, seed=1)
    p = sbc_uniformity(res["ranks"], res["n_post"])
    np.savez(f"{OUT}/sbc_{case}.npz", ranks=res["ranks"],
             n_post=res["n_post"], names=np.array(res["names"]),
             pvals=np.asarray(p), train_s=t_train)
    print(f"[{case}] train {t_train:.0f}s  p=" +
          " ".join(f"{nm}:{v:.2f}" for nm, v in zip(res["names"], p)),
          flush=True)


# ----------------------------------------------------------------- budget
def _gbm_refs():
    from baseline_npe import exact_posterior
    from amortix.problems import gbm
    prob = gbm.make()
    idx = observed_indices(prob)
    gen = torch.Generator().manual_seed(999)
    m = prob.prior.sample(200, gen)
    traj = prob.simulate_paths(m, gen)
    tokens = prob.observer.tokens_from_traj(traj)
    exact = np.stack([exact_posterior(prob, traj[i, :, 0].numpy(), idx,
                                      seed=5000 + i) for i in range(200)])
    return prob, tokens, exact, 6000


def _ou_refs():
    from amortix.problems import ou
    prob = ou.make()
    idx = observed_indices(prob)
    dt = prob.observer.dt_sim
    gaps = np.diff(idx).astype(np.float64) * dt
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    gen = torch.Generator().manual_seed(999)
    m = prob.prior.sample(100, gen)
    traj = prob.simulate_paths(m, gen)
    tokens = prob.observer.tokens_from_traj(traj)
    cache = f"{OUT}/refcache_ou.npz"
    if os.path.exists(cache):
        exact = np.load(cache)["exact"]
    else:
        chains = []
        for i in range(100):
            s = traj[i, idx, 0].numpy().astype(np.float64)

            def lp(v, s=s):
                th, sg = float(v[0]), float(v[1])
                if th <= 0 or sg <= 0:
                    return -np.inf
                base = log_likelihood_ou(s, v, gaps, scheme="euler",
                                         dt_fine=dt)
                var0 = sg ** 2 / (2.0 * th)
                return base - 0.5 * (s[0] ** 2 / var0
                                     + math.log(2 * math.pi * var0))
            c, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=2000,
                              prior_low=lo, prior_high=hi, seed=i)
            chains.append(np.asarray(c))
        exact = np.stack(chains)
        np.savez(cache, exact=exact)
    return prob, tokens, exact, 6000


def _kpp_refs():
    from baseline_kpp import _fixed_design, _mcmc_worker, _tokens_from_y, K_OBS
    from concurrent.futures import ProcessPoolExecutor
    prob, tidx, cidx = _fixed_design()
    B = 16
    gen = torch.Generator().manual_seed(999)
    m = prob.prior.sample(B, gen)
    raw = prob.trajectories(m)
    gen_noise = torch.Generator().manual_seed(777)
    y = raw[:, tidx, :].gather(
        2, cidx.view(1, -1, 1).expand(B, -1, 1)).squeeze(-1)
    y = y + prob.obs_noise * torch.randn(y.shape, generator=gen_noise)
    tokens = torch.stack([_tokens_from_y(prob, tidx, cidx, y[i])
                          for i in range(B)])
    cache = f"{OUT}/refcache_kpp.npz"
    if os.path.exists(cache):
        exact = np.load(cache)["exact"]
    else:
        jobs = [(i, y[i].numpy().astype(np.float64), 2000) for i in range(B)]
        exact = [None] * B
        with ProcessPoolExecutor(max_workers=16) as ex:
            for i, chain, disc, dtm in ex.map(_mcmc_worker, jobs):
                exact[i] = chain
        exact = np.stack(exact)
        np.savez(cache, exact=exact, y=y.numpy())
    return prob, tokens, exact, 12000, (tidx, cidx)


def mode_budget(case, n_train):
    if case == "gbm":
        prob, tokens, exact, base_steps = _gbm_refs()
        retok = None
    elif case == "ou":
        prob, tokens, exact, base_steps = _ou_refs()
        retok = None
    elif case == "kpp40":
        prob, tokens, exact, base_steps, (tidx, cidx) = _kpp_refs()
        gen_retok = torch.Generator().manual_seed(31337)
        K = tokens.shape[1]

        def retok(raw_b, _g):
            B = raw_b.shape[0]
            tk = torch.zeros(B, K, 6)
            for i in range(B):
                tk[i] = prob.tokens_for(raw_b[i], tidx, cidx, gen_retok)
            return tk, torch.ones(B, K, dtype=torch.bool)
    steps = max(300, round(base_steps * n_train / 20000))
    torch.manual_seed(0)
    post = FlowPosterior(prob)
    t0 = time.time()
    post.fit(n_train=n_train, steps=steps, seed=0, verbose=False,
             device="cuda", retokenize=retok)
    t_train = time.time() - t0
    smp = post.sample_batch(tokens, n=2000, seed=0).numpy()
    B = exact.shape[0]
    out = dict(case=case, n_train=n_train, steps=steps, train_s=t_train)
    for j, nm in enumerate(prob.prior.names):
        bias = np.array([(smp[i, :, j].mean() - exact[i, :, j].mean())
                         / max(exact[i, :, j].std(), 1e-12)
                         for i in range(B)])
        width = np.array([smp[i, :, j].std()
                          / max(exact[i, :, j].std(), 1e-12)
                          for i in range(B)])
        out[nm] = dict(bias=float(bias.mean()),
                       bias_se=float(bias.std() / np.sqrt(B)),
                       width=float(width.mean()),
                       width_se=float(width.std() / np.sqrt(B)))
    with open(f"{OUT}/budget_{case}_{n_train}.json", "w") as f:
        json.dump(out, f, indent=1)
    if case == "kpp40" and n_train == 20000:
        torch.save(post.state_dict(), f"{OUT}/kpp40_20000.pt")
    print(f"[budget {case} {n_train}] steps={steps} train {t_train:.0f}s "
          + " ".join(f"{nm}:{out[nm]['bias']:+.2f}/{out[nm]['width']:.2f}"
                     for nm in prob.prior.names), flush=True)


# ----------------------------------------------------------------- pairs
def _run_two_chains(lp, lo, hi, seed, n=4000):
    c1, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=n,
                       prior_low=lo, prior_high=hi, seed=seed)
    c2, _ = metropolis(lp, 0.25 * lo + 0.75 * hi, n_samples=n,
                       prior_low=lo, prior_high=hi, seed=seed + 5000)
    d = max(abs(c1[:, j].mean() - c2[:, j].mean())
            / max(c1[:, j].std(), 1e-12) for j in range(c1.shape[1]))
    return np.asarray(c1), float(d)


def mode_pairs(case):
    from amortix.problems.design_zoo import (
        MertonDesign, PharmacoKineticsDesign, merton_logpost_factory,
        pk_logpost_factory)
    if case == "merton":
        prob = MertonDesign()
        K = 48
    elif case == "pk":
        prob = PharmacoKineticsDesign()
        K = 6
    elif case == "kpp":
        return mode_pairs_kpp()
    torch.manual_seed(0)
    post = FlowPosterior(prob)
    t0 = time.time()
    post.fit(n_train=20000, steps=12000, seed=0, verbose=False,
             device="cuda", retokenize=prob.make_retokenizer())
    print(f"[pairs {case}] trained {time.time()-t0:.0f}s", flush=True)
    gen = torch.Generator().manual_seed(2024)
    m_true = prob.prior.sample(3, gen)
    raw = prob.trajectories(m_true, generator=gen)
    tidx, cidx = prob.sample_design(gen, K)
    tidx = torch.unique(tidx)          # repeated times give zero gaps -> -inf
    cidx = torch.zeros_like(tidx)
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    toks, refs, discs = [], [], []
    for i in range(3):
        tok = prob.tokens_for(raw[i], tidx, cidx, gen)
        toks.append(tok)
        y = tok[:, 1].numpy().astype(np.float64)
        t_obs = tidx.numpy() * prob.observer.dt_sim
        if case == "merton":
            vals = np.concatenate([[1.0], y])
            tt = np.concatenate([[0.0], t_obs])
            r = np.diff(np.log(np.maximum(vals, 1e-9)))
            tau = np.diff(tt)
            lp = merton_logpost_factory(r, tau, lo, hi)
        else:
            lp = pk_logpost_factory(t_obs, y, dose=prob.DOSE,
                                    logsd=prob.LOGSD)
        c, d = _run_two_chains(lp, lo, hi, seed=100 + i)
        refs.append(c)
        discs.append(d)
        print(f"[pairs {case}] ds{i} chain-check {d:.3f} sd", flush=True)
    tokens = torch.stack(toks)
    smp = post.sample_batch(tokens, n=4000, seed=0).numpy()
    np.savez(f"{OUT}/pairs_{case}.npz", amortix=smp,
             ref=np.stack(refs), m_true=m_true.numpy(),
             names=np.array(prob.prior.names), chain_check=np.array(discs))
    print(f"[pairs {case}] saved", flush=True)


def mode_pairs_kpp():
    from baseline_kpp import _fixed_design, _tokens_from_y
    prob, tidx, cidx = _fixed_design()
    torch.manual_seed(0)
    post = FlowPosterior(prob)
    # build with one forward-shaped batch before loading weights
    gen0 = torch.Generator().manual_seed(1)
    m0 = prob.prior.sample(2, gen0)
    raw0 = prob.trajectories(m0)
    t0 = prob.tokens_for(raw0[0], tidx, cidx, gen0)
    post.sample_batch(t0[None], n=2, seed=0)
    post.load_state_dict(torch.load(f"{OUT}/kpp40_20000.pt",
                                    map_location="cpu"))
    d = np.load(f"{OUT}/refcache_kpp.npz")
    exact, y = d["exact"], d["y"]
    pick = [0, 1, 2]
    tokens = torch.stack([_tokens_from_y(prob, tidx, cidx,
                                         torch.tensor(y[i]))
                          for i in pick])
    smp = post.sample_batch(tokens, n=4000, seed=0).numpy()
    gen = torch.Generator().manual_seed(999)
    m_true = prob.prior.sample(16, gen)[pick].numpy()
    np.savez(f"{OUT}/pairs_kpp.npz", amortix=smp, ref=exact[pick],
             m_true=m_true, names=np.array(prob.prior.names))
    print("[pairs kpp] saved", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gallery", "budget", "pairs"])
    ap.add_argument("case")
    ap.add_argument("n_train", nargs="?", type=int)
    a = ap.parse_args()
    if a.mode == "gallery":
        mode_gallery(a.case)
    elif a.mode == "budget":
        mode_budget(a.case, a.n_train)
    else:
        mode_pairs(a.case)
    print("JOB_DONE", flush=True)
