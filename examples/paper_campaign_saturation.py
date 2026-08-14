"""Saturation campaign: every system, small and big model, trained long.

Modes:
  refs merton|pk          build 32-set MCMC reference batteries (CPU)
  run <case> <size>       train to saturation + evaluate (FID or SBC)

size: small = default (dim 64, depth 3); big = dim 128, depth 4.
Saturation budget: 120k simulations; fixed-design systems 35 epochs,
variable-design systems 72k steps with the canonical retokenizer.
Outputs to $OUT (json + sample dumps for FID systems).
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

sys.path.insert(0, os.path.expanduser("~/amortix_exp"))
sys.path.insert(0, os.path.expanduser("~/amortix_exp/examples"))
OUT = os.environ.get("OUT", os.path.expanduser("~/amortix_exp/paper2/sat"))
os.makedirs(OUT, exist_ok=True)

from amortix import FlowPosterior
from amortix.diagnostics import run_sbc, sbc_uniformity
from amortix.mcmc import observed_indices, metropolis

FIXED = ["linear_gaussian", "ou", "gbm", "cir", "double_well", "stoch_lv",
         "fhn", "seir", "sindy_sde"]
ZOO = ["heston", "merton", "henon_heiles", "hodgkin_huxley", "pk"]
REF_CASES = ["linear_gaussian", "ou", "gbm", "merton", "pk", "kpp40"]


def make_problem(case):
    if case in FIXED:
        mod = importlib.import_module(f"amortix.problems.{case}")
        return mod.make(), "fixed"
    if case == "kpp40":
        from baseline_kpp import _fixed_design
        prob, tidx, cidx = _fixed_design()
        return (prob, tidx, cidx), "kpp40"
    from amortix.problems.design_zoo import DESIGN_ZOO
    return DESIGN_ZOO[case](), "zoo"


def build_model(prob, size):
    if size == "small":
        return FlowPosterior(prob)
    return FlowPosterior(prob, dim_model=128, depth=4)


# --------------------------------------------------------- reference batteries
def mode_refs(which):
    from amortix.problems.design_zoo import (MertonDesign,
                                             PharmacoKineticsDesign,
                                             merton_logpost_factory,
                                             pk_logpost_factory)
    B = 32
    if which == "merton":
        prob = MertonDesign()
        gen = torch.Generator().manual_seed(4243)
        m_true = prob.prior.sample(B, gen)
        raw = prob.trajectories(m_true, generator=gen)
        tidx, cidx = prob.sample_design(gen, 56)
        tidx = torch.unique(tidx)
        cidx = torch.zeros_like(tidx)
        lo = prob.prior.low.numpy().astype(np.float64)
        hi = prob.prior.high.numpy().astype(np.float64)
        refs, toks = [], []
        for i in range(B):
            tok = prob.tokens_for(raw[i], tidx, cidx, gen)
            toks.append(tok)
            y = tok[:, 1].numpy().astype(np.float64)
            vals = np.concatenate([[1.0], y])
            tt = np.concatenate([[0.0], tidx.numpy() * prob.observer.dt_sim])
            r = np.diff(np.log(np.maximum(vals, 1e-9)))
            tau = np.diff(tt)
            lp = merton_logpost_factory(r, tau, lo, hi)
            c, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=2000,
                              prior_low=lo, prior_high=hi, seed=100 + i)
            refs.append(np.asarray(c))
            if (i + 1) % 8 == 0:
                print(f"  merton ref {i+1}/{B}", flush=True)
        np.savez(f"{OUT}/refbat_merton.npz", refs=np.stack(refs),
                 tokens=torch.stack(toks).numpy(), m_true=m_true.numpy())
    else:
        prob = PharmacoKineticsDesign()
        gen = torch.Generator().manual_seed(4244)
        m_true = prob.prior.sample(B, gen)
        raw = prob.trajectories(m_true, generator=gen)
        tidx, cidx = prob.sample_design(gen, 6)
        lo = prob.prior.low.numpy().astype(np.float64)
        hi = prob.prior.high.numpy().astype(np.float64)
        refs, toks = [], []
        for i in range(B):
            tok = prob.tokens_for(raw[i], tidx, cidx, gen)
            toks.append(tok)
            y = tok[:, 1].numpy().astype(np.float64)
            t_obs = tidx.numpy() * prob.observer.dt_sim
            lp = pk_logpost_factory(t_obs, y, dose=prob.DOSE,
                                    logsd=prob.LOGSD)
            c, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=2000,
                              prior_low=lo, prior_high=hi, seed=100 + i)
            refs.append(np.asarray(c))
        np.savez(f"{OUT}/refbat_pk.npz", refs=np.stack(refs),
                 tokens=torch.stack(toks).numpy(), m_true=m_true.numpy())
    print(f"[refs {which}] saved", flush=True)


# --------------------------------------------------------------------- fid
def fd2(A, R):
    from scipy.linalg import sqrtm
    s = R.std(0) + 1e-12
    A, R = A / s, R / s
    S1, S2 = np.cov(A.T), np.cov(R.T)
    cm = sqrtm(S2 @ S1)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((A.mean(0) - R.mean(0)) ** 2).sum()
                 + np.trace(S1 + S2 - 2 * cm))


def eval_fid(case, post):
    if case in ("linear_gaussian", "gbm", "ou"):
        mod = importlib.import_module(f"amortix.problems.{case}")
        prob = mod.make()
        n_test = 200 if case in ("linear_gaussian", "gbm") else 100
        gen = torch.Generator().manual_seed(999)
        m_true = prob.prior.sample(n_test, gen)
        if case == "linear_gaussian":
            tokens, _ = prob.observe(m_true, generator=gen)
            from amortix.problems.linear_gaussian import exact_posterior
            refs = [exact_posterior(np.asarray(tokens[i, :, 0]), prob,
                                    n=2000, seed=5000 + i).numpy()
                    for i in range(n_test)]
        elif case == "gbm":
            traj = prob.simulate_paths(m_true, gen)
            tokens = prob.observer.tokens_from_traj(traj)
            from baseline_npe import exact_posterior
            idx = observed_indices(prob)
            refs = [exact_posterior(prob, traj[i, :, 0].numpy(), idx,
                                    seed=5000 + i) for i in range(n_test)]
        else:
            traj = prob.simulate_paths(m_true, gen)
            tokens = prob.observer.tokens_from_traj(traj)
            refs = list(np.load(os.path.expanduser(
                "~/amortix_exp/paper2/refcache_ou.npz"))["exact"])
        smp = post.sample_batch(tokens, n=2000, seed=0).numpy()
    elif case in ("merton", "pk"):
        d = np.load(f"{OUT}/refbat_{case}.npz")
        refs = list(d["refs"])
        tokens = torch.tensor(d["tokens"])
        smp = post.sample_batch(tokens, n=2000, seed=0).numpy()
    elif case == "kpp40":
        from baseline_kpp import _fixed_design, _tokens_from_y
        prob, tidx, cidx = _fixed_design()
        d = np.load(os.path.expanduser("~/amortix_exp/fiddump/ref_kpp.npz"))
        refs = list(d["samples"])
        gen = torch.Generator().manual_seed(999)
        m = prob.prior.sample(32, gen)
        raw = prob.trajectories(m)
        gen_noise = torch.Generator().manual_seed(777)
        y = raw[:, tidx, :].gather(
            2, cidx.view(1, -1, 1).expand(32, -1, 1)).squeeze(-1)
        y = y + prob.obs_noise * torch.randn(y.shape, generator=gen_noise)
        tokens = torch.stack([_tokens_from_y(prob, tidx, cidx, y[i])
                              for i in range(32)])
        smp = post.sample_batch(tokens, n=2000, seed=0).numpy()
    vals = np.array([fd2(smp[i], refs[i]) for i in range(len(refs))])
    return dict(fid_median=float(np.median(vals)),
                fid_mean=float(vals.mean()),
                fid_se=float(vals.std() / np.sqrt(len(vals))),
                n_sets=len(refs)), smp


# --------------------------------------------------------------------- run
def mode_run(case, size):
    obj, kind = make_problem(case)
    torch.manual_seed(0)
    t0 = time.time()
    if kind == "fixed":
        prob = obj
        post = build_model(prob, size)
        post.fit(n_train=120000, epochs=35, seed=0, verbose=False,
                 device="cuda")
    elif kind == "zoo":
        prob = obj
        post = build_model(prob, size)
        post.fit(n_train=120000, steps=72000, seed=0, verbose=False,
                 device="cuda", retokenize=prob.make_retokenizer())
    else:
        prob, tidx, cidx = obj
        post = build_model(prob, size)
        gen_retok = torch.Generator().manual_seed(31337)
        K = tidx.numel()

        def retok(raw_b, _g):
            Bn = raw_b.shape[0]
            tk = torch.zeros(Bn, K, 6)
            for i in range(Bn):
                tk[i] = prob.tokens_for(raw_b[i], tidx, cidx, gen_retok)
            return tk, torch.ones(Bn, K, dtype=torch.bool)
        post.fit(n_train=120000, steps=72000, seed=0, verbose=False,
                 device="cuda", retokenize=retok)
    t_train = time.time() - t0
    n_par = sum(p.numel() for p in post.parameters())

    out = dict(case=case, size=size, params=n_par, train_s=t_train,
               n_train=120000)
    if case in REF_CASES:
        fid, smp = eval_fid(case, post)
        out.update(fid)
        np.savez(f"{OUT}/samples_{case}_{size}.npz", samples=smp)
        # online timing: single observation set
        tk1 = (torch.tensor(np.load(f"{OUT}/refbat_{case}.npz")["tokens"][:1])
               if case in ("merton", "pk") else None)
        print(f"[{case} {size}] train {t_train:.0f}s "
              f"FID median {out['fid_median']:.3f} mean {out['fid_mean']:.3f}",
              flush=True)
    else:
        prb = obj if kind != "kpp40" else obj[0]
        if kind == "zoo":
            from amortix.designs import sbc_design
            p = sbc_design(post, prb, n_sims=500, n_post=200, seed=1)
            out["sbc_p"] = [float(x) for x in p]
            out["sbc_pass"] = int(sum(x > 0.05 for x in p))
        else:
            res = run_sbc(post, prb, n_sims=500, n_post=200, seed=1)
            p = sbc_uniformity(res["ranks"], res["n_post"])
            out["sbc_p"] = [float(x) for x in p]
            out["sbc_pass"] = int(sum(x > 0.05 for x in p))
            np.savez(f"{OUT}/sbc_{case}_{size}.npz", ranks=res["ranks"],
                     n_post=res["n_post"])
        print(f"[{case} {size}] train {t_train:.0f}s "
              f"SBC pass {out['sbc_pass']}/{len(out['sbc_p'])}", flush=True)
    with open(f"{OUT}/sat_{case}_{size}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("JOB_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["refs", "run"])
    ap.add_argument("case")
    ap.add_argument("size", nargs="?", default="small")
    a = ap.parse_args()
    if a.mode == "refs":
        mode_refs(a.case)
    else:
        mode_run(a.case, a.size)
