"""Fisher-KPP head-to-head: the case where the reference itself is expensive.

Third system of the comparison suite (see baseline_npe.py for the protocol).
On GBM the exact-likelihood MCMC reference costs 42 ms per dataset, which
invites the question of why one would amortize at all. Here every likelihood
evaluation is a full PDE solve, so the same adaptive-Metropolis reference
costs minutes per dataset -- amortization changes from a convenience into
the only way to process many datasets. The reference is still computed (that
is what makes the accuracy columns possible), with a second chain per
dataset for validation.

Design: one fixed draw of K=40 random (time, sensor) observation points,
shared by all datasets and all methods. amortix trains on this design with
fresh observation noise each optimizer step; sbi arms receive the 40 noisy
values as a vector.

    uv run --with sbi python examples/baseline_kpp.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_npe import score, fmt  # noqa: E402

from amortix import FlowPosterior                          # noqa: E402
from amortix.problems.design_zoo import (FisherKPPDesign,  # noqa: E402
                                         kpp_logpost_factory)
from amortix.mcmc import metropolis                        # noqa: E402

K_OBS = 40
DESIGN_SEED = 4242


def _fixed_design():
    prob = FisherKPPDesign()
    gen = torch.Generator().manual_seed(DESIGN_SEED)
    tidx, cidx = prob.sample_design(gen, K_OBS)
    return prob, tidx, cidx


def _tokens_from_y(prob, tidx, cidx, y):
    obs = prob.observer
    t = tidx.float() * obs.dt_sim / obs.horizon
    z = torch.zeros_like(y)
    kf = torch.full_like(y, math.log(K_OBS) / math.log(obs.k_max))
    return torch.stack([t, y, z, z, kf, cidx.float()], dim=-1)


def _mcmc_worker(job):
    """Two chains for one dataset; module-level so it pickles."""
    i, y_obs, n_draw = job
    prob, tidx, cidx = _fixed_design()
    lp = kpp_logpost_factory(prob, tidx.numpy(), cidx.numpy(), y_obs)
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    t0 = time.time()
    c1, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=n_draw,
                       prior_low=lo, prior_high=hi, seed=i)
    c2, _ = metropolis(lp, 0.25 * lo + 0.75 * hi, n_samples=n_draw,
                       prior_low=lo, prior_high=hi, seed=5000 + i)
    dt = time.time() - t0
    disc = [abs(c1[:, j].mean() - c2[:, j].mean())
            / max(c1[:, j].std(), 1e-12) for j in range(2)]
    return i, np.asarray(c1), disc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--n_test", type=int, default=32)
    ap.add_argument("--n_draw", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "auto"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save", type=str, default="results/BASELINE_KPP.json")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quick:
        args.n_train, args.steps, args.n_test, args.n_draw = 500, 400, 3, 300

    prob, tidx, cidx = _fixed_design()
    names = prob.prior.names
    print(f"[setup] Fisher-KPP, fixed K={K_OBS} (time,sensor) design, "
          f"device={args.device}", flush=True)

    # ---- test set --------------------------------------------------------------
    gen = torch.Generator().manual_seed(999)
    m_true = prob.prior.sample(args.n_test, gen)
    raw = prob.trajectories(m_true)
    gen_noise = torch.Generator().manual_seed(777)
    y_test = raw[:, tidx, :].gather(
        2, cidx.view(1, -1, 1).expand(args.n_test, -1, 1)).squeeze(-1)
    y_test = y_test + prob.obs_noise * torch.randn(y_test.shape,
                                                   generator=gen_noise)
    tokens_test = torch.stack(
        [_tokens_from_y(prob, tidx, cidx, y_test[i])
         for i in range(args.n_test)])

    # ---- reference: PDE-likelihood MCMC (parallel, 2 chains/dataset) -----------
    t0 = time.time()
    jobs = [(i, y_test[i].numpy().astype(np.float64), args.n_draw)
            for i in range(args.n_test)]
    exact = [None] * args.n_test
    val = np.zeros((args.n_test, 2))
    times = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, chain, disc, dt in ex.map(_mcmc_worker, jobs):
            exact[i] = chain
            val[i] = disc
            times.append(dt)
    t_ref_per_ds = float(np.median(times))
    print(f"[reference] PDE-MCMC {t_ref_per_ds:.0f} s/dataset (2 chains, "
          f"median; wall {time.time()-t0:.0f}s at {args.workers} workers); "
          f"validation worst {val.max():.3f} sd, mean {val.mean():.3f} sd",
          flush=True)

    dump = os.environ.get("AMX_DUMP", "")
    if dump:
        os.makedirs(dump, exist_ok=True)
        np.savez(f"{dump}/ref_kpp.npz", samples=np.stack(exact))
    report = dict(n_train=args.n_train, steps=args.steps, n_test=args.n_test,
                  n_draw=args.n_draw, k_obs=K_OBS, device=args.device,
                  mcmc_s_per_ds_two_chains=t_ref_per_ds,
                  ref_validation_worst_sd=float(val.max()),
                  ref_validation_mean_sd=float(val.mean()))

    # ---- amortix: fixed design, fresh noise each step ---------------------------
    gen_retok = torch.Generator().manual_seed(31337)

    def retok_fixed(raw_b, _g):
        B = raw_b.shape[0]
        tokens = torch.zeros(B, K_OBS, 6)
        for i in range(B):
            tokens[i] = prob.tokens_for(raw_b[i], tidx, cidx, gen_retok)
        return tokens, torch.ones(B, K_OBS, dtype=torch.bool)

    torch.manual_seed(args.seed)
    post = FlowPosterior(prob)
    n_par = sum(p.numel() for p in post.parameters())
    t0 = time.time()
    post.fit(n_train=args.n_train, steps=args.steps, seed=args.seed,
             verbose=False, device=args.device, retokenize=retok_fixed)
    t_train = time.time() - t0
    t0 = time.time()
    amx = post.sample_batch(tokens_test, n=args.n_draw, seed=0).numpy()
    t_inf = time.time() - t0
    if dump:
        np.savez(f"{dump}/kpp_amortix.npz", samples=amx)
    res = score(amx, exact, names)
    print(f"[amortix] {n_par:,} params, train {t_train:.0f}s, "
          f"inference {1e3 * t_inf / args.n_test:.0f} ms/dataset", flush=True)
    fmt("amortix", res, names)
    report["amortix"] = dict(params=n_par, train_s=t_train,
                             inf_ms_per_ds=1e3 * t_inf / args.n_test, **res)

    # ---- sbi arms on the 40 observed values -------------------------------------
    import sbi
    import sbi.inference as sbi_inf
    from sbi.utils import BoxUniform as SbiBox

    xt_test = y_test.float().to(args.device)
    for method in ["NPE", "FMPE"]:
        sbi_prior = SbiBox(low=prob.prior.low.float(),
                           high=prob.prior.high.float(), device=args.device)
        gen2 = torch.Generator().manual_seed(args.seed + 1)
        theta = prob.prior.sample(args.n_train, gen2)
        raw_tr = prob.trajectories(theta)
        x_tr = raw_tr[:, tidx, :].gather(
            2, cidx.view(1, -1, 1).expand(args.n_train, -1, 1)).squeeze(-1)
        x_tr = x_tr + prob.obs_noise * torch.randn(x_tr.shape,
                                                   generator=gen2)
        t0 = time.time()
        inf = getattr(sbi_inf, method)(prior=sbi_prior, device=args.device)
        de = inf.append_simulations(theta.float(), x_tr.float()) \
                .train(show_train_summary=False)
        arm_post = inf.build_posterior(de)
        t_train = time.time() - t0
        t0 = time.time()
        smp = np.stack([
            arm_post.sample((args.n_draw,), x=xt_test[i],
                            show_progress_bars=False).cpu().numpy()
            for i in range(args.n_test)])
        t_inf = time.time() - t0
        tag = method.lower()
        if dump:
            np.savez(f"{dump}/kpp_{tag}.npz", samples=smp)
        res = score(smp, exact, names)
        print(f"[sbi {sbi.__version__} {tag}] train {t_train:.0f}s, "
              f"inference {1e3 * t_inf / args.n_test:.0f} ms/dataset",
              flush=True)
        fmt(tag, res, names)
        report[tag] = dict(sbi_version=sbi.__version__, train_s=t_train,
                           inf_ms_per_ds=1e3 * t_inf / args.n_test, **res)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[saved] {args.save}", flush=True)
    print("DONE_KPP", flush=True)


if __name__ == "__main__":
    main()
