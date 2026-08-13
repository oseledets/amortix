"""OU head-to-head: amortix CFM vs sbi NPE/FMPE vs exact-likelihood MCMC.

Second system of the comparison suite (see baseline_npe.py for the GBM
protocol). Ornstein-Uhlenbeck is the deliberate control case: the process is
additive, so the floating-scale conditioning failure that breaks external
methods on GBM is absent by construction -- if the externals were failing on
GBM for any reason other than input conditioning, they would fail here too.

Reference posterior: adaptive Metropolis with the EXACT likelihood of the
generative process -- per-gap Euler-Maruyama transitions (scheme="euler",
the chain that actually produced the data) plus the stationary density of
the initial state (x0 ~ N(0, sigma^2/2theta) carries parameter information
and the network sees it, so the reference must include it). Every dataset
gets a second chain from a different start; the worst mean-discrepancy
between the two chains is reported as the reference's own validation.

    uv run --with sbi python examples/baseline_ou.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_npe import score, fmt  # noqa: E402

from amortix import FlowPosterior                       # noqa: E402
from amortix.problems import ou                         # noqa: E402
from amortix.mcmc import (observed_indices, log_likelihood_ou,  # noqa: E402
                          metropolis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--n_draw", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "auto"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save", type=str, default="results/BASELINE_OU.json")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quick:
        args.n_train, args.steps, args.n_test, args.n_draw = 2000, 600, 12, 500

    prob = ou.make()
    names = prob.prior.names
    idx = observed_indices(prob)
    dt_sim = prob.observer.dt_sim
    gaps = np.diff(idx).astype(np.float64) * dt_sim
    lo = prob.prior.low.numpy().astype(np.float64)
    hi = prob.prior.high.numpy().astype(np.float64)
    print(f"[setup] OU fixed design, {len(idx)} observed values, "
          f"device={args.device}", flush=True)

    # ---- test set ------------------------------------------------------------
    gen = torch.Generator().manual_seed(999)
    m_true = prob.prior.sample(args.n_test, gen)
    traj = prob.simulate_paths(m_true, gen)
    tokens_test = prob.observer.tokens_from_traj(traj)
    x_test = traj[:, idx, 0].contiguous()

    # ---- reference: exact-likelihood MCMC, two chains per dataset -------------
    def log_post_factory(s):
        def lp(v):
            th, sg = float(v[0]), float(v[1])
            if th <= 0 or sg <= 0:
                return -np.inf
            base = log_likelihood_ou(s, v, gaps, scheme="euler",
                                     dt_fine=dt_sim)
            var0 = sg ** 2 / (2.0 * th)
            base += -0.5 * (s[0] ** 2 / var0 + math.log(2 * math.pi * var0))
            return base
        return lp

    t0 = time.time()
    exact = []
    val = np.zeros((args.n_test, 2))
    for i in range(args.n_test):
        s = x_test[i].numpy().astype(np.float64)
        lp = log_post_factory(s)
        c1, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=args.n_draw,
                           prior_low=lo, prior_high=hi, seed=i)
        c2, _ = metropolis(lp, 0.25 * lo + 0.75 * hi, n_samples=args.n_draw,
                           prior_low=lo, prior_high=hi, seed=5000 + i)
        exact.append(np.asarray(c1))
        val[i] = [abs(c1[:, j].mean() - c2[:, j].mean())
                  / max(c1[:, j].std(), 1e-12) for j in range(2)]
    t_mcmc = (time.time() - t0) / (2 * args.n_test)
    print(f"[reference] MCMC {1e3 * t_mcmc:.0f} ms/chain; two-chain "
          f"validation worst {val.max():.3f} sd, mean {val.mean():.3f} sd",
          flush=True)

    report = dict(n_train=args.n_train, steps=args.steps, n_test=args.n_test,
                  n_draw=args.n_draw, device=args.device,
                  mcmc_ms_per_chain=1e3 * t_mcmc,
                  ref_validation_worst_sd=float(val.max()),
                  ref_validation_mean_sd=float(val.mean()))

    # ---- amortix ---------------------------------------------------------------
    torch.manual_seed(args.seed)
    post = FlowPosterior(prob)
    n_par = sum(p.numel() for p in post.parameters())
    t0 = time.time()
    post.fit(n_train=args.n_train, steps=args.steps, seed=args.seed,
             verbose=False, device=args.device)
    t_train = time.time() - t0
    t0 = time.time()
    amx = post.sample_batch(tokens_test, n=args.n_draw, seed=0).numpy()
    t_inf = time.time() - t0
    res = score(amx, exact, names)
    print(f"[amortix] {n_par:,} params, train {t_train:.0f}s, "
          f"inference {1e3 * t_inf / args.n_test:.0f} ms/dataset", flush=True)
    fmt("amortix", res, names)
    report["amortix"] = dict(params=n_par, train_s=t_train,
                             inf_ms_per_ds=1e3 * t_inf / args.n_test, **res)

    # ---- sbi arms on raw values ------------------------------------------------
    import sbi
    import sbi.inference as sbi_inf
    from sbi.utils import BoxUniform as SbiBox

    x_idx = idx[idx > 0]
    xt_test = traj[:, x_idx, 0].float().to(args.device)
    for method in ["NPE", "FMPE"]:
        sbi_prior = SbiBox(low=prob.prior.low.float(),
                           high=prob.prior.high.float(), device=args.device)
        gen2 = torch.Generator().manual_seed(args.seed + 1)
        theta = prob.prior.sample(args.n_train, gen2)
        x_tr = prob.simulate_paths(theta, gen2)[:, x_idx, 0].contiguous()
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
        res = score(smp, exact, names)
        tag = method.lower()
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
    print("DONE_OU", flush=True)


if __name__ == "__main__":
    main()
