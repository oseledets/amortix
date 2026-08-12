"""Head-to-head: amortix (CFM) vs the `sbi` package (NPE) vs per-dataset MCMC.

The one benchmark where nobody can argue about the ground truth: GBM with the
package's fixed two-channel design, scored against the *exact* closed-form
posterior on the same observed points (conjugate sampler, importance-corrected
to the box prior; validated against exact-likelihood MCMC in CALIBRATION.md).

Both learners get the same simulation budget from the same simulator and
prior, both run on the same device with their default hyperparameters, and
both are scored on the same test datasets against the same references:

  * accuracy : signed bias of the posterior mean (in exact-posterior sd)
               and width ratio (method sd / exact sd), per parameter;
  * cost     : wall-clock training time at equal budget, wall-clock
               inference time per dataset (2000 posterior samples), and --
               for scale -- exact-likelihood MCMC time per dataset.

Needs `sbi` (not a package dependency):  uv run --with sbi python examples/baseline_npe.py

    uv run --with sbi python examples/baseline_npe.py --n_train 20000 --steps 6000
    uv run --with sbi python examples/baseline_npe.py --quick        # smoke sizes
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.problems import gbm
from amortix.mcmc import observed_indices, log_likelihood_gbm, metropolis


def exact_posterior(prob, path, idx, n_samples=4000, seed=0, pool_factor=8):
    """Exact GBM posterior on the observed indices (conjugate in (b, sigma^2),
    importance-resampled to the uniform box prior in (mu, sigma))."""
    rng = np.random.default_rng(seed)
    x = np.asarray(path, dtype=np.float64)
    vals = x[np.asarray(idx, dtype=np.int64)]
    tau = np.diff(idx).astype(np.float64) * float(prob.observer.dt_sim)
    r = np.diff(np.log(vals))
    n = r.size
    T = float(tau.sum())
    R1 = float(r.sum())
    R2 = float((r ** 2 / tau).sum())
    SSw = R2 - R1 ** 2 / T
    npool = n_samples * pool_factor
    chi = rng.chisquare(max(n - 1, 1), size=npool)
    v = SSw / np.maximum(chi, 1e-300)
    b = R1 / T + np.sqrt(v / T) * rng.standard_normal(npool)
    sigma = np.sqrt(v)
    mu = b + 0.5 * v
    draws = np.stack([mu, sigma], axis=1)
    low = prob.prior.low.numpy().astype(np.float64)
    high = prob.prior.high.numpy().astype(np.float64)
    inbox = np.all((draws >= low) & (draws <= high), axis=1)
    w = sigma * inbox
    if w.sum() <= 0:
        raise RuntimeError("no conjugate draws landed in the prior box")
    pick = rng.choice(npool, size=n_samples, replace=True, p=w / w.sum())
    return draws[pick]


def score(samples, exact, names):
    """samples [B, n, d] vs exact list of [n_e, d] -> per-param bias/width."""
    B = len(exact)
    out = {}
    for j, nm in enumerate(names):
        bias = np.empty(B)
        width = np.empty(B)
        for i in range(B):
            em, es = exact[i][:, j].mean(), exact[i][:, j].std()
            bias[i] = (samples[i, :, j].mean() - em) / max(es, 1e-12)
            width[i] = samples[i, :, j].std() / max(es, 1e-12)
        out[nm] = dict(bias=float(bias.mean()),
                       bias_se=float(bias.std() / np.sqrt(B)),
                       width=float(width.mean()),
                       width_se=float(width.std() / np.sqrt(B)))
    return out


def fmt(tag, res, names):
    cells = "  ".join(
        f"{nm}: bias {res[nm]['bias']:+.3f}±{res[nm]['bias_se']:.3f} "
        f"width {res[nm]['width']:.3f}" for nm in names)
    print(f"  {tag:<10s} {cells}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_draw", type=int, default=2000)
    ap.add_argument("--n_mcmc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", type=str, default="amortix,npe,mcmc",
                    help="comma list from {amortix,npe,npe_log,npe_ret,mcmc}; npe_log "
                         "gives NPE hand log-transformed prices -- the manual "
                         "feature engineering amortix's learnable warp replaces")
    ap.add_argument("--save", type=str, default="results/BASELINE_NPE.json")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.steps, args.n_test, args.n_draw, args.n_mcmc = \
            2000, 600, 24, 500, 1

    prob = gbm.make()
    names = prob.prior.names
    idx = observed_indices(prob)
    dt_gaps = np.diff(idx).astype(np.float64) * prob.observer.dt_sim
    print(f"[setup] GBM fixed design, {len(idx)} observed grid values, "
          f"budget {args.n_train} sims", flush=True)

    # ---- shared test set + exact references --------------------------------
    gen = torch.Generator().manual_seed(999)
    m_true = prob.prior.sample(args.n_test, gen)
    traj = prob.simulate_paths(m_true, gen)                     # [B, S+1, 1]
    tokens_test = prob.observer.tokens_from_traj(traj)
    x_idx = idx[idx > 0]          # drop the S0=1 anchor: constant, carries no info
    x_test = traj[:, x_idx, 0].contiguous()                     # [B, U]
    exact = [exact_posterior(prob, traj[i, :, 0].numpy(), idx, seed=5000 + i)
             for i in range(args.n_test)]
    print("[setup] exact references ready", flush=True)

    report = dict(n_train=args.n_train, steps=args.steps, n_test=args.n_test,
                  n_draw=args.n_draw, device="cpu",
                  torch_threads=torch.get_num_threads())
    arms = set(args.arms.split(","))

    # ---- amortix CFM --------------------------------------------------------
    if "amortix" in arms:
        torch.manual_seed(args.seed)
        post = FlowPosterior(prob)
        n_par = sum(p.numel() for p in post.parameters())
        t0 = time.time()
        post.fit(n_train=args.n_train, steps=args.steps, seed=args.seed,
                 verbose=False)
        t_train_amx = time.time() - t0
        t0 = time.time()
        amx = post.sample_batch(tokens_test, n=args.n_draw, seed=0).numpy()
        t_inf_amx = time.time() - t0
        t0 = time.time()
        post.sample_batch(tokens_test[:1], n=args.n_draw, seed=1)
        t_one_amx = time.time() - t0
        res_amx = score(amx, exact, names)
        print(f"[amortix] {n_par:,} params, train {t_train_amx:.0f}s, "
              f"batch inference {1e3 * t_inf_amx / args.n_test:.0f} ms/dataset, "
              f"single dataset {1e3 * t_one_amx:.0f} ms", flush=True)
        fmt("amortix", res_amx, names)
        report["amortix"] = dict(params=n_par, train_s=t_train_amx,
                                 inf_ms_per_ds=1e3 * t_inf_amx / args.n_test,
                                 inf_ms_single=1e3 * t_one_amx, **res_amx)

    # ---- sbi NPE (raw prices and/or hand log-transformed prices) ------------
    def run_npe(tag, transform):
        import sbi
        from sbi.inference import NPE
        from sbi.utils import BoxUniform as SbiBox

        sbi_prior = SbiBox(low=prob.prior.low.float(),
                           high=prob.prior.high.float())
        gen2 = torch.Generator().manual_seed(args.seed + 1)
        theta = prob.prior.sample(args.n_train, gen2)
        x_tr = transform(prob.simulate_paths(theta, gen2)[:, x_idx, 0])
        t0 = time.time()
        inf = NPE(prior=sbi_prior)                 # package defaults throughout
        de = inf.append_simulations(theta.float(), x_tr.float()) \
                .train(show_train_summary=False)
        npe_post = inf.build_posterior(de)
        t_train = time.time() - t0
        n_par = sum(p.numel() for p in de.parameters())
        xt = transform(x_test)
        t0 = time.time()
        smp = np.stack([
            npe_post.sample((args.n_draw,), x=xt[i].float(),
                            show_progress_bars=False).numpy()
            for i in range(args.n_test)])
        t_inf = time.time() - t0
        res = score(smp, exact, names)
        print(f"[sbi {sbi.__version__} {tag}] {n_par:,} params, "
              f"train {t_train:.0f}s, "
              f"inference {1e3 * t_inf / args.n_test:.0f} ms/dataset",
              flush=True)
        fmt(tag, res, names)
        report[tag] = dict(sbi_version=sbi.__version__, params=n_par,
                           train_s=t_train,
                           inf_ms_per_ds=1e3 * t_inf / args.n_test, **res)

    def to_returns(x):
        """Per-gap log-returns anchored at log(S0)=0 -- the coordinates of the
        exact likelihood factorization, hand-delivered to the baseline."""
        xl = x.clamp_min(1e-8).log()
        return torch.diff(xl, dim=1,
                          prepend=torch.zeros(xl.shape[0], 1)).contiguous()

    if "npe" in arms:
        run_npe("npe", lambda x: x.contiguous())
    if "npe_log" in arms:
        run_npe("npe_log", lambda x: x.clamp_min(1e-8).log().contiguous())
    if "npe_ret" in arms:
        run_npe("npe_ret", to_returns)

    # ---- per-dataset exact-likelihood MCMC (cost scale only) ----------------
    if "mcmc" in arms:
        lo = prob.prior.low.numpy().astype(np.float64)
        hi = prob.prior.high.numpy().astype(np.float64)
        t_mcmc = []
        mcmc_res = []
        for i in range(args.n_mcmc):
            s = traj[i, idx, 0].numpy().astype(np.float64)

            def lp(v):
                return log_likelihood_gbm(s, v, dt_gaps)

            t0 = time.time()
            chain, _acc = metropolis(lp, 0.5 * (lo + hi),
                                     n_samples=args.n_draw,
                                     prior_low=lo, prior_high=hi, seed=i)
            t_mcmc.append(time.time() - t0)
            ch = np.asarray(chain)
            em = exact[i]
            mcmc_res.append([(ch[:, j].mean() - em[:, j].mean())
                             / max(em[:, j].std(), 1e-12) for j in range(2)])
        t_mcmc_med = float(np.median(t_mcmc))
        print(f"[MCMC] exact likelihood, {1e3 * t_mcmc_med:.0f} ms/dataset "
              f"(median of {args.n_mcmc}); sanity bias vs exact: "
              + "  ".join(f"{names[j]} {np.mean([r[j] for r in mcmc_res]):+.3f}"
                          for j in range(2)), flush=True)
        report["mcmc"] = dict(ms_per_ds=1e3 * t_mcmc_med, n=args.n_mcmc)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[saved] {args.save}", flush=True)
    print("DONE_BASELINE", flush=True)


if __name__ == "__main__":
    main()
