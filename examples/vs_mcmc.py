"""Amortized CFM posterior vs the TRUE posterior, dataset by dataset (MCMC).

SBC only checks self-consistency averaged over the prior: it can say "this
posterior is mis-calibrated" but never *how*. For the two gallery cases with an
exact transition density (OU, GBM) we can compute the true posterior of one
concrete dataset by MCMC (`amortix.mcmc`) and diff the amortized posterior
against it, per dataset and per parameter:

  mean diff   |E_flow[m] - E_mcmc[m]|          as % of the prior range
  bias / std  the same shift in units of the MCMC posterior std  -> "shifted?"
  std ratio   std_flow / std_mcmc              -> ">1 too wide, <1 over-confident"
  W1 / std    1-D Wasserstein distance, normalized by the MCMC posterior std
  corr diff   mean |corr_flow - corr_mcmc| over off-diagonal entries

**The reference conditions on exactly the data the network sees.** The observer
exposes only a subsample of the fine path (OU: 73 of 501 points, GBM: 95 of 501),
so `--data observed` (default) is the matched reference. Against the full-path
posterior (`--data full`) even a perfect amortized posterior would look
over-dispersed by 2.7x on sigma (measured std ratio observed/full: OU sigma 2.73,
GBM sigma 2.54) -- so the full-path comparison is never the verdict.

    uv run python examples/vs_mcmc.py ou  --n_train 12000 --epochs 40
    uv run python examples/vs_mcmc.py gbm --n_train 12000 --epochs 40
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from amortix import FlowPosterior
from amortix.mcmc import TRACTABLE_CASES, posterior_samples


def corr_pairs(x: np.ndarray) -> np.ndarray:
    """Upper-triangle correlations of samples [n, d], one entry per parameter pair."""
    d = x.shape[1]
    if d < 2:
        return np.zeros(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(x, rowvar=False)
    return np.nan_to_num(c)[np.triu_indices(d, 1)]


def compare(a: np.ndarray, b: np.ndarray, rng_span: np.ndarray) -> dict:
    """Amortized samples `a` vs reference samples `b`, both [n, d]."""
    d = a.shape[1]
    ma, mb = a.mean(0), b.mean(0)
    sa, sb = a.std(0), b.std(0)
    sb_safe = np.maximum(sb, 1e-12)
    ca, cb = corr_pairs(a), corr_pairs(b)
    return dict(
        mean_diff_pct=np.abs(ma - mb) / rng_span * 100,
        bias_over_std=(ma - mb) / sb_safe,
        std_ratio=sa / sb_safe,
        w1_over_std=np.array([wasserstein_distance(a[:, j], b[:, j]) / sb_safe[j]
                              for j in range(d)]),
        std_a_pct=sa / rng_span * 100,
        std_b_pct=sb / rng_span * 100,
        corr_a=ca, corr_b=cb,
        corr_diff=float(np.abs(ca - cb).mean()) if d > 1 else 0.0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=list(TRACTABLE_CASES),
                    help="gallery case with a tractable likelihood")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=12, help="held-out datasets")
    ap.add_argument("--n_post", type=int, default=2000, help="amortized draws per dataset")
    ap.add_argument("--mcmc_samples", type=int, default=8000, help="MCMC draws per dataset")
    ap.add_argument("--data", choices=["observed", "full"], default="observed",
                    help="what the reference likelihood conditions on (default: the "
                         "observed subsample -- the data the network actually sees)")
    ap.add_argument("--scheme", choices=["exact", "euler"], default="exact",
                    help="OU only: continuous-time or Euler-Maruyama transition density")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    mod = importlib.import_module(f"amortix.problems.{args.case}")
    prob = mod.make()
    names = prob.prior.names
    d = prob.prior.dim
    span = (prob.prior.high - prob.prior.low).numpy()

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 96)
    emit(f"{args.case.upper()}: amortized CFM posterior vs MCMC gold standard")
    emit("=" * 96)
    emit(f"params {names} | tokens {prob.observer.n_tokens} | "
         f"path {prob.observer.n_steps} steps @ dt={prob.observer.dt_sim}")
    emit(f"budget n_train={args.n_train} epochs={args.epochs} | "
         f"{args.n_test} held-out datasets | {args.n_post} amortized draws vs "
         f"{args.mcmc_samples} MCMC draws")

    t0 = time.time()
    post = FlowPosterior(prob)                       # base='data', conditioning='xattn'
    post.fit(n_train=args.n_train, epochs=args.epochs, seed=args.seed, verbose=True)
    train_t = time.time() - t0

    gen = torch.Generator().manual_seed(1234 + args.seed)
    m_true = prob.prior.sample(args.n_test, generator=gen)
    tokens, traj = prob.observe(m_true, generator=gen)
    mt = m_true.numpy()

    chunk = max(1, min(16, 8000 // max(1, args.n_post)))
    t0 = time.time()
    draws = post.sample_batch(tokens, n=args.n_post, seed=7, chunk=chunk).numpy()
    amort_t = (time.time() - t0) / args.n_test

    # --- MCMC reference, one run per dataset --------------------------------
    # A second, independent MCMC run per dataset gives the *Monte-Carlo floor*:
    # the same metrics evaluated between two correct samplings of the very same
    # posterior, at the same sample sizes. Any amortized number at that level is
    # indistinguishable from exact.
    per, null, mcmc_means, rhats, esss, accs = [], [], [], [], [], []
    n_obs = 0
    t0 = time.time()
    for i in range(args.n_test):
        ref, info = posterior_samples(prob, traj[i].numpy(), args.case,
                                      n_samples=args.mcmc_samples, seed=9000 + i,
                                      data=args.data, scheme=args.scheme,
                                      return_info=True)
        ref2 = posterior_samples(prob, traj[i].numpy(), args.case,
                                 n_samples=args.mcmc_samples, seed=770000 + i,
                                 data=args.data, scheme=args.scheme)
        sub = ref2[::max(1, ref2.shape[0] // args.n_post)][:args.n_post]
        per.append(compare(draws[i], ref, span))
        null.append(compare(sub, ref, span))
        mcmc_means.append(ref.mean(0))
        rhats.append(info["rhat"])
        esss.append(info["ess"])
        accs.append(info["acceptance"])
        n_obs = info["n_obs"]
    mcmc_t = (time.time() - t0) / (2 * args.n_test)
    mcmc_means = np.array(mcmc_means)

    def agg(k, rows=None):
        return np.mean([p[k] for p in (rows if rows is not None else per)], axis=0)

    mean_diff = agg("mean_diff_pct")
    bias = np.mean([np.abs(p["bias_over_std"]) for p in per], axis=0)
    sr = agg("std_ratio")
    w1 = agg("w1_over_std")
    sa = agg("std_a_pct")
    sb = agg("std_b_pct")
    corr_diff = float(np.mean([p["corr_diff"] for p in per]))
    worst = np.max([p["w1_over_std"] for p in per], axis=0)

    n_mean_diff = agg("mean_diff_pct", null)
    n_bias = np.mean([np.abs(p["bias_over_std"]) for p in null], axis=0)
    n_sr = agg("std_ratio", null)
    n_w1 = agg("w1_over_std", null)
    n_corr = float(np.mean([p["corr_diff"] for p in null]))

    emit()
    emit(f"reference: exact {args.case.upper()} likelihood, data={args.data} "
         f"({n_obs} of {prob.observer.n_steps + 1} path points), scheme={args.scheme}")
    rmax, emin = float(np.nanmax(rhats)), float(np.nanmin(esss))
    emit(f"MCMC quality: max split-Rhat {rmax:.3f} | min ESS {emin:.0f} / "
         f"{args.mcmc_samples} | mean acceptance {np.mean(accs):.2f}"
         + ("   <-- WARNING: reference not converged, raise --mcmc_samples"
            if (rmax > 1.05 or emin < 400) else ""))
    emit()
    hdr = (f"{'param':>8} | {'mean diff':>9} | {'|bias|/std':>10} | {'std ratio':>9} | "
           f"{'W1/std':>7} | {'std amort':>9} | {'std MCMC':>9}")
    emit(hdr)
    emit("-" * len(hdr))
    for j, nm in enumerate(names):
        emit(f"{nm:>8} | {mean_diff[j]:8.2f}% | {bias[j]:10.2f} | {sr[j]:9.3f} | "
             f"{w1[j]:7.3f} | {sa[j]:8.2f}% | {sb[j]:8.2f}%")
    emit("-" * len(hdr))
    emit(f"{'ALL':>8} | {mean_diff.mean():8.2f}% | {bias.mean():10.2f} | "
         f"{sr.mean():9.3f} | {w1.mean():7.3f} | {sa.mean():8.2f}% | {sb.mean():8.2f}%")
    emit(f"{'MC floor':>8} | {n_mean_diff.mean():8.2f}% | {n_bias.mean():10.2f} | "
         f"{n_sr.mean():9.3f} | {n_w1.mean():7.3f} |  (two independent MCMC runs "
         f"of the SAME posterior)")
    emit()
    emit(f"correlation-matrix difference (mean |d corr| over off-diagonals): "
         f"{corr_diff:.3f}   (MC floor {n_corr:.3f})")
    if d > 1:
        ca, cb = agg("corr_a"), agg("corr_b")
        cn = agg("corr_a", null)
        emit("per pair (mean over datasets)  -- a pair whose amortized posterior is "
             "degenerate makes its correlation meaningless:")
        for k, (i, j) in enumerate(zip(*np.triu_indices(d, 1))):
            emit(f"    corr({names[i]},{names[j]}): amortized {ca[k]:+.3f} | "
                 f"MCMC {cb[k]:+.3f} | diff {abs(ca[k]-cb[k]):.3f} | "
                 f"MC floor {abs(cn[k]-cb[k]):.3f}")
    emit("worst-dataset W1/std per param: " +
         "  ".join(f"{nm}:{worst[j]:.2f}" for j, nm in enumerate(names)))
    emit("targets: mean diff -> 0, |bias|/std -> 0, std ratio -> 1.000, W1/std -> 0, "
         "corr diff -> 0;")
    emit("the MC floor row is what those metrics read when BOTH sample sets are "
         "exact -- that is the resolution limit.")
    emit("(std ratio > 1 = too wide/under-confident, < 1 = over-confident; "
         "mean diff and stds are % of prior range)")

    # --- anchor: how far is each posterior mean from the truth? -------------
    rmse_a = np.sqrt(((draws.mean(1) - mt) ** 2).mean(0)) / span * 100
    rmse_m = np.sqrt(((mcmc_means - mt) ** 2).mean(0)) / span * 100
    emit()
    emit("anchor -- RMSE of the posterior mean to the true parameter (% of prior range):")
    emit(f"  amortized {'  '.join(f'{nm}:{rmse_a[j]:.2f}%' for j, nm in enumerate(names))}"
         f"   (mean {rmse_a.mean():.2f}%)")
    emit(f"  MCMC      {'  '.join(f'{nm}:{rmse_m[j]:.2f}%' for j, nm in enumerate(names))}"
         f"   (mean {rmse_m.mean():.2f}%)")

    # --- OU only: the simulator starts every path at X_0 = mu ---------------
    leak = None
    if args.case == "ou":
        x0 = np.array([float(traj[i].numpy().reshape(-1)[0]) for i in range(args.n_test)])
        jmu = names.index("mu")
        amort_gap = np.abs(draws[:, :, jmu].mean(1) - x0) / span[jmu] * 100
        amort_std = draws[:, :, jmu].std(1) / span[jmu] * 100
        mcmc_std = np.array([p["std_b_pct"][jmu] for p in per])
        emit()
        emit("note -- OU initial condition: the simulator sets X_0 = mu exactly, so the "
             "data pins mu.")
        emit("The conditional likelihood above deliberately ignores that (as the "
             "closed-form MLE does),")
        emit("so the reference posterior for mu is artificially broad. Measured:")
        emit(f"  amortized |E[mu] - X_0| = {amort_gap.mean():.2f}% of prior range, "
             f"amortized std(mu) = {amort_std.mean():.2f}%, MCMC std(mu) = {mcmc_std.mean():.2f}%")
        leak = dict(amort_gap_pct=amort_gap.tolist(), amort_std_pct=amort_std.tolist(),
                    mcmc_std_pct=mcmc_std.tolist())

        # reference B: the exact posterior of the true generative model, mu pinned
        perB = []
        for i in range(args.n_test):
            refB = posterior_samples(prob, traj[i].numpy(), "ou",
                                     n_samples=args.mcmc_samples, seed=9000 + i,
                                     data=args.data, scheme=args.scheme,
                                     fixed={"mu": x0[i]})
            perB.append(compare(draws[i], refB, span))
        srB = np.mean([p["std_ratio"] for p in perB], axis=0)
        w1B = np.mean([p["w1_over_std"] for p in perB], axis=0)
        mdB = np.mean([p["mean_diff_pct"] for p in perB], axis=0)
        emit("  reference B (mu pinned at X_0 -- the exact posterior of the actual "
             "generative model):")
        for j, nm in enumerate(names):
            if j == jmu:
                continue
            emit(f"    {nm:>6}: mean diff {mdB[j]:.2f}%  std ratio {srB[j]:.3f}  "
                 f"W1/std {w1B[j]:.3f}")
        leak["reference_B"] = dict(mean_diff_pct=mdB.tolist(), std_ratio=srB.tolist(),
                                   w1_over_std=w1B.tolist())

    emit()
    emit(f"timing: train {train_t:.0f}s once | amortized {amort_t*1e3:.0f} ms/dataset "
         f"({args.n_post} draws) | MCMC {mcmc_t*1e3:.0f} ms/dataset "
         f"({args.mcmc_samples} draws)")
    emit("(MCMC is cheap here precisely because the likelihood is closed-form -- these "
         "two cases exist to")
    emit(" validate the machinery, not to be beaten on speed. The amortization pays "
         "off where no such likelihood exists.)")

    # --- persist ------------------------------------------------------------
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    md_path = args.out or os.path.join(repo, "results", f"VS_MCMC_{args.case}{suffix}.md")
    if not os.path.isabs(md_path):
        md_path = os.path.join(repo, md_path)
    json_path = os.path.splitext(md_path)[0] + ".json"

    with open(md_path, "w") as f:
        f.write(f"# {args.case} — amortized CFM posterior vs MCMC gold standard\n\n"
                "Per held-out dataset, the amortized posterior is compared with MCMC "
                "samples from the exact posterior conditioned on **the same data the "
                "network sees** (the observer's subsample of the path).\n\n```\n"
                + "\n".join(lines) + "\n```\n")
    payload = dict(
        case=args.case,
        config=dict(n_train=args.n_train, epochs=args.epochs, n_test=args.n_test,
                    n_post=args.n_post, mcmc_samples=args.mcmc_samples,
                    data=args.data, scheme=args.scheme, seed=args.seed,
                    n_obs_points=int(n_obs), n_path_points=int(prob.observer.n_steps + 1)),
        names=names,
        summary=dict(mean_diff_pct=mean_diff.tolist(), abs_bias_over_std=bias.tolist(),
                     std_ratio=sr.tolist(), w1_over_std=w1.tolist(),
                     std_amort_pct=sa.tolist(), std_mcmc_pct=sb.tolist(),
                     corr_diff=corr_diff, worst_w1_over_std=worst.tolist(),
                     rmse_to_truth_amortized_pct=rmse_a.tolist(),
                     rmse_to_truth_mcmc_pct=rmse_m.tolist()),
        mc_floor=dict(mean_diff_pct=n_mean_diff.tolist(),
                      abs_bias_over_std=n_bias.tolist(), std_ratio=n_sr.tolist(),
                      w1_over_std=n_w1.tolist(), corr_diff=n_corr),
        mcmc_quality=dict(max_rhat=float(np.nanmax(rhats)), min_ess=float(np.nanmin(esss)),
                          mean_acceptance=float(np.mean(accs))),
        per_dataset=[{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in p.items()} for p in per],
        ou_initial_condition=leak,
        timing=dict(train_s=train_t, amortized_ms=amort_t * 1e3, mcmc_ms=mcmc_t * 1e3),
    )
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {md_path}\n[saved] {json_path}")


if __name__ == "__main__":
    main()
