"""Why does GBM's sigma fail SBC while being accurate?

Diagnose the *shape* of the sigma posterior, not its centre. Two views:
  1. the SBC rank histogram for sigma (no reference needed) -- a slope means bias,
     a U means over-confident, a peak means over-dispersed, a monotone pile at one
     end means skew;
  2. amortix vs the MCMC reference per dataset: width ratio (amortix/mcmc) and the
     skewness of each sigma marginal.

    uv run python examples/debug_gbm_sigma.py --steps 6000
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import skew

from amortix import FlowPosterior
from amortix.diagnostics import run_sbc
from amortix.mcmc import posterior_samples
from amortix.problems.gbm import make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n_mcmc", type=int, default=20)
    args = ap.parse_args()

    prob = make()
    names = prob.prior.names          # ['mu', 'sigma']
    j = names.index("sigma")
    post = FlowPosterior(prob).fit(n_train=40000, steps=args.steps, verbose=True)

    # --- 1. SBC rank histogram for sigma ---------------------------------
    res = run_sbc(post, prob, n_sims=400, n_post=200, seed=0)
    u = res["ranks"][:, j] / 200.0
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    exp = len(u) / 10
    print(f"\nsigma rank histogram (flat≈{exp:.0f}/bin if calibrated):")
    for b, c in enumerate(counts):
        bar = "#" * int(round(40 * c / max(counts.max(), 1)))
        print(f"  [{b/10:.1f}-{(b+1)/10:.1f}] {c:4d} {bar}")
    lo, hi = counts[0] + counts[1], counts[-1] + counts[-2]
    mid = counts[4] + counts[5]
    print(f"  edges(low {lo}, high {hi}) vs middle {mid}: "
          f"{'U -> over-confident' if lo+hi > 1.5*mid else 'peak -> over-dispersed' if mid > 1.5*(lo+hi)/1 else 'slope/skew' if abs(lo-hi) > 0.4*(lo+hi) else 'flat-ish'}")

    # --- 2. amortix vs MCMC sigma marginal, per dataset ------------------
    gen = torch.Generator().manual_seed(321)
    m_true = prob.prior.sample(args.n_mcmc, generator=gen)
    tok, traj = prob.observe(m_true, generator=gen)
    ours = post.sample_batch(tok, n=2000, seed=0).numpy()

    wr, sk_o, sk_m, shift = [], [], [], []
    for i in range(args.n_mcmc):
        mc = posterior_samples(prob, traj[i].numpy(), "gbm", n_samples=3000, seed=i)
        o, m = ours[i][:, j], np.asarray(mc)[:, j]
        wr.append(o.std() / (m.std() + 1e-12))
        sk_o.append(skew(o)); sk_m.append(skew(m))
        shift.append((o.mean() - m.mean()) / (m.std() + 1e-12))
    print(f"\nsigma marginal, amortix vs MCMC ({args.n_mcmc} datasets):")
    print(f"  width ratio  amortix/mcmc : {np.mean(wr):.3f}  (1 = right; <1 over-confident)")
    print(f"  skewness     amortix      : {np.mean(sk_o):+.3f}")
    print(f"  skewness     mcmc (truth) : {np.mean(sk_m):+.3f}")
    print(f"  centre shift / mcmc sd    : {np.mean(np.abs(shift)):.3f}")
    print("\nreading: sd ratio far from 1 => width; |skew_ours - skew_mcmc| large => "
          "the flow makes sigma too symmetric.")


if __name__ == "__main__":
    main()
