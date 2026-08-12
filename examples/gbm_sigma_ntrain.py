"""Does GBM's sigma bias shrink with more simulations?

sigma's posterior is ~4x sharper than mu's (contraction 5.66x vs 1.48x), so the
same absolute location error is ~4x more posterior-sd on sigma -- which is why it
alone fails SBC. Location error is known to respond to n_train, not to steps. Test
it directly: fixed step budget, vary n_train, report sigma's SBC p-value and the
sign of its rank slope (low - high; negative = sigma biased low).

    uv run python examples/gbm_sigma_ntrain.py --n_train 40000
"""
from __future__ import annotations

import argparse

import numpy as np

from amortix import FlowPosterior
from amortix.diagnostics import run_sbc, sbc_uniformity
from amortix.problems.gbm import make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()

    prob = make()
    j = prob.prior.names.index("sigma")
    post = FlowPosterior(prob).fit(n_train=args.n_train, steps=args.steps, verbose=True)
    res = run_sbc(post, prob, n_sims=500, n_post=200, seed=0)
    ranks = res["ranks"]
    p = sbc_uniformity(ranks, 200)[j]
    u = ranks[:, j] / 200.0
    lo = np.mean(u < 0.5); slope = np.mean(u) - 0.5      # >0 => ranks pile high => sigma biased LOW
    contr = ((prob.prior.high - prob.prior.low).numpy()[j] / np.sqrt(12)) / (res["std"][:, j].mean())
    print(f"\nRESULT n_train={args.n_train} steps={args.steps}: "
          f"sigma SBC-p={p:.3f}  mean_rank-0.5={slope:+.3f}  contraction={contr:.2f}x")


if __name__ == "__main__":
    main()
