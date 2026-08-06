"""Calibration A/B: what actually flattens the SBC for OU?

Compares encoder pooling (mean vs attention) and training budget, scoring each
config by SBC uniformity p-values and a calibration error (mean |empirical -
nominal| coverage). Higher SBC-p and lower calib-err = better calibrated.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import OrnsteinUhlenbeck, FlowPosterior
from amortix.diagnostics import run_sbc, coverage_from_ranks, sbc_uniformity

LEVELS = (0.5, 0.9, 0.95)
CONFIGS = [
    dict(tag="standard-base attn 8k/25", base="standard", pool="attn",
         dim=64, layer=3, hidden=256, n_train=8000, epochs=25),
    dict(tag="data-base     attn 8k/25", base="data", pool="attn",
         dim=64, layer=3, hidden=256, n_train=8000, epochs=25),
    dict(tag="data-base     attn 12k/40", base="data", pool="attn",
         dim=64, layer=3, hidden=256, n_train=12000, epochs=40),
]


def calib_error(ranks, n_post):
    cov = coverage_from_ranks(ranks, n_post, LEVELS)        # per-level per-param
    errs = [abs(cov[q] - q) for q in LEVELS]
    return float(np.mean(errs)) * 100                       # % points


def main():
    prob = OrnsteinUhlenbeck()
    names = prob.prior.names
    rows = []
    for cfg in CONFIGS:
        print(f"... training [{cfg['tag']}]", flush=True)
        post = FlowPosterior(prob, dim_model=cfg["dim"], n_layer=cfg["layer"],
                             hidden=cfg["hidden"], pool=cfg["pool"], base=cfg["base"])
        post.fit(n_train=cfg["n_train"], epochs=cfg["epochs"], verbose=False)
        res = run_sbc(post, prob, n_sims=400, n_post=200, seed=0)
        ranks = res["ranks"]
        pvals = sbc_uniformity(ranks, 200)
        cov50 = coverage_from_ranks(ranks, 200, (0.5,))[0.5]
        cov90 = coverage_from_ranks(ranks, 200, (0.9,))[0.9]
        rows.append(dict(tag=cfg["tag"], pvals=pvals, cov50=cov50, cov90=cov90,
                         ce=calib_error(ranks, 200)))

    print("\n" + "=" * 78)
    print("CALIBRATION A/B (OU, 400x200 SBC)  -- higher SBC-p, lower calib-err = better")
    print("=" * 78)
    for r in rows:
        print(f"\n[{r['tag']}]   calib-err={r['ce']:.1f}pp   "
              f"min SBC-p={r['pvals'].min():.3f}")
        for j, nm in enumerate(names):
            print(f"   {nm:>7}: cov50={r['cov50'][j]*100:3.0f}%  cov90={r['cov90'][j]*100:3.0f}%"
                  f"  SBC-p={r['pvals'][j]:.3f}")

    print("\nsummary (calibration error, percentage points; lower better):")
    for r in rows:
        print(f"   {r['tag']:>24}: {r['ce']:5.1f}pp")


if __name__ == "__main__":
    main()
