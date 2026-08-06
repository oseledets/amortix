"""Quick test of the sigma/diffusion hypothesis on GBM.

Hypothesis: sigma fails SBC because the fast channel is too narrow (few
high-frequency increments -> noisy quadratic variation). Same budget, vary only
the fast-channel width; if sigma's SBC-p rises with more fast tokens, confirmed.

    uv run python examples/test_sigma_fast.py                 # fast=60 vs 240
    uv run python examples/test_sigma_fast.py --fast 60 120 240 480

Writes SIGMA_FAST_RESULTS.md (+ .json) at the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import FlowPosterior, PathObserver, Channel
from amortix.problems.gbm import GeometricBrownianMotion, DT_SIM, N_STEPS
from amortix.diagnostics import run_sbc, coverage_from_ranks, sbc_uniformity


def run(fast_count, n_train, epochs, n_sims, n_post):
    prob = GeometricBrownianMotion()
    prob.observer = PathObserver(
        dt_sim=DT_SIM, n_steps=N_STEPS,
        channels=[Channel(every=1, count=fast_count, label="fast"),
                  Channel(every=10, count=40, label="slow")],
        n_paths=1, obs_dims=(0,),
    )
    post = FlowPosterior(prob, pool="attn", base="data")
    post.fit(n_train=n_train, epochs=epochs, verbose=False)
    res = run_sbc(post, prob, n_sims=n_sims, n_post=n_post, seed=0)
    ranks = res["ranks"]
    p = sbc_uniformity(ranks, n_post)
    cov50 = coverage_from_ranks(ranks, n_post, (0.5,))[0.5]
    return prob.observer.n_tokens, dict(zip(res["names"], zip(p, cov50)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=int, nargs="+", default=[60, 240])
    # 12k/40 is the converged regime: at 12k/40 GBM's mu already passes SBC while
    # sigma fails, so varying the fast channel isolates the sigma effect cleanly.
    # (At 8k/30 the whole posterior is under-converged -- mu fails too -- which
    # masks any fast-channel effect; don't test the hypothesis there.)
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_sims", type=int, default=250)
    ap.add_argument("--n_post", type=int, default=120)
    ap.add_argument("--out", type=str, default="SIGMA_FAST_RESULTS.md")
    args = ap.parse_args()

    out_lines, rows = [], []

    def emit(line=""):
        print(line, flush=True)
        out_lines.append(line)

    emit(f"GBM sigma/fast-channel test (budget {args.n_train}/{args.epochs}, "
         f"SBC {args.n_sims}x{args.n_post})")
    emit(f"{'fast':>6} | {'tokens':>6} | {'mu (p / cov50)':>20} | {'sigma (p / cov50)':>20}")
    for fc in args.fast:
        ntok, d = run(fc, args.n_train, args.epochs, args.n_sims, args.n_post)
        mp, mc = d["mu"]; sp, sc = d["sigma"]
        emit(f"{fc:>6} | {ntok:>6} | {mp:8.3f} / {mc*100:4.0f}%      "
             f"| {sp:8.3f} / {sc*100:4.0f}%")
        rows.append(dict(fast=fc, tokens=ntok, mu_p=float(mp), mu_cov50=float(mc),
                         sigma_p=float(sp), sigma_cov50=float(sc)))
    emit("")
    emit("hypothesis confirmed if sigma SBC-p rises (and cov50 -> ~50%) with more "
         "fast tokens, while mu stays ~unchanged")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    with open(md_path, "w") as f:
        f.write("# GBM sigma / fast-channel SBC test\n\n```\n" + "\n".join(out_lines) + "\n```\n")
    json_path = os.path.splitext(md_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump({"config": {"n_train": args.n_train, "epochs": args.epochs,
                              "n_sims": args.n_sims, "n_post": args.n_post},
                   "rows": rows}, f, indent=2)
    print(f"\n[saved] {md_path}\n[saved] {json_path}")


if __name__ == "__main__":
    main()
