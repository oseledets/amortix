"""A/B: concat (pooled vector) vs xattn (dense cross-attention) conditioning.

Tests whether dense cross-attention conditioning fixes the residual sigma
shape/skew mis-calibration (and helps weakly-identified params), by SBC on a
chosen case at a converged budget. Saves COND_<case>_RESULTS.md (+ .json).

    uv run python examples/test_conditioning.py gbm
    uv run python examples/test_conditioning.py fhn --n_train 12000 --epochs 40
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import FlowPosterior
from amortix.diagnostics import run_sbc, coverage_from_ranks, sbc_uniformity

LEVELS = (0.5, 0.9, 0.95)


def calib_error(ranks, n_post):
    cov = coverage_from_ranks(ranks, n_post, LEVELS)
    return float(np.mean([np.abs(cov[q] - q) for q in LEVELS])) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="gbm")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_sims", type=int, default=300)
    ap.add_argument("--n_post", type=int, default=150)
    ap.add_argument("--modes", nargs="+", default=["concat", "xattn"])
    ap.add_argument("--dim", type=int, default=64, help="dim_model (network capacity)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="repeat each mode over these seeds and aggregate (controls seed noise)")
    ap.add_argument("--tag", type=str, default="", help="suffix for the output filename")
    args = ap.parse_args()

    mod = importlib.import_module(f"amortix.problems.{args.case}")
    out_lines, payload = [], []

    def emit(line=""):
        print(line, flush=True)
        out_lines.append(line)

    emit(f"conditioning A/B on '{args.case}'  (budget {args.n_train}/{args.epochs}, "
         f"dim={args.dim}, SBC {args.n_sims}x{args.n_post})")
    for cond in args.modes:
        P, CE, names, nparam = [], [], None, 0
        for sd in args.seeds:
            prob = mod.make()
            post = FlowPosterior(prob, conditioning=cond, dim_model=args.dim)
            nparam = sum(p.numel() for p in post.parameters())
            post.fit(n_train=args.n_train, epochs=args.epochs, seed=sd, verbose=False)
            res = run_sbc(post, prob, n_sims=args.n_sims, n_post=args.n_post, seed=sd)
            P.append(sbc_uniformity(res["ranks"], args.n_post))
            CE.append(calib_error(res["ranks"], args.n_post))
            names = res["names"]
        P = np.stack(P)                                  # [n_seeds, d]
        meanp = P.mean(0)
        n_seeds = len(args.seeds)
        passed = (P > 0.05).sum(0)                        # per param: #seeds passing
        ce_mean, ce_std = float(np.mean(CE)), float(np.std(CE))
        emit("")
        emit(f"[{cond}]  params={nparam/1e3:.0f}K  seeds={n_seeds}  "
             f"calib-err={ce_mean:.1f}±{ce_std:.1f}pp  "
             f"mean SBC-pass={(passed/n_seeds).sum():.1f}/{len(names)}")
        for j, nm in enumerate(names):
            emit(f"   {nm:>8}: mean SBC-p={meanp[j]:.3f}  passed {int(passed[j])}/{n_seeds} seeds")
        payload.append(dict(cond=cond, calib_err_mean=ce_mean, calib_err_std=ce_std,
                            names=names, mean_sbc_p=[float(x) for x in meanp],
                            seeds_passed=[int(x) for x in passed], n_seeds=n_seeds))

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    suffix = f"_{args.tag}" if args.tag else ""
    md = os.path.join(repo, f"COND_{args.case}{suffix}_RESULTS.md")
    with open(md, "w") as f:
        f.write(f"# conditioning A/B: {args.case}\n\n```\n" + "\n".join(out_lines) + "\n```\n")
    with open(os.path.splitext(md)[0] + ".json", "w") as f:
        json.dump({"case": args.case, "n_train": args.n_train, "epochs": args.epochs,
                   "results": payload}, f, indent=2)
    print(f"\n[saved] {md}\n[saved] {os.path.splitext(md)[0] + '.json'}")


if __name__ == "__main__":
    main()
