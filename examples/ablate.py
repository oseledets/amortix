"""Controlled ablation: vary ONE design axis, hold the budget fixed, average over seeds.

    uv run python examples/ablate.py cir --axis base --values data full --seeds 0 1 2
    uv run python examples/ablate.py gbm --axis conditioning --values concat xattn

Scores each variant by SBC (strict rank-uniformity, averaged over seeds) and by
calibration error. Seed averaging matters: a single-seed difference of a few
SBC-passes is usually noise. Saves results/ABL_<case>_<axis>.md (+ .json).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np

from amortix import FlowPosterior
from amortix.diagnostics import run_sbc, sbc_uniformity, calibration_error

AXES = {"base": ["standard", "data", "full"],
        "conditioning": ["concat", "xattn"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--axis", default="base", choices=list(AXES))
    ap.add_argument("--values", nargs="+", default=None,
                    help=f"variants to compare (defaults per axis: {AXES})")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_sims", type=int, default=400)
    ap.add_argument("--n_post", type=int, default=200)
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()
    values = args.values or AXES[args.axis]

    mod = importlib.import_module(f"amortix.problems.{args.case}")
    out_lines, payload = [], []

    def emit(line=""):
        print(line, flush=True)
        out_lines.append(line)

    emit(f"ablation on '{args.case}' over {args.axis}={values}")
    emit(f"budget {args.n_train}/{args.epochs}, dim={args.dim}, "
         f"SBC {args.n_sims}x{args.n_post}, seeds={args.seeds}")

    for val in values:
        P, CE, names = [], [], None
        for sd in args.seeds:
            prob = mod.make()
            post = FlowPosterior(prob, dim_model=args.dim, **{args.axis: val})
            post.fit(n_train=args.n_train, epochs=args.epochs, seed=sd, verbose=False)
            res = run_sbc(post, prob, n_sims=args.n_sims, n_post=args.n_post, seed=sd)
            P.append(sbc_uniformity(res["ranks"], args.n_post))
            CE.append(calibration_error(res["ranks"], args.n_post))
            names = res["names"]
        P = np.stack(P)
        passed = (P > 0.05).sum(0)
        ce_m, ce_s = float(np.mean(CE)), float(np.std(CE))
        emit("")
        emit(f"[{args.axis}={val}]  calib-err={ce_m:.1f}±{ce_s:.1f}pp  "
             f"mean SBC-pass={(passed / len(args.seeds)).sum():.1f}/{len(names)}")
        for j, nm in enumerate(names):
            emit(f"   {nm:>9}: mean SBC-p={P[:, j].mean():.3f}  "
                 f"passed {int(passed[j])}/{len(args.seeds)} seeds")
        payload.append(dict(value=val, calib_err_mean=ce_m, calib_err_std=ce_s,
                            names=names, mean_sbc_p=[float(x) for x in P.mean(0)],
                            seeds_passed=[int(x) for x in passed]))

    best = min(payload, key=lambda r: r["calib_err_mean"])
    emit("")
    emit(f"=> lowest calib-err: {args.axis}={best['value']} ({best['calib_err_mean']:.1f}pp)")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    md = os.path.join(repo, "results", f"ABL_{args.case}_{args.axis}.md")
    with open(md, "w") as f:
        f.write(f"# ablation: {args.case} / {args.axis}\n\n```\n" + "\n".join(out_lines) + "\n```\n")
    with open(os.path.splitext(md)[0] + ".json", "w") as f:
        json.dump(dict(case=args.case, axis=args.axis, n_train=args.n_train,
                       epochs=args.epochs, seeds=args.seeds, results=payload), f, indent=2)
    print(f"\n[saved] {md}")


if __name__ == "__main__":
    main()
