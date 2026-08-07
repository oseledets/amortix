"""Does the calibration fix generalize? SBC across the whole gallery.

Trains every case with the improved configuration (attention pooling +
data-dependent base + adequate budget) and runs SBC. Reports, per case, the
calibration error (mean |empirical - nominal| coverage) and how many parameters
pass the SBC uniformity test (p > 0.05). The prior config (mean-pool + standard
base) failed SBC even on OU, so this is the generalization check of the fix.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np


from amortix import FlowPosterior
from amortix.problems import GALLERY
from amortix.diagnostics import (run_sbc, coverage_from_ranks, sbc_uniformity,
                                 calibration_error)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=GALLERY,
                    help="case names (default: all gallery cases)")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--steps", type=int, default=None,
                    help="optimizer steps -- the honest budget unit")
    ap.add_argument("--n_sims", type=int, default=300)
    ap.add_argument("--n_post", type=int, default=150)
    ap.add_argument("--conditioning", type=str, default="xattn",
                    choices=["concat", "xattn"],
                    help="how the velocity is conditioned on the data (xattn = dense)")
    ap.add_argument("--out", type=str, default="results/CALIB_GALLERY_RESULTS.md",
                    help="results file (markdown; a .json sibling is also written)")
    args = ap.parse_args()
    cases = args.cases if args.cases else GALLERY

    print("=" * 84)
    print(f"GALLERY CALIBRATION  --  current defaults, conditioning={args.conditioning}, SBC")
    budget = f"{args.steps} steps" if args.steps else f"{args.epochs} epochs"
    print(f"cases={list(cases)} | {args.n_train} sims, {budget}, "
          f"SBC {args.n_sims}x{args.n_post}")
    print("=" * 84)
    rows = []
    for name in cases:
        print(f"... {name}", flush=True)
        mod = importlib.import_module(f"amortix.problems.{name}")
        prob = mod.make()
        post = FlowPosterior(prob, conditioning=args.conditioning)
        post.fit(n_train=args.n_train, epochs=args.epochs, steps=args.steps,
                 verbose=False)
        res = run_sbc(post, prob, n_sims=args.n_sims, n_post=args.n_post, seed=0)
        ranks = res["ranks"]
        pvals = sbc_uniformity(ranks, args.n_post)
        ce = calibration_error(ranks, args.n_post)
        cov50 = coverage_from_ranks(ranks, args.n_post, (0.5,))[0.5]
        cov90 = coverage_from_ranks(ranks, args.n_post, (0.9,))[0.9]
        rows.append(dict(name=name, d=prob.prior.dim, names=res["names"],
                         ce=ce, npass=int((pvals > 0.05).sum()), pvals=pvals,
                         cov50=cov50.mean()*100, cov90=cov90.mean()*100))

    out_lines = []

    def emit(line=""):
        print(line)
        out_lines.append(line)

    hdr = f"{'case':>12} | {'dim':>3} | {'calib-err':>9} | {'SBC pass':>9} | {'mean cov50':>10} | {'mean cov90':>10}"
    emit(); emit(hdr); emit("-" * len(hdr))
    for r in rows:
        emit(f"{r['name']:>12} | {r['d']:>3} | {r['ce']:8.1f}pp |"
             f" {r['npass']}/{r['d']:<7} | {r['cov50']:9.0f}% | {r['cov90']:9.0f}%")
    emit("-" * len(hdr))
    tot_p = sum(r['npass'] for r in rows); tot_d = sum(r['d'] for r in rows)
    emit(f"SBC-pass parameters: {tot_p}/{tot_d}  |  "
         f"mean calib-err: {np.mean([r['ce'] for r in rows]):.1f}pp  "
         f"(target: low err, cov50~50%, cov90~90%)")
    emit()
    emit("--- per-parameter SBC-p (p>0.05 = calibrated) ---")
    for r in rows:
        cells = "  ".join(f"{nm}:{p:.2f}" for nm, p in zip(r['names'], r['pvals']))
        emit(f"  {r['name']:>12}: {cells}")

    # --- persist results to disk (repo root by default) ------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    header = (f"# Gallery calibration (attn-pool + data-base)\n\n"
              f"cases={list(cases)} | n_train={args.n_train} epochs={args.epochs} "
              f"SBC {args.n_sims}x{args.n_post}\n\n```\n")
    with open(md_path, "w") as f:
        f.write(header + "\n".join(out_lines) + "\n```\n")
    json_path = os.path.splitext(md_path)[0] + ".json"
    payload = {"config": {"cases": list(cases), "n_train": args.n_train,
                          "epochs": args.epochs, "n_sims": args.n_sims,
                          "n_post": args.n_post},
               "rows": [{"name": r["name"], "dim": r["d"], "names": r["names"],
                         "calib_err_pp": r["ce"], "sbc_pass": r["npass"],
                         "cov50": r["cov50"], "cov90": r["cov90"],
                         "sbc_p": [float(p) for p in r["pvals"]]} for r in rows]}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {md_path}\n[saved] {json_path}")


if __name__ == "__main__":
    main()
