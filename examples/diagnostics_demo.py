"""Calibration diagnostics demo: train a posterior, then run SBC + coverage.

    python examples/diagnostics_demo.py            # OU
    python examples/diagnostics_demo.py double_well
"""
from __future__ import annotations

import importlib
import os


from amortix import FlowPosterior, diagnose


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="ou")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    name = args.case
    mod = importlib.import_module(f"amortix.problems.{name}")
    prob = mod.make()
    print(f"training {name} posterior ({args.n_train}/{args.epochs}) ...")
    post = FlowPosterior(prob).fit(n_train=args.n_train, epochs=args.epochs, verbose=False)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sbc_{name}.png")
    diagnose(post, prob, n_sims=400, n_post=200, plot_path=out)


if __name__ == "__main__":
    main()
