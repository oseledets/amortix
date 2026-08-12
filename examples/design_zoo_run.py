"""Train and validate any design-amortized zoo case.

    uv run python examples/design_zoo_run.py --case pk --steps 12000
    uv run python examples/design_zoo_run.py --case merton --n_train 120000 --steps 72000

Uses the package defaults end to end: embed/rope resolve by the class rule,
training follows the canonical recipe (fresh designs each step + mix K).
Validation: SBC over mixed designs plus fixed-K buckets. For verdicts on
individual cells prefer an exact-reference probe where a tractable
likelihood exists (merton_logpost_factory / pk_logpost_factory /
kpp_logpost_factory + amortix.mcmc.metropolis); SBC at these sizes is a
screen, not the record -- see CALIBRATION.md.
"""
from __future__ import annotations

import argparse
import time

import torch

from amortix import FlowPosterior
from amortix.designs import sbc_design
from amortix.problems.design_zoo import DESIGN_ZOO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(DESIGN_ZOO))
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None,
                    help="optional path for the trained state_dict")
    args = ap.parse_args()

    prob = DESIGN_ZOO[args.case]()
    torch.manual_seed(args.seed)
    post = FlowPosterior(prob)
    print(f"[{args.case}] embed={post.embed_mode}, "
          f"params={sum(p.numel() for p in post.parameters()):,}")
    t0 = time.time()
    post.fit(n_train=args.n_train, steps=args.steps, seed=args.seed,
             retokenize=prob.make_retokenizer(), verbose=True)
    print(f"[trained in {time.time() - t0:.0f}s]")
    if args.save:
        torch.save(post.state_dict(), args.save)
        print(f"[saved] {args.save}")

    names = prob.prior.names
    p = sbc_design(post, prob, n_sims=400, n_post=200, seed=1)
    print("SBC mixed designs: "
          + "  ".join(f"{nm}:{v:.3f}" for nm, v in zip(names, p)))
    kmax = prob.observer.k_max
    for kf in [prob.k_min + 2, kmax // 8, kmax // 3, kmax - kmax // 8]:
        p = sbc_design(post, prob, n_sims=300, n_post=150, seed=100 + kf,
                       k_fixed=kf)
        print(f"SBC K={kf:>3}:        "
              + "  ".join(f"{nm}:{v:.3f}" for nm, v in zip(names, p)))


if __name__ == "__main__":
    main()
