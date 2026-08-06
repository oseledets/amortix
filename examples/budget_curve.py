"""Is a weak result undertraining, or a ceiling? Measure, do not infer.

Comparing case A at a small budget against case B at a large one says nothing:
the cases differ. The only way to attribute a gap to budget is a learning curve on
ONE case -- if the error falls monotonically and has not flattened, more training
is the answer; if it plateaus well above the reference, something else is wrong.

Reports per budget: error of the posterior mean, posterior contraction
(prior sd / posterior sd, 1.0 = the posterior is still the prior) and the ridge
control, which does not depend on the budget and so acts as a fixed yardstick.

    uv run python examples/budget_curve.py sindy_sde
"""
from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.controls import contraction, prior_mean_error, ridge_control_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--budgets", type=int, nargs="+", default=[3000, 6000, 12000, 24000])
    ap.add_argument("--epochs", type=int, nargs="+", default=[10, 20, 35, 50])
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--n_post", type=int, default=400)
    ap.add_argument("--param", type=str, default=None,
                    help="focus on one parameter (default: report all)")
    args = ap.parse_args()
    assert len(args.budgets) == len(args.epochs)

    mod = importlib.import_module(f"amortix.problems.{args.case}")
    prob = mod.make()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()

    gen = torch.Generator().manual_seed(123)
    m_true = prob.prior.sample(args.n_test, generator=gen)
    tokens, _ = prob.observe(m_true, generator=gen)

    prior_e = prior_mean_error(prob, m_true.numpy())
    ridge_e, _ = ridge_control_error(prob, n_train=12000, n_test=args.n_test)

    lines, rows = [], []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"learning curve on '{args.case}' | {args.n_test} datasets x {args.n_post} draws")
    emit(f"fixed yardsticks -- prior-only {prior_e.mean():.2f}%, "
         f"ridge {ridge_e.mean():.2f}% (budget-independent)")
    emit("")
    emit(f"{'n_train':>8} {'epochs':>7} | {'err':>7} | {'contraction':>11} | per-parameter err")
    emit("-" * 82)
    for n_train, epochs in zip(args.budgets, args.epochs):
        post = FlowPosterior(prob).fit(n_train=n_train, epochs=epochs, verbose=False)
        draws = post.sample_batch(tokens, n=args.n_post, seed=0)
        d = draws.numpy()
        err = (np.abs(d.mean(1) - m_true.numpy()) / rng * 100).mean(0)
        contr = contraction(draws, prob)
        per = "  ".join(f"{nm}:{e:.1f}" for nm, e in zip(names, err))
        emit(f"{n_train:>8} {epochs:>7} | {err.mean():6.2f}% | {contr.mean():10.2f}x | {per}")
        rows.append(dict(n_train=n_train, epochs=epochs, err=err.tolist(),
                         contraction=contr.tolist()))
    emit("-" * 82)
    first, last = rows[0]["err"], rows[-1]["err"]
    drop = (np.mean(first) - np.mean(last)) / max(np.mean(first), 1e-9) * 100
    emit(f"error fell {drop:.0f}% from the smallest to the largest budget")
    if len(rows) >= 3:
        recent = [np.mean(r["err"]) for r in rows[-3:]]
        flat = abs(recent[-1] - recent[-2]) < 0.05 * recent[-2]
        emit("the curve has FLATTENED -- more budget is not the answer" if flat
             else "the curve is still falling -- the budget was the binding constraint")
    emit("")
    emit("per-parameter contraction across budgets (1.0 = still the prior):")
    for j, nm in enumerate(names):
        emit(f"   {nm:>10}: " + "  ".join(f"{r['contraction'][j]:.2f}x" for r in rows))

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    p = os.path.join(repo, "results", f"CURVE_{args.case}.md")
    with open(p, "w") as f:
        f.write(f"# learning curve: {args.case}\n\n```\n" + "\n".join(lines) + "\n```\n")
    with open(os.path.splitext(p)[0] + ".json", "w") as f:
        json.dump(dict(case=args.case, prior=prior_e.tolist(), ridge=ridge_e.tolist(),
                       rows=rows), f, indent=2)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
