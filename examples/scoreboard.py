"""The honest scoreboard: amortized posterior against the controls it must clear.

Reported per case (MAE as % of prior range, lower is better):

    prior-only   ignore the data, predict the prior mean. 25.00% by construction
                 for a uniform prior. A parameter scored near this carries no
                 recoverable information -- "errors" there are noise, and a
                 posterior that just returns the prior passes coverage checks.
    ridge        degree-2 ridge on summary statistics: the cheapest serious
                 attempt. Not beating it means the neural machinery earns nothing.
    amortized    our conditional flow matching posterior.
    classical    the case's `sota` estimator (note: several of these consume the
                 full fine path while the network only sees the token set, so
                 they are NOT information-matched -- see results/CRITIC_*.md).

    uv run python examples/scoreboard.py --n_train 12000 --epochs 40
"""
from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.controls import prior_mean_error, ridge_control_error, contraction
from amortix.problems import GALLERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=GALLERY)
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--n_post", type=int, default=400)
    ap.add_argument("--out", type=str, default="results/SCOREBOARD.md")
    args = ap.parse_args()
    cases = args.cases or GALLERY

    lines, rows = [], []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"honest scoreboard | budget {args.n_train}/{args.epochs} | "
         f"{args.n_test} datasets x {args.n_post} draws")
    emit(f"{'case':>14} | {'prior-only':>10} | {'ridge':>7} | {'amortized':>9} | "
         f"{'classical':>9} | verdict")
    emit("-" * 78)

    for name in cases:
        mod = importlib.import_module(f"amortix.problems.{name}")
        prob = mod.make()
        rng = (prob.prior.high - prob.prior.low).numpy()

        post = FlowPosterior(prob).fit(n_train=args.n_train, epochs=args.epochs,
                                       verbose=False)
        gen = torch.Generator().manual_seed(123)
        m_true = prob.prior.sample(args.n_test, generator=gen)
        tokens, traj = prob.observe(m_true, generator=gen)
        draws = post.sample_batch(tokens, n=args.n_post, seed=0).numpy()

        amort = (np.abs(draws.mean(1) - m_true.numpy()) / rng * 100).mean(0)
        prior_e = prior_mean_error(prob, m_true.numpy())
        ridge_e, _ = ridge_control_error(prob, n_train=args.n_train, n_test=args.n_test)
        classical = np.stack([mod.sota(tokens[i].numpy(), traj[i].numpy(), prob)
                              for i in range(args.n_test)])
        # Clip the classical estimate into the prior box. The amortized posterior
        # physically cannot leave it, so scoring an unclipped baseline is an
        # unfairness in the other direction: 19-23% of raw estimates land outside,
        # and clipping improves them by 16-21%.
        classical = np.clip(classical, prob.prior.low.numpy(), prob.prior.high.numpy())
        class_e = (np.abs(classical - m_true.numpy()) / rng * 100).mean(0)

        beats_prior = amort.mean() < prior_e.mean() - 1.0
        beats_ridge = amort.mean() < ridge_e.mean()
        verdict = ("beats ridge" if beats_ridge else
                   "loses to ridge" if beats_prior else "NO SIGNAL (~prior)")
        emit(f"{name:>14} | {prior_e.mean():9.2f}% | {ridge_e.mean():6.2f}% | "
             f"{amort.mean():8.2f}% | {class_e.mean():8.2f}% | {verdict}")
        contr = contraction(draws, prob)
        rows.append(dict(case=name, names=prob.prior.names,
                         prior=prior_e.tolist(), ridge=ridge_e.tolist(),
                         amort=amort.tolist(), classical=class_e.tolist(),
                         contraction=contr.tolist()))

    emit("-" * 78)
    n_ridge = sum(1 for r in rows if np.mean(r["amort"]) < np.mean(r["ridge"]))
    emit(f"amortized beats the ridge control in {n_ridge}/{len(rows)} cases")
    emit("")
    emit("--- per parameter: prior-only / ridge / amortized | contraction | verdict ---")
    emit("contraction = prior sd / posterior sd (1.0 = the posterior IS the prior).")
    emit("A wide posterior is NOT a failure: where the likelihood is flat the correct")
    emit("posterior is the prior. What a flat posterior does mean is that error-to-truth")
    emit("is capped at the prior's 25% and says nothing about method quality there.")
    emit("The ridge control disambiguates the two flat cases:")
    emit("  PRIOR-LIMITED - flat, and the ridge cannot do better either")
    emit("                  => near-prior is the right answer; judge it by SBC, not MAE")
    emit("  WIDTH WRONG   - flat, but the ridge locates the parameter")
    emit("                  => the true posterior is narrow and ours is not: a real error")
    no_info, we_failed = [], []
    for r in rows:
        emit(f"\n{r['case']}")
        for j, nm in enumerate(r["names"]):
            c = r["contraction"][j]
            ridge_helps = r["ridge"][j] < 0.8 * r["prior"][j]
            if c < 1.15 and not ridge_helps:
                verdict, tag = "PRIOR-LIMITED", no_info
            elif c < 1.15:
                verdict, tag = "WIDTH WRONG", we_failed
            else:
                verdict, tag = "", None
            if tag is not None:
                tag.append(f"{r['case']}.{nm}")
            emit(f"   {nm:>10}: {r['prior'][j]:6.2f}% / {r['ridge'][j]:6.2f}% / "
                 f"{r['amort'][j]:6.2f}% | {c:5.2f}x  {verdict}")
    emit("")
    if no_info:
        emit(f"prior-limited, not a defect ({len(no_info)}): " + ", ".join(no_info)
             + "  -- score these by SBC / distance to the true posterior, not by MAE")
    if we_failed:
        emit(f"POSTERIOR TOO WIDE -- the ridge locates what we leave at the prior "
             f"({len(we_failed)}): " + ", ".join(we_failed))

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Honest scoreboard\n\n```\n" + "\n".join(lines) + "\n```\n")
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(dict(n_train=args.n_train, epochs=args.epochs, rows=rows), f, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
