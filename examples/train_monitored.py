"""Train while watching the only quantity that means anything: the distance
between the sampled posterior and the reference posterior.

On `linear_gaussian` the reference is exact, so this is a true convergence curve
in full dimension -- not a loss (which is dominated by irreducible noise and can
rise while the posterior improves) and not a marginal width (which is blind to
shape and dependence).

    uv run python examples/train_monitored.py --steps 12000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.metrics import energy_distance_scaled
from amortix.problems.linear_gaussian import make, exact_posterior

N_REF, N_DRAW = 12, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--every", type=int, default=500)
    args = ap.parse_args()

    prob = make()
    scale = (prob.prior.high - prob.prior.low).numpy()

    gen = torch.Generator().manual_seed(77)
    m_ref = prob.prior.sample(N_REF, generator=gen)
    tok_ref, y_ref = prob.observe(m_ref, generator=gen)
    exact = [exact_posterior(y_ref[i], prob, n=N_DRAW, seed=i).numpy() for i in range(N_REF)]

    def monitor(post):
        d = post.sample_batch(tok_ref, n=N_DRAW, seed=0).numpy()
        return float(np.mean([energy_distance_scaled(d[i], exact[i], scale)
                              for i in range(N_REF)]))

    print(f"linear-Gaussian, exact reference | n_train={args.n_train}, "
          f"batch={args.batch}, target {args.steps} steps")
    print("watching: energy distance to the exact posterior (0 = identical)\n")
    post = FlowPosterior(prob)
    post.fit(n_train=args.n_train, steps=args.steps, batch=args.batch,
             monitor=monitor, monitor_every=args.every, verbose=True)

    h = post.history
    print("\nstep     dist-to-exact")
    for r in h:
        print(f"{r['step']:6d}   {r['metric']:.4f}")
    if len(h) >= 4:
        # Judge the trend over the last third, not two adjacent points: early on
        # the curve moves slowly and any two neighbours look flat, which would
        # declare convergence while the distance is still far from zero.
        k = max(2, len(h) // 3)
        rate = (h[-k]["metric"] - h[-1]["metric"]) / max(h[-k]["metric"], 1e-9) * 100
        print(f"\nimprovement over the last {k} checks: {rate:.1f}%")
        print("converged" if rate < 1.0 else "still improving -- more steps would help")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    with open(os.path.join(repo, "results", "MONITORED_linear_gaussian.json"), "w") as f:
        json.dump(dict(n_train=args.n_train, batch=args.batch, history=h), f, indent=2)


if __name__ == "__main__":
    main()
