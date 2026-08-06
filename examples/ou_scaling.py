"""Isolate training-headroom vs information-limit for OU recovery.

Same observation budget (74 tokens), bigger/longer-trained network. Reports, per
parameter: mean abs error (% of prior range), mean posterior std (% of range),
and 90% coverage. If error >> posterior std and both shrink with training, we
were undertrained; if error ~ posterior std and it plateaus, it's the
information limit of the data (one path), not the network.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import OrnsteinUhlenbeck, FlowPosterior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=24000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--n_post", type=int, default=1000)
    args = ap.parse_args()

    prob = OrnsteinUhlenbeck()
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()

    nparam = None
    post = FlowPosterior(prob, dim_model=args.dim, n_layer=args.layers, hidden=args.hidden)
    nparam = sum(x.numel() for x in post.parameters())
    print(f"config: n_train={args.n_train} epochs={args.epochs} dim={args.dim} "
          f"layers={args.layers} hidden={args.hidden} | params={nparam/1e3:.0f}K "
          f"| tokens={prob.observer.n_tokens}")
    post.fit(n_train=args.n_train, epochs=args.epochs)

    gen = torch.Generator().manual_seed(123)
    m_true = prob.prior.sample(args.n_test, generator=gen)
    traj = prob.simulate_paths(m_true, generator=gen)
    tokens = prob.observer.tokens_from_traj(traj)

    est = np.zeros((args.n_test, 3)); std = np.zeros((args.n_test, 3))
    lo = np.zeros((args.n_test, 3)); hi = np.zeros((args.n_test, 3))
    for i in range(args.n_test):
        s = post.sample(tokens[i], n=args.n_post, seed=i).numpy()
        est[i] = s.mean(0); std[i] = s.std(0)
        lo[i] = np.quantile(s, 0.05, 0); hi[i] = np.quantile(s, 0.95, 0)

    mt = m_true.numpy()
    err = np.abs(est - mt) / rng * 100
    pstd = std / rng * 100
    cov = ((mt >= lo) & (mt <= hi)).mean(0) * 100

    print(f"\n{'param':>6} | {'err%':>6} | {'post.std%':>9} | {'cov90':>6}")
    for j, nm in enumerate(names):
        print(f"{nm:>6} | {err[:, j].mean():5.2f}% | {pstd[:, j].mean():8.2f}% | {cov[j]:5.0f}%")
    print(f"{'ALL':>6} | {err.mean():5.2f}% | {pstd.mean():8.2f}% | {cov.mean():5.0f}%")
    print("\nref (small 2.5-min run): ALL err 7.84%, cov 93%")


if __name__ == "__main__":
    main()
