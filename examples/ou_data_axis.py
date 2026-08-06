"""The information axis for OU recovery.

The headroom test showed theta's ~18% error is the posterior width, not model
error -- so it shrinks only with more data, not more network. Here we vary the
number of independent replicate paths and watch err and posterior std fall
(roughly ~1/sqrt(n_paths)). Same small net throughout.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortix import OrnsteinUhlenbeck, FlowPosterior

N_TRAIN, EPOCHS = 8000, 25
N_TEST, N_POST = 60, 500
PATHS = [1, 2, 4]


def evaluate(n_paths):
    prob = OrnsteinUhlenbeck(n_paths=n_paths)
    names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()
    print(f"\n### n_paths={n_paths}  tokens={prob.observer.n_tokens}")
    post = FlowPosterior(prob).fit(n_train=N_TRAIN, epochs=EPOCHS, verbose=True)

    gen = torch.Generator().manual_seed(123)          # same params across configs
    m_true = prob.prior.sample(N_TEST, generator=gen)
    tokens, _ = prob._tokens_for(m_true, generator=gen)

    est = np.zeros((N_TEST, 3)); std = np.zeros((N_TEST, 3))
    lo = np.zeros((N_TEST, 3)); hi = np.zeros((N_TEST, 3))
    for i in range(N_TEST):
        s = post.sample(tokens[i], n=N_POST, seed=i).numpy()
        est[i] = s.mean(0); std[i] = s.std(0)
        lo[i] = np.quantile(s, 0.05, 0); hi[i] = np.quantile(s, 0.95, 0)

    mt = m_true.numpy()
    err = (np.abs(est - mt) / rng * 100).mean(0)
    pstd = (std / rng * 100).mean(0)
    cov = (((mt >= lo) & (mt <= hi)).mean(0)) * 100
    return names, err, pstd, cov


def main():
    rows = []
    for k in PATHS:
        names, err, pstd, cov = evaluate(k)
        rows.append((k, err, pstd, cov))

    print("\n" + "=" * 60)
    print("INFORMATION AXIS: error & posterior std vs number of paths")
    print("=" * 60)
    print(f"{'n_paths':>7} | " + " | ".join(f"{n:>14}" for n in names))
    for k, err, pstd, cov in rows:
        cells = [f"{err[j]:5.1f}/{pstd[j]:4.1f}%" for j in range(3)]
        print(f"{k:>7} | " + " | ".join(f"{c:>14}" for c in cells))
    print("(cell = mean abs error% / posterior std%, both as % of prior range)")
    print("expectation: ~1/sqrt(n_paths) shrinkage, strongest on theta")


if __name__ == "__main__":
    main()
