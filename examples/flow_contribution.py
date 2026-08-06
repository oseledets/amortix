"""How much work does the flow actually do?

The data-dependent base already predicts the posterior's mean (and, with
base="full", its covariance). If the flow adds nothing on top, we are really
doing amortized Gaussian regression with a decorative ODE -- which would also
mean any SBC "win" came from Gaussianizing the posterior, exactly the wrong
thing for multimodal problems.

This measures it directly: compare the base alone (skip the ODE) against the
full flow, on accuracy, calibration, and non-Gaussianity of the samples.

    uv run python examples/flow_contribution.py cir --base data
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.diagnostics import coverage_from_ranks, sbc_uniformity, calibration_error


@torch.no_grad()
def base_only(post, tokens, n, seed=0, chunk=16):
    """Samples straight from the learned base, without integrating the ODE."""
    gen = torch.Generator().manual_seed(seed)
    outs = []
    for s in range(0, tokens.shape[0], chunk):
        tb = tokens[s:s + chunk]
        B = tb.shape[0]
        ctx = post.encoder.pool(post.encoder.encode(tb))
        eps = torch.randn(B, n, post.d, generator=gen)
        if post.base == "full":
            mu, L = post.base_head(ctx)
            z = mu[:, None] + torch.einsum("bij,bnj->bni", L, eps)
        elif post.base_head is not None:
            mu, sd = post.base_head(ctx)
            z = mu[:, None] + sd[:, None] * eps
        else:
            z = eps
        outs.append(post.prior.denormalize(z))
    return torch.cat(outs, 0)


def score(draws, m_true, rng, tag, prior=None):
    s = draws.numpy(); mt = m_true.numpy()
    ranks = (s < mt[:, None, :]).sum(1)
    n_post = s.shape[1]
    err = (np.abs(s.mean(1) - mt) / rng * 100).mean()
    ce = calibration_error(ranks, n_post)
    p = sbc_uniformity(ranks, n_post)
    # Non-Gaussianity must be measured in the flow's own (normalized) space: the
    # probit denormalization is nonlinear, so a Gaussian base looks skewed in
    # parameter space. In z-space, base-only samples are Gaussian by construction,
    # so any skew/kurtosis there is what the FLOW added.
    zz = prior.normalize(torch.as_tensor(s).reshape(-1, s.shape[-1])).numpy()
    zz = zz.reshape(s.shape)
    zz = (zz - zz.mean(1, keepdims=True)) / (zz.std(1, keepdims=True) + 1e-9)
    skew = np.abs((zz ** 3).mean(1)).mean()
    kurt = np.abs((zz ** 4).mean(1) - 3.0).mean()
    print(f"{tag:>12} | {err:6.2f}% | {ce:6.1f}pp | {int((p > 0.05).sum())}/{len(p)} | "
          f"{skew:6.3f} | {kurt:6.3f}")
    return dict(err=err, ce=ce, npass=int((p > 0.05).sum()), skew=skew, kurt=kurt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--base", default="data", choices=["standard", "data", "full"])
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n_sims", type=int, default=200)
    ap.add_argument("--n_post", type=int, default=200)
    args = ap.parse_args()

    mod = importlib.import_module(f"amortix.problems.{args.case}")
    prob = mod.make()
    rng = (prob.prior.high - prob.prior.low).numpy()
    post = FlowPosterior(prob, base=args.base)
    post.fit(n_train=args.n_train, epochs=args.epochs, verbose=False)

    gen = torch.Generator().manual_seed(7)
    m_true = prob.prior.sample(args.n_sims, generator=gen)
    tokens, _ = prob.observe(m_true, generator=gen)

    print(f"\n{args.case}  base={args.base}  ({args.n_train}/{args.epochs}, "
          f"SBC {args.n_sims}x{args.n_post})")
    print(f"{'variant':>12} | {'err':>7} | {'calib':>8} | SBC | {'|skew|':>6} | {'|kurt|':>6}")
    print("-" * 62)
    b = score(base_only(post, tokens, args.n_post), m_true, rng, "base only", prob.prior)
    f = score(post.sample_batch(tokens, n=args.n_post, seed=0), m_true, rng,
              "base + flow", prob.prior)
    # control: the same flow forced to do ALL the transport from a plain N(0,I)
    ctl = FlowPosterior(prob, base="standard")
    ctl.fit(n_train=args.n_train, epochs=args.epochs, verbose=False)
    c = score(ctl.sample_batch(tokens, n=args.n_post, seed=0), m_true, rng,
              "flow only", prob.prior)
    print("-" * 62)
    print(f"flow contribution: err {b['err']-f['err']:+.2f}pp, "
          f"calib {b['ce']-f['ce']:+.1f}pp, SBC {f['npass']-b['npass']:+d} params")
    print("(|skew|,|kurt| measure departure from Gaussian: if the flow leaves them at ~0\n"
          " the posterior is effectively Gaussian and the ODE is decorative.)")


if __name__ == "__main__":
    main()
