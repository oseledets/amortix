"""Encoder A/B on `linear_gaussian`, judged by distance to the EXACT posterior.

The only case where the answer is known in closed form, so the only case where an
encoder change can be attributed rather than guessed at. Two measurements:

  --probe   linear probe from the encoder's pooled context / token memory to the
            exact posterior mean mu = Sigma A^T y / sigma^2. Reported as residual
            RMSE in units of the exact posterior sd, because R^2 is unreadable
            here: the prior sd is 1.73 against a posterior sd of 0.26-0.43, so
            R^2 = 0.99 already means half a posterior sd of location error.

  (default) train the full model and report energy_distance_scaled to the exact
            posterior. The metric oscillates from checkpoint to checkpoint by as
            much as the effects being chased, so the reported number averages the
            late checkpoints, and every claim uses >= 2 seeds.

    uv run python examples/encoder_ab.py --floor                        # what perfect scores
    uv run python examples/encoder_ab.py --probe
    uv run python examples/encoder_ab.py --seeds 0 1                    # today's default
    uv run python examples/encoder_ab.py --cfg '{"input_norm":false}' --seeds 0 1   # before it

Findings and the resulting defaults: results/DEBUG_encoder.md
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.encoder import SetTransformer
from amortix.metrics import energy_distance_scaled
from amortix.problems.linear_gaussian import make, exact_posterior

N_EVAL, N_DRAW = 32, 400
ENC_KEYS = ("rope", "pool", "n_query", "pool_dim", "embed", "embed_hidden",
            "input_norm", "final_norm")


def eval_set(prob, n=N_EVAL, seed=1234, n_draw=N_DRAW):
    gen = torch.Generator().manual_seed(seed)
    m = prob.prior.sample(n, generator=gen)
    tok, y = prob.observe(m, generator=gen)
    exact = [exact_posterior(y[i], prob, n=n_draw, seed=1000 + i).numpy() for i in range(n)]
    return tok, y, exact


def energy_to_exact(post, tok, exact, scale, n_draw=N_DRAW):
    d = post.sample_batch(tok, n=n_draw, seed=0).numpy()
    return float(np.mean([energy_distance_scaled(d[i], exact[i], scale)
                          for i in range(len(exact))]))


def mc_floor(prob, y, exact, scale, n_draw=N_DRAW):
    """Two independent EXACT draws: the value a perfect posterior would score."""
    return float(np.mean([
        energy_distance_scaled(
            exact_posterior(y[i], prob, n=n_draw, seed=50_000 + i).numpy(), exact[i], scale)
        for i in range(len(exact))]))


def build(prob, cfg, seed=0):
    torch.manual_seed(seed)
    post = FlowPosterior(prob)
    enc_kw = {k: v for k, v in cfg.items() if k in ENC_KEYS}
    if enc_kw:
        dim = post.encoder.dim
        post.encoder = SetTransformer(
            n_features=prob.observer.N_FEATURES, dim=dim,
            max_tokens=prob.observer.n_tokens + 8, **enc_kw)
        if post.encoder.ctx_dim != dim:
            raise SystemExit(
                f"pool_dim={post.encoder.ctx_dim} != dim_model={dim}: the base head "
                f"built by FlowPosterior expects a {dim}-wide context. Rebuild "
                f"post.base_head too if you want to try a wider pooled context.")
    if post.encoder.in_norm is not None:
        # `input_norm` keeps running statistics that training would fill in. Probing
        # an *untrained* encoder with empty statistics would measure an unnormalized
        # encoder and blame the wrong thing, so warm them up here.
        _, tk = prob.simulate(4000, generator=torch.Generator().manual_seed(seed + 7))
        post.encoder.in_norm(tk)
    return post


def probe(post, prob, n=6000, seed=99):
    """Ridge probe encoder-output -> exact posterior mean, fitted/scored on halves."""
    gen = torch.Generator().manual_seed(seed)
    m = prob.prior.sample(n, generator=gen)
    tok, y = prob.observe(m, generator=gen)
    tgt = ((y @ (prob.A / prob.noise ** 2)) @ prob.Sigma.T).numpy()
    post_sd = torch.sqrt(torch.diagonal(prob.Sigma)).numpy()
    with torch.no_grad():
        mem = post.encoder.encode(tok)
        ctx = post.encoder.pool(mem)
    out, ntr = {}, n // 2
    for name, X in (("ctx", ctx.numpy()),
                    ("memory", mem.reshape(n, -1).numpy()),
                    ("y_raw", y.numpy())):
        mx, sx = X[:ntr].mean(0), X[:ntr].std(0) + 1e-8
        Xtr = np.c_[(X[:ntr] - mx) / sx, np.ones(ntr)]
        Xte = np.c_[(X[ntr:] - mx) / sx, np.ones(n - ntr)]
        W = np.linalg.solve(Xtr.T @ Xtr + 1e-6 * ntr * np.eye(Xtr.shape[1]), Xtr.T @ tgt[:ntr])
        res = tgt[ntr:] - Xte @ W
        out[name] = dict(r2=(1 - (res ** 2).mean(0) / tgt[ntr:].var(0)).tolist(),
                         rmse_over_postsd=(np.sqrt((res ** 2).mean(0)) / post_sd).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="{}", help="JSON of SetTransformer options")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=9360)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--every", type=int, default=1500, help="checkpoint interval")
    ap.add_argument("--burnin", type=int, default=4500)
    ap.add_argument("--probe", action="store_true", help="probe only, no training")
    ap.add_argument("--floor", action="store_true", help="print the Monte-Carlo floor")
    args = ap.parse_args()
    cfg = json.loads(args.cfg)

    prob = make()
    scale = (prob.prior.high - prob.prior.low).numpy()
    tok_e, y_e, exact_e = eval_set(prob)

    if args.floor:
        print(f"Monte-Carlo floor (two independent exact draws): "
              f"{mc_floor(prob, y_e, exact_e, scale):.5f}")
        return

    print(f"cfg {cfg}")
    for s in args.seeds:
        t0 = time.time()
        post = build(prob, cfg, seed=s)
        if args.probe:                       # untrained: what the architecture carries
            pr = probe(post, prob)
        else:
            post.fit(n_train=args.n_train, steps=args.steps, seed=s, verbose=False,
                     monitor=lambda p: energy_to_exact(p, tok_e, exact_e, scale),
                     monitor_every=args.every)
            late = [h["metric"] for h in post.history if h["step"] >= args.burnin]
            late.append(energy_to_exact(post, tok_e, exact_e, scale))
            pr = probe(post, prob)
            print(f"  seed {s}  energy-to-exact {np.mean(late):.5f}   "
                  f"checkpoints {np.round(late, 4).tolist()}")
        print(f"  seed {s}  probe rmse/posterior-sd   "
              f"ctx {np.round(pr['ctx']['rmse_over_postsd'], 3).tolist()}   "
              f"memory {np.round(pr['memory']['rmse_over_postsd'], 3).tolist()}"
              f"   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
