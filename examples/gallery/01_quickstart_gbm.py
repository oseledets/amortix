"""Quickstart: recover GBM parameters and check against the exact posterior.

The whole interface is in this file: the problem is a prior box, a simulator,
and an observation grid, written out below. A tiny design-amortized posterior
trains in minutes on a laptop CPU; its samples for one observation set are
then compared with the exact conjugate posterior on the same points, which
serves as the referee and is imported from the package.

Run:  python examples/gallery/01_quickstart_gbm.py
      --png          also render docs/media/quickstart_gbm.png
      --ckpt PATH    load the checkpoint if it exists, else train and save it
"""
import argparse
import math
import os

import torch

from amortix.designs import DesignObserver, DesignProblem
from amortix.evaluation import fid, load_posterior, model_of_size
from amortix.prior import BoxUniform
from amortix.problems.design_basic import gbm_exact_from_points

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "docs", "media")


class GBM(DesignProblem):
    """dS = mu S dt + sigma S dW, S0 = 1, observed at arbitrary times."""


    def __init__(self):
        self.prior = BoxUniform(low=[-0.20, 0.10], high=[0.40, 0.60],
                                names=["mu", "sigma"])
        self.observer = DesignObserver(dt_sim=0.01, n_steps=500, k_max=128)
        self.k_min = 2

    def trajectories(self, m, generator=None):
        B = m.shape[0]
        dt = self.observer.dt_sim
        mu, sigma = m[:, 0], m[:, 1]
        logS = torch.zeros(B)
        out = torch.zeros(B, self.observer.n_steps + 1, 1)
        out[:, 0, 0] = 1.0
        for i in range(self.observer.n_steps):
            z = torch.randn(B, generator=generator)
            logS = logS + (mu - 0.5 * sigma ** 2) * dt \
                + sigma * math.sqrt(dt) * z
            out[:, i + 1, 0] = torch.exp(logS)
        return out


def get_posterior(prob, ckpt=None):
    if ckpt and os.path.exists(ckpt):
        return load_posterior(prob, ckpt)
    post = model_of_size(prob, "tiny")
    post.fit(n_train=3000, steps=1200, batch=256,
             retokenize=prob.make_retokenizer(), verbose=True)
    if ckpt:
        torch.save(post.state_dict(), ckpt)
    return post


def render_png(prob, m_true, raw, tidx, draws, exact, f, path):
    import matplotlib.pyplot as plt

    from amortix.plotting import (BLUE, DPI, FIGSIZE, GREY, ORANGE, hdr_contours,
                        param_axes, save_figure)

    fig, (al, ar) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    t_grid = torch.arange(raw.shape[1]) * prob.observer.dt_sim
    al.plot(t_grid.numpy(), raw[0, :, 0].numpy(), color=GREY, lw=1.0)
    al.plot(tidx.numpy() * prob.observer.dt_sim, raw[0, tidx, 0].numpy(),
            "o", color=BLUE, ms=4)
    al.set_xlabel("t")
    al.set_ylabel("S(t)")
    al.set_title("one observation set (K = 20)", fontsize=10)

    param_axes(ar, prob)
    hdr_contours(ar, exact, ORANGE)
    ar.scatter(draws[:, 0], draws[:, 1], s=3, alpha=0.25, color=BLUE,
               linewidths=0)
    ar.plot(*m_true[0].numpy(), "x", color="black", ms=8, mew=1.6)
    ar.set_title(f"amortized posterior, FID vs exact {f:.4f}", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", action="store_true",
                    help="render docs/media/quickstart_gbm.png")
    ap.add_argument("--ckpt", metavar="PATH",
                    help="load this checkpoint if it exists; otherwise train "
                         "and save one here")
    args = ap.parse_args(argv)

    prob = GBM()
    post = get_posterior(prob, args.ckpt)

    # one fresh observation set: 20 points at random times
    gen = torch.Generator().manual_seed(1)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)
    tidx, cidx = prob.sample_design(gen, 20)
    tokens = prob.tokens_for(raw[0], tidx, cidx, gen)

    draws = post.sample(tokens, n=2000)                  # milliseconds
    exact = gbm_exact_from_points(prob, raw[0, :, 0], tidx, n_samples=2000)

    f = fid(draws.numpy(), exact)
    print(f"\ntrue parameters : {m_true[0].tolist()}")
    print(f"posterior mean  : {draws.mean(0).tolist()}")
    print(f"FID vs exact    : {f:.4f} "
          f"(estimator floor at n=2000 is ~0.004)")

    if args.png:
        render_png(prob, m_true, raw, tidx, draws.numpy(), exact, f,
                   os.path.join(MEDIA, "quickstart_gbm.png"))


if __name__ == "__main__":
    main()
