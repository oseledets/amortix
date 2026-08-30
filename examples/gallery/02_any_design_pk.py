"""Design amortization: one network answers any number of blood draws.

Pharmacokinetics, written out below: an oral one-compartment Bateman curve
with log-normal assay noise, the real-world archetype of irregular designs
(blood draws). Train once, then query the SAME network at 6, 20, and 50
irregular sampling times: the posterior tightens as the design densifies,
without retraining or a fixed grid. A few minutes on GPU, longer on CPU.

Run:  python examples/gallery/02_any_design_pk.py
      --png          also render docs/media/pk_design.png
      --gif          also render docs/media/pk_design.gif (the K sweep)
      --ckpt PATH    load the checkpoint if it exists, else train and save it
"""
import argparse
import os

import torch

from amortix.designs import DesignObserver, DesignProblem, tokens_from_data
from amortix.evaluation import load_posterior, model_of_size
from amortix.prior import BoxUniform

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "docs", "media")


class PK(DesignProblem):
    """C(t) = D ka / (V (ka - ke)) (e^{-ke t} - e^{-ka t}), noisy assays."""

    DOSE = 500.0
    LOGSD = 0.10                # multiplicative log-normal measurement noise

    def __init__(self):
        self.prior = BoxUniform(low=[0.5, 0.05, 20.0],
                                high=[4.0, 0.40, 100.0],
                                names=["ka", "ke", "V"])
        self.observer = DesignObserver(dt_sim=24.0 / 500.0, n_steps=500,
                                       k_max=64)
        self.k_min = 3

    def trajectories(self, m, generator=None):
        tg = torch.arange(self.observer.n_steps + 1,
                          dtype=torch.float32) * self.observer.dt_sim
        ka, ke, V = m[:, 0:1], m[:, 1:2], m[:, 2:3]
        c = (self.DOSE * ka / (V * (ka - ke))
             * (torch.exp(-ke * tg[None]) - torch.exp(-ka * tg[None])))
        return c[..., None]


def get_posterior(prob, ckpt=None):
    if ckpt and os.path.exists(ckpt):
        return load_posterior(prob, ckpt)
    post = model_of_size(prob, "small")
    post.fit(n_train=20000, steps=6000, batch=256,
             retokenize=prob.make_retokenizer(), verbose=True)
    if ckpt:
        torch.save(post.state_dict(), ckpt)
    return post


def render_png(prob, m_true, clouds, path):
    import matplotlib.pyplot as plt

    from amortix.plotting import BLUE, DPI, FIGSIZE, param_axes, save_figure

    name0 = prob.prior.names[0]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    for ax, (K, d) in zip(axes, clouds):
        param_axes(ax, prob)
        ax.scatter(d[:, 0], d[:, 1], s=3, alpha=0.25, color=BLUE,
                   linewidths=0)
        ax.plot(*m_true[0, :2].numpy(), "x", color="black", ms=8, mew=1.6)
        ax.set_title(f"K = {K}", fontsize=10)
        ax.text(0.03, 0.97, f"sd({name0}) = {d[:, 0].std():.3f}",
                transform=ax.transAxes, va="top", fontsize=9)
    fig.suptitle("same network, same path -- three designs", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path)


def render_gif(prob, post, m_true, raw, path):
    """The K sweep as an animation: one network, 3 to 64 observation points.

    The permutation of sampling times and the assay noise are drawn once, so
    a design of size K is the first K entries of the same noisy record and
    the frames differ only in how much of it the network sees. tokens_from_data
    builds the tokens, through the same entry point that serves measured data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from amortix.plotting import (BLUE, DPI, FIGSIZE, GREY, param_axes,
                                  save_figure)

    obs = prob.observer
    gen = torch.Generator().manual_seed(11)
    perm = torch.randperm(obs.n_steps, generator=gen) + 1
    times = perm.float() * obs.dt_sim
    noise = torch.randn(obs.n_steps, generator=gen)
    y = raw[0, perm, 0] * torch.exp(prob.LOGSD * noise)   # draw the assay noise once

    ks = [3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64]
    fig, (al, ar) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    t_grid = torch.arange(raw.shape[1]) * obs.dt_sim
    al.plot(t_grid.numpy(), raw[0, :, 0].numpy(), color=GREY, lw=1.0)
    pts, = al.plot([], [], "o", color=BLUE, ms=4)
    al.set_xlabel("t, hours")
    al.set_ylabel("concentration")
    tl = al.set_title("", fontsize=10)

    param_axes(ar, prob)
    ar.plot(*m_true[0, :2].numpy(), "x", color="black", ms=8, mew=1.6)
    ar.set_title("posterior from the same network", fontsize=10)
    sc = ar.scatter([], [], s=3, alpha=0.25, color=BLUE, linewidths=0)
    fig.tight_layout()

    def draw(j):
        k = ks[min(j, len(ks) - 1)]
        tokens = tokens_from_data(prob, times[:k], y[:k])
        d = post.sample(tokens, n=1200, seed=0).numpy()
        pts.set_data(times[:k].numpy(), y[:k].numpy())
        tl.set_text(f"K = {k} blood draws")
        sc.set_offsets(d[:, :2])
        return pts, sc

    FuncAnimation(fig, draw, frames=len(ks) + 2).save(
        path, writer=PillowWriter(fps=1.6))
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", action="store_true",
                    help="render docs/media/pk_design.png")
    ap.add_argument("--gif", action="store_true",
                    help="render docs/media/pk_design.gif (the K sweep)")
    ap.add_argument("--ckpt", metavar="PATH",
                    help="load this checkpoint if it exists; otherwise train "
                         "and save one here")
    args = ap.parse_args(argv)

    prob = PK()
    post = get_posterior(prob, args.ckpt)

    gen = torch.Generator().manual_seed(7)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)
    print(f"\ntrue (ka, ke, V): {m_true[0].tolist()}")
    clouds = []
    for K in (6, 20, 50):
        tidx, cidx = prob.sample_design(gen, K)
        tokens = prob.tokens_for(raw[0], tidx, cidx, gen)
        d = post.sample(tokens, n=2000)
        lo, hi = d.quantile(0.05, 0), d.quantile(0.95, 0)
        inside = bool(((m_true[0] >= lo) & (m_true[0] <= hi)).all())
        print(f"K={K:>3}: posterior sd {d.std(0).tolist()}"
              f"   truth in 90% intervals: {inside}")
        clouds.append((K, d.numpy()))

    if args.png:
        render_png(prob, m_true, clouds, os.path.join(MEDIA, "pk_design.png"))
    if args.gif:
        render_gif(prob, post, m_true, raw,
                   os.path.join(MEDIA, "pk_design.gif"))


if __name__ == "__main__":
    main()
