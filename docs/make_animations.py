"""Generate the README animations (docs/transport.gif, docs/design.gif).

Trains a small design-amortized posterior for geometric Brownian motion
(about 15 minutes on a laptop CPU; the checkpoint is cached in
docs/_gbm_readme.pt, so re-rendering is fast), then renders two animations:

  transport.gif   flow-matching transport: posterior sampling is one ODE
                  solve, shown frame by frame -- 1,500 samples carried from
                  N(0, I) to the posterior by the learned velocity field,
                  against the exact posterior for the same points;
  design.gif      design amortization: the same trained network queried with
                  3..100 observation points of one path; the posterior
                  tightens as points are added, tracking the exact posterior
                  recomputed for every design.

Run:  uv run python docs/make_animations.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import gaussian_kde

from amortix.evaluation import load_posterior, model_of_size
from amortix.problems.design_basic import GBMDesign, gbm_exact_from_points

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_gbm_readme.pt")

BLUE, ORANGE, GREY = "#3b6bd6", "#d97706", "#8a8f98"
DPI = 100


def get_posterior(prob):
    if os.path.exists(CKPT):
        return load_posterior(prob, CKPT, device="cpu")
    post = model_of_size(prob, "tiny")
    post.fit(n_train=8000, steps=3000, batch=256,
             retokenize=prob.make_retokenizer(), verbose=True)
    torch.save(post.state_dict(), CKPT)
    return post


def hdr_contours(ax, pts, color, levels=(0.9, 0.5)):
    """Contours enclosing `levels` of the probability mass, by KDE."""
    kde = gaussian_kde(pts.T)
    dens = kde(pts.T)
    cut = sorted(np.quantile(dens, 1.0 - np.asarray(levels)))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xg, yg = np.meshgrid(np.linspace(x0, x1, 140), np.linspace(y0, y1, 140))
    z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    ax.contour(xg, yg, z, levels=cut, colors=color, linewidths=1.4)


def param_axes(ax, prob, pad=0.06):
    lo, hi = prob.prior.low.numpy(), prob.prior.high.numpy()
    span = hi - lo
    ax.set_xlim(lo[0] - pad * span[0], hi[0] + pad * span[0])
    ax.set_ylim(lo[1] - pad * span[1], hi[1] + pad * span[1])
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")


def transport_frames(post, tokens, n=1500, n_steps=48, seed=0):
    """States of the sampling ODE after every midpoint step, denormalized."""
    with torch.no_grad():
        tb = tokens[None]
        memory = post.encoder.encode(tb, None)
        ctx = post.encoder.pool(memory, None)
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(1, n, post.d, generator=gen)
        if post.base_head is not None:
            mu, s = post.base_head(ctx)
            z = mu[:, None] + s[:, None] * z
        cond = (post.velocity.encode_memory(memory)
                if post.conditioning == "xattn" else ctx)
        grouped = post.velocity.forward_grouped
        dt = 1.0 / n_steps
        out = [post.prior.denormalize(z[0]).numpy()]
        for i in range(n_steps):
            t = torch.full((1,), i * dt)
            k1 = grouped(z, t, cond, None)
            z = z + dt * grouped(z + 0.5 * dt * k1, t + 0.5 * dt, cond, None)
            out.append(post.prior.denormalize(z[0]).numpy())
    return out


def make_transport_gif(prob, post, path):
    gen = torch.Generator().manual_seed(5)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)
    tidx, cidx = prob.sample_design(gen, 20)
    tokens = prob.tokens_for(raw[0], tidx, cidx, gen)

    frames = transport_frames(post, tokens)
    exact = gbm_exact_from_points(prob, raw[0, :, 0], tidx, n_samples=6000)

    fig, (al, ar) = plt.subplots(1, 2, figsize=(8.2, 3.8), dpi=DPI)
    t_grid = np.arange(raw.shape[1]) * prob.observer.dt_sim
    al.plot(t_grid, raw[0, :, 0].numpy(), color=GREY, lw=1.0)
    al.plot(tidx.numpy() * prob.observer.dt_sim, raw[0, tidx, 0].numpy(),
            "o", color=BLUE, ms=4)
    al.set_xlabel("t")
    al.set_ylabel("S(t)")
    al.set_title("one observation set (K = 20)", fontsize=10)

    param_axes(ar, prob)
    hdr_contours(ar, exact, ORANGE)
    sc = ar.scatter(frames[0][:, 0], frames[0][:, 1], s=3, alpha=0.25,
                    color=BLUE, linewidths=0)
    ar.plot(*m_true[0].numpy(), "x", color="black", ms=8, mew=1.6)
    label = ar.set_title("", fontsize=10)
    fig.tight_layout()

    seq = [0] * 6 + list(range(len(frames))) + [len(frames) - 1] * 14

    def draw(j):
        i = seq[j]
        sc.set_offsets(frames[i])
        label.set_text(
            rf"samples at $\tau$ = {i / (len(frames) - 1):.2f}"
            "   (contours: exact posterior)")
        return sc,

    FuncAnimation(fig, draw, frames=len(seq)).save(
        path, writer=PillowWriter(fps=18))
    plt.close(fig)


def make_design_gif(prob, post, path):
    gen = torch.Generator().manual_seed(5)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)

    perm = torch.randperm(prob.observer.n_steps, generator=gen) + 1
    ks = [3, 4, 5, 7, 9, 12, 16, 20, 26, 34, 44, 58, 76, 100]

    fig, (al, ar) = plt.subplots(1, 2, figsize=(8.2, 3.8), dpi=DPI)
    t_grid = np.arange(raw.shape[1]) * prob.observer.dt_sim
    al.plot(t_grid, raw[0, :, 0].numpy(), color=GREY, lw=1.0)
    pts, = al.plot([], [], "o", color=BLUE, ms=4)
    al.set_xlabel("t")
    al.set_ylabel("S(t)")
    tl = al.set_title("", fontsize=10)

    param_axes(ar, prob)
    truth, = ar.plot(*m_true[0].numpy(), "x", color="black", ms=8, mew=1.6)
    ar.set_title("posterior from the same network", fontsize=10)
    fig.tight_layout()

    dyn = []

    def draw(j):
        k = ks[min(j, len(ks) - 1)]
        tidx = torch.sort(perm[:k]).values
        cidx = torch.zeros(k, dtype=torch.long)
        tokens = prob.tokens_for(raw[0], tidx, cidx,
                                 torch.Generator().manual_seed(7))
        draws = post.sample(tokens, n=1200, seed=0).numpy()
        exact = gbm_exact_from_points(prob, raw[0, :, 0], tidx,
                                      n_samples=6000)
        for art in dyn:
            art.remove()
        dyn.clear()
        pts.set_data(tidx.numpy() * prob.observer.dt_sim,
                     raw[0, tidx, 0].numpy())
        tl.set_text(f"K = {k} observation points")
        dyn.append(ar.scatter(draws[:, 0], draws[:, 1], s=3, alpha=0.25,
                              color=BLUE, linewidths=0))
        kde = gaussian_kde(exact.T)
        dens = kde(exact.T)
        cut = sorted(np.quantile(dens, [0.1, 0.5]))
        x0, x1 = ar.get_xlim()
        y0, y1 = ar.get_ylim()
        xg, yg = np.meshgrid(np.linspace(x0, x1, 140),
                             np.linspace(y0, y1, 140))
        z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
        dyn.append(ar.contour(xg, yg, z, levels=cut, colors=ORANGE,
                              linewidths=1.4))
        return pts,

    seq = list(range(len(ks))) + [len(ks) - 1] * 2
    FuncAnimation(fig, draw, frames=len(seq)).save(
        path, writer=PillowWriter(fps=1.6))
    plt.close(fig)


if __name__ == "__main__":
    torch.manual_seed(0)
    prob = GBMDesign()
    post = get_posterior(prob)
    post.eval()
    make_transport_gif(prob, post, os.path.join(HERE, "transport.gif"))
    print("transport.gif done")
    make_design_gif(prob, post, os.path.join(HERE, "design.gif"))
    print("design.gif done")
