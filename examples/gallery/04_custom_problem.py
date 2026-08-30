"""Add your own system in ~25 lines: a damped oscillator, observed anywhere.

Subclass DesignProblem, provide a prior and a trajectories() simulator, and
everything else -- fresh-design training, tokenization, any-K inference --
comes from the base class. This is the assembled script of the README
section "Your own ODE, step by step". A few minutes on CPU.

Run:  python examples/gallery/04_custom_problem.py
      --png          also render docs/media/custom_oscillator.png
      --ckpt PATH    load the checkpoint if it exists, else train and save it
"""
import argparse
import os

import torch

from amortix.designs import DesignObserver, DesignProblem
from amortix.evaluation import load_posterior, model_of_size
from amortix.prior import BoxUniform

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "docs", "media")


class DampedOscillator(DesignProblem):
    obs_noise = 0.05

    def __init__(self):
        self.prior = BoxUniform(low=[0.5, 0.05], high=[3.0, 0.5],
                                names=["omega", "gamma"])
        self.observer = DesignObserver(dt_sim=0.05, n_steps=400, k_max=64)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        om, ga = m[:, 0], m[:, 1]
        x = torch.ones(m.shape[0]); v = torch.zeros(m.shape[0])
        out = torch.zeros(m.shape[0], 401, 1); out[:, 0, 0] = x
        dt = self.observer.dt_sim
        for i in range(400):
            a = -(om ** 2) * x - 2.0 * ga * v
            v = v + dt * a
            x = x + dt * v
            out[:, i + 1, 0] = x
        return out


def get_posterior(prob, ckpt=None):
    if ckpt and os.path.exists(ckpt):
        return load_posterior(prob, ckpt)
    torch.manual_seed(0)
    post = model_of_size(prob, "tiny")
    post.fit(n_train=3000, steps=1200, batch=256,
             retokenize=prob.make_retokenizer(), verbose=True)
    if ckpt:
        torch.save(post.state_dict(), ckpt)
    return post


def render_png(prob, m_true, raw, tokens, draws, path):
    import matplotlib.pyplot as plt

    from amortix.plotting import BLUE, DPI, FIGSIZE, GREY, param_axes, save_figure

    fig, (al, am) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    t_grid = torch.arange(raw.shape[1]) * prob.observer.dt_sim
    al.plot(t_grid.numpy(), raw[0, :, 0].numpy(), color=GREY, lw=1.0)
    al.plot(tokens[:, 0].numpy() * prob.observer.horizon, tokens[:, 1].numpy(),
            "o", color=BLUE, ms=4)
    al.set_xlabel("t")
    al.set_ylabel("x(t)")
    al.set_title(f"one observation set (K = {tokens.shape[0]})", fontsize=10)

    param_axes(am, prob)
    am.scatter(draws[:, 0], draws[:, 1], s=3, alpha=0.25, color=BLUE,
               linewidths=0)
    am.plot(*m_true[0].numpy(), "x", color="black", ms=8, mew=1.6)
    am.set_title("posterior, 2,000 draws", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", action="store_true",
                    help="render docs/media/custom_oscillator.png")
    ap.add_argument("--ckpt", metavar="PATH",
                    help="load this checkpoint if it exists; otherwise train "
                         "and save one here")
    args = ap.parse_args(argv)

    prob = DampedOscillator()
    post = get_posterior(prob, args.ckpt)

    gen = torch.Generator().manual_seed(3)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)
    tidx, cidx = prob.sample_design(gen, 12)
    tokens = prob.tokens_for(raw[0], tidx, cidx, gen)
    d = post.sample(tokens, n=2000)
    print(f"\ntrue (omega, gamma): {m_true[0].tolist()}")
    print(f"posterior mean     : {d.mean(0).tolist()}  sd {d.std(0).tolist()}")

    if args.png:
        render_png(prob, m_true, raw, tokens, d.numpy(),
                   os.path.join(MEDIA, "custom_oscillator.png"))


if __name__ == "__main__":
    main()
