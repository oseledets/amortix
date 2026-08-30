"""Command-line interface: `amortix <command>`.

    amortix cases                     list the gallery
    amortix recover ou                train a posterior and benchmark vs the classical baseline
    amortix sbc ou                    strict calibration check (SBC) for one case
    amortix gallery                   accuracy benchmark across all cases

Installed as a console script; from the repository it runs via uv:
    uv run amortix cases
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

from .problems import GALLERY


def _load(case):
    mod = importlib.import_module(f"amortix.problems.{case}")
    return mod, mod.make()


def cmd_cases(args):
    from .sde import SDEProblem
    print(f"{'case':>12} | {'kind':>6} | {'dim':>3} | {'tokens':>6} | baseline")
    print("-" * 62)
    for name in GALLERY:
        mod, prob = _load(name)
        kind = "SDE" if isinstance(prob, SDEProblem) else "ODE"
        print(f"{name:>12} | {kind:>3}-{prob.state_dim}D | {prob.prior.dim:>3} | "
              f"{prob.observer.n_tokens:>6} | {mod.SOTA_NAME}")


def cmd_recover(args):
    from .flow import FlowPosterior
    mod, prob = _load(args.case)
    rng = (prob.prior.high - prob.prior.low).numpy()
    post = FlowPosterior(prob).fit(n_train=args.n_train, epochs=args.epochs)

    gen = torch.Generator().manual_seed(123)
    m_true = prob.prior.sample(args.n_test, generator=gen)
    tokens, traj = prob.observe(m_true, generator=gen)
    draws = post.sample_batch(tokens, n=args.n_post, seed=0).numpy()
    base = np.stack([mod.sota(tokens[i].numpy(), traj[i].numpy(), prob)
                     for i in range(args.n_test)])

    mt = m_true.numpy()
    a_err = (np.abs(draws.mean(1) - mt) / rng * 100).mean(0)
    b_err = (np.abs(base - mt) / rng * 100).mean(0)
    pstd = (draws.std(1) / rng * 100).mean(0)
    lo, hi = np.quantile(draws, 0.05, axis=1), np.quantile(draws, 0.95, axis=1)
    cov = ((mt >= lo) & (mt <= hi)).mean(0) * 100
    print(f"\n{'param':>10} | {'amort':>7} | {'post.std':>8} | {mod.SOTA_NAME[:14]:>14} | {'cov90':>6}")
    for j, nm in enumerate(prob.prior.names):
        print(f"{nm:>10} | {a_err[j]:6.2f}% | {pstd[j]:7.2f}% | {b_err[j]:13.2f}% | {cov[j]:5.0f}%")
    print(f"{'ALL':>10} | {a_err.mean():6.2f}% | {pstd.mean():7.2f}% | "
          f"{b_err.mean():13.2f}% | {cov.mean():5.0f}%")


def cmd_sbc(args):
    from .flow import FlowPosterior
    from .diagnostics import diagnose
    _, prob = _load(args.case)
    post = FlowPosterior(prob).fit(n_train=args.n_train, epochs=args.epochs)
    diagnose(post, prob, n_sims=args.n_sims, n_post=args.n_post, plot_path=args.plot)


def cmd_gallery(args):
    from .flow import FlowPosterior
    print(f"{'case':>12} | {'amort':>7} | {'baseline':>9} | winner")
    print("-" * 46)
    for name in GALLERY:
        mod, prob = _load(name)
        rng = (prob.prior.high - prob.prior.low).numpy()
        post = FlowPosterior(prob).fit(n_train=args.n_train, epochs=args.epochs, verbose=False)
        gen = torch.Generator().manual_seed(123)
        m_true = prob.prior.sample(args.n_test, generator=gen)
        tokens, traj = prob.observe(m_true, generator=gen)
        draws = post.sample_batch(tokens, n=args.n_post, seed=0).numpy()
        base = np.stack([mod.sota(tokens[i].numpy(), traj[i].numpy(), prob)
                         for i in range(args.n_test)])
        mt = m_true.numpy()
        a = (np.abs(draws.mean(1) - mt) / rng * 100).mean()
        b = (np.abs(base - mt) / rng * 100).mean()
        print(f"{name:>12} | {a:6.1f}% | {b:8.1f}% | {'amort' if a <= b else 'baseline'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amortix",
                                 description="Amortized parameter recovery for dynamical systems")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cases", help="list gallery problems").set_defaults(fn=cmd_cases)

    def budget(p, n_train=8000, epochs=25):
        p.add_argument("--n_train", type=int, default=n_train)
        p.add_argument("--epochs", type=int, default=epochs)
        p.add_argument("--n_test", type=int, default=40)
        p.add_argument("--n_post", type=int, default=600)

    p = sub.add_parser("recover", help="train + benchmark one case vs its classical baseline")
    p.add_argument("case", choices=GALLERY); budget(p); p.set_defaults(fn=cmd_recover)

    p = sub.add_parser("sbc", help="strict calibration check for one case")
    p.add_argument("case", choices=GALLERY)
    p.add_argument("--n_train", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n_sims", type=int, default=300)
    p.add_argument("--n_post", type=int, default=150)
    p.add_argument("--plot", type=str, default=None, help="save an SBC plot here")
    p.set_defaults(fn=cmd_sbc)

    p = sub.add_parser("gallery", help="accuracy benchmark across all cases")
    budget(p, n_train=6000, epochs=20); p.set_defaults(fn=cmd_gallery)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
