"""The full evaluation instrument on Cox-Ingersoll-Ross.

CIR's transition density is a noncentral chi-square, so the package can build
a frozen evaluation set with EXACT-likelihood MCMC references, validated by
two independent chains, and score a trained model against it -- with the
set's own resolution floor reported alongside. CIR is imported from the
package rather than defined here: its simulator draws the exact noncentral
chi-square transitions, and that simulator is what makes the exact reference
possible (defining your own system is shown in examples 01 and 04).
~30 minutes end to end on CPU.

Run:  python examples/gallery/03_exact_reference_cir.py
      --png          also render docs/media/cir_reference.png
      --ckpt PATH    load the checkpoint if it exists, else train and save it
"""
import argparse
import os

import torch

from amortix.evaluation import (build_eval_set, evaluate, fid, load_posterior,
                                model_of_size)
from amortix.problems.design_basic import CIRDesign

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "docs", "media")


def get_posterior(prob, ckpt=None):
    if ckpt and os.path.exists(ckpt):
        return load_posterior(prob, ckpt)
    post = model_of_size(prob, "tiny")
    post.fit(n_train=8000, steps=3000, batch=256,
             retokenize=prob.make_retokenizer(), verbose=True)
    if ckpt:
        torch.save(post.state_dict(), ckpt)
    return post


def render_png(prob, post, es, r, path):
    import math

    import matplotlib.pyplot as plt

    from amortix.plotting import (BLUE, DPI, ORANGE, hdr_contours, param_axes,
                        save_figure)

    # same draw configuration as evaluate() above, so the per-panel FIDs
    # match the printed median
    smp = post.sample_batch(torch.as_tensor(es.tokens), n=2000, seed=0,
                            mask=torch.as_tensor(es.mask)).numpy()
    n = len(es.chain_a)
    ncols = 2 if n <= 4 else 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, 3.6 * nrows),
                             dpi=DPI)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.set_axis_off()
            continue
        param_axes(ax, prob)
        hdr_contours(ax, es.chain_a[i], ORANGE)
        ax.scatter(smp[i][:, 0], smp[i][:, 1], s=3, alpha=0.25, color=BLUE,
                   linewidths=0)
        ax.plot(*es.m_true[i][:2], "x", color="black", ms=8, mew=1.6)
        ax.set_title(f"set {i}: FID {fid(smp[i], es.chain_a[i]):.4f}",
                     fontsize=10)
    fig.suptitle(f"CIR evaluation sets -- floor {r['null_median']:.4f} at "
                 f"2,000 draws", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", action="store_true",
                    help="render docs/media/cir_reference.png")
    ap.add_argument("--ckpt", metavar="PATH",
                    help="load this checkpoint if it exists; otherwise train "
                         "and save one here")
    args = ap.parse_args(argv)

    prob = CIRDesign()
    post = get_posterior(prob, args.ckpt)

    es = build_eval_set(prob, "cir", K=20, n_sets=4, n_chain=20000,
                        seed=11, workers=4)
    print(f"\nevaluation set: {es!r}")
    r = evaluate(post, es, n_draw=2000)
    print(f"median FID {r['fid_median']:.4f} against a floor of "
          f"{r['null_median']:.4f}")

    if args.png:
        render_png(prob, post, es, r, os.path.join(MEDIA,
                                                   "cir_reference.png"))


if __name__ == "__main__":
    main()
