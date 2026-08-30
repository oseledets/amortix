"""Plotting helpers: HDR contours, prior-box axes, and a save helper.

Small and dependency-light: KDE highest-density-region contours, axes fixed
to the prior box, and a save helper that creates the target directory.
Requires matplotlib (``pip install amortix[plot]``).
"""
from __future__ import annotations

import os
import sys

import matplotlib

if "matplotlib.pyplot" not in sys.modules:  # keep an interactive backend if
    matplotlib.use("Agg")                   # the caller already chose one
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

# The gallery's fixed palette: model draws, reference contours, raw signal.
BLUE, ORANGE, GREY = "#3b6bd6", "#d97706", "#8a8f98"
DPI = 110
FIGSIZE = (8.2, 3.6)


def hdr_contours(ax, pts, color, levels=(0.9, 0.5)):
    """Contours enclosing ``levels`` of the probability mass, by KDE.

    ``pts`` is an [n, 2] sample of the distribution to outline. The density
    thresholds are the sample quantiles of the KDE evaluated at the sample
    itself, so each contour encloses approximately the stated mass.
    """
    pts = np.asarray(pts)[:, :2]
    kde = gaussian_kde(pts.T)
    dens = kde(pts.T)
    cut = sorted(np.quantile(dens, 1.0 - np.asarray(levels)))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xg, yg = np.meshgrid(np.linspace(x0, x1, 140), np.linspace(y0, y1, 140))
    z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    ax.contour(xg, yg, z, levels=cut, colors=color, linewidths=1.4)


_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
          "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho",
          "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"}


def _math_label(name: str) -> str:
    """Parameter name as math text; Greek names become Greek letters."""
    return rf"$\{name}$" if name in _GREEK else rf"$\mathrm{{{name}}}$"


def param_axes(ax, prob, pad=0.06):
    """Fix the axes to the prior box of the first two parameters and label
    them with ``prob.prior.names`` in math text."""
    lo, hi = prob.prior.low.numpy(), prob.prior.high.numpy()
    span = hi - lo
    ax.set_xlim(lo[0] - pad * span[0], hi[0] + pad * span[0])
    ax.set_ylim(lo[1] - pad * span[1], hi[1] + pad * span[1])
    ax.set_xlabel(_math_label(prob.prior.names[0]))
    ax.set_ylabel(_math_label(prob.prior.names[1]))


def save_figure(fig, path: str):
    """Write the figure to ``path``, creating the directory if needed.

    Layout (tight_layout / suptitle spacing) is the caller's responsibility.
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
