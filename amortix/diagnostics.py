"""Calibration diagnostics for amortized posteriors.

Simulation-Based Calibration (SBC, Talts et al. 2018) plus coverage and
error-vs-uncertainty, as a reusable harness that works for ANY `Problem`.

Idea: if the amortized posterior is the true posterior, then for data drawn from
the prior+simulator the rank of the true parameter among L posterior draws is
uniform on {0,...,L}. Deviations diagnose mis-calibration (over/under-confidence,
bias). We also derive the full coverage curve from the same ranks and report
mean-error vs mean posterior-std per parameter.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def run_sbc(post, prob, n_sims: int = 300, n_post: int = 200, seed: int = 0,
            chunk: int = 32):
    """One pass of SBC. Returns a dict with ranks, estimates, stds, truths.

    ranks[i, j] in {0,...,n_post} is the rank of the true j-th parameter among
    the n_post posterior draws for simulated dataset i. Datasets are sampled in
    batches (`chunk`), so this is one vectorized ODE solve per chunk rather than
    one per dataset.
    """
    gen = torch.Generator().manual_seed(seed)
    m_true = prob.prior.sample(n_sims, generator=gen)
    tokens, _ = prob.observe(m_true, generator=gen)

    draws = post.sample_batch(tokens, n=n_post, seed=seed, chunk=chunk)  # [n_sims, L, d]
    mt = m_true.numpy()
    s = draws.numpy()
    ranks = (s < mt[:, None, :]).sum(1).astype(np.int64)
    est = s.mean(1)
    std = s.std(1)
    return {
        "ranks": ranks, "est": est, "std": std, "truth": mt,
        "n_post": n_post, "names": list(prob.prior.names),
        "rng": (prob.prior.high - prob.prior.low).numpy(),
    }


def coverage_from_ranks(ranks: np.ndarray, n_post: int, levels: Sequence[float]):
    """Empirical central-interval coverage at each nominal level, per parameter.

    The truth lies in the central q-credible interval iff its normalized rank
    r/L is within [(1-q)/2, (1+q)/2]."""
    u = ranks / float(n_post)                       # in [0,1]
    out = {}
    for q in levels:
        lo, hi = (1 - q) / 2, (1 + q) / 2
        out[q] = ((u >= lo) & (u <= hi)).mean(0)    # per-parameter
    return out


def calibration_error(ranks: np.ndarray, n_post: int,
                      levels=(0.5, 0.9, 0.95)) -> float:
    """Mean |empirical - nominal| central-interval coverage, in percentage points.

    One scalar summarising how far the posterior's credible intervals are from
    their nominal level (0 = perfectly calibrated)."""
    cov = coverage_from_ranks(ranks, n_post, levels)
    return float(np.mean([np.abs(cov[q] - q) for q in levels])) * 100


def sbc_uniformity(ranks: np.ndarray, n_post: int, n_bins: int = 20):
    """Chi-square uniformity p-value per parameter (low p => mis-calibrated)."""
    from scipy.stats import chi2
    n_sims, d = ranks.shape
    u = ranks / float(n_post)
    pvals = np.zeros(d)
    for j in range(d):
        counts, _ = np.histogram(u[:, j], bins=n_bins, range=(0, 1))
        exp = n_sims / n_bins
        stat = ((counts - exp) ** 2 / exp).sum()
        pvals[j] = chi2.sf(stat, df=n_bins - 1)
    return pvals


def _ascii_hist(vals01, width: int = 24, n_bins: int = 10):
    counts, _ = np.histogram(vals01, bins=n_bins, range=(0, 1))
    mx = max(counts.max(), 1)
    exp = len(vals01) / n_bins
    bars = []
    for c in counts:
        fill = int(round(width * c / mx))
        bars.append("#" * fill + "." * (width - fill))
    return counts, bars, exp


def diagnose(post, prob, n_sims: int = 300, n_post: int = 200, seed: int = 0,
             levels=(0.5, 0.9, 0.95), plot_path: str = None, verbose: bool = True):
    """Run SBC + coverage + error/uncertainty and print a report. Returns the
    raw results dict (so callers can post-process or plot)."""
    res = run_sbc(post, prob, n_sims=n_sims, n_post=n_post, seed=seed)
    ranks, names, rng = res["ranks"], res["names"], res["rng"]
    cov = coverage_from_ranks(ranks, n_post, levels)
    pvals = sbc_uniformity(ranks, n_post)
    err = np.abs(res["est"] - res["truth"]) / rng * 100
    pstd = res["std"] / rng * 100
    res.update({"coverage": cov, "uniformity_p": pvals, "err": err, "pstd": pstd})

    if verbose:
        print(f"\nSBC diagnostics  ({n_sims} sims x {n_post} posterior draws)")
        print(f"{'param':>10} | {'err%':>6} | {'std%':>6} | "
              + " | ".join(f"cov{int(q*100):>2}" for q in levels)
              + f" | {'SBC-p':>6}")
        for j, nm in enumerate(names):
            cells = " | ".join(f"{cov[q][j]*100:4.0f}%" for q in levels)
            flag = "" if pvals[j] > 0.05 else "  <-- check"
            print(f"{nm:>10} | {err[j].mean():5.1f}% | {pstd[j].mean():5.1f}% | "
                  f"{cells} | {pvals[j]:6.3f}{flag}")
        print("\nSBC rank histograms (flat = calibrated; ∪ = under-confident; "
              "∩ = over-confident; slope = biased):")
        u = ranks / float(n_post)
        for j, nm in enumerate(names):
            counts, bars, exp = _ascii_hist(u[:, j])
            print(f"  {nm:>10}: " + bars[0])
            for b in bars[1:]:
                print(f"  {'':>10}  " + b)
        print(f"  (expected ~{int(exp)} per bin under uniformity)")

    if plot_path is not None:
        try:
            _plot(res, levels, plot_path)
            if verbose:
                print(f"\nsaved SBC + coverage plot -> {plot_path}")
        except Exception as e:
            if verbose:
                print(f"[plot skipped: {e}]")
    return res


def _plot(res, levels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, ranks, n_post = res["names"], res["ranks"], res["n_post"]
    d = len(names)
    u = ranks / float(n_post)
    n_sims = ranks.shape[0]
    n_bins = 12
    exp = n_sims / n_bins
    band = 2.0 * np.sqrt(exp * (1 - 1.0 / n_bins))      # ~95% binomial band

    fig, axes = plt.subplots(1, d + 1, figsize=(3.0 * (d + 1), 3.0))
    for j in range(d):
        ax = axes[j]
        ax.hist(u[:, j], bins=n_bins, range=(0, 1), color="#4C72B0", alpha=0.8)
        ax.axhline(exp, color="k", lw=1)
        ax.axhspan(exp - band, exp + band, color="grey", alpha=0.25)
        ax.set_title(f"SBC: {names[j]}")
        ax.set_xlabel("rank / L")
    # coverage calibration plot
    ax = axes[d]
    qs = np.linspace(0.02, 0.98, 25)
    full = coverage_from_ranks(ranks, n_post, qs)
    for j in range(d):
        ax.plot(qs, [full[q][j] for q in qs], lw=1.4, label=names[j])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("coverage calibration")
    ax.set_xlabel("nominal"); ax.set_ylabel("empirical")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
