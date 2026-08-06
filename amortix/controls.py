"""Controls that any honest comparison must report.

An estimator's error only means something next to two reference points:

* **prior-only** -- predict the prior mean and ignore the data entirely. For a
  uniform prior this is exactly 25% of the range in mean absolute error. Any
  parameter scored near 25% carries essentially no recoverable information, and
  "improving" it is noise. It also exposes vacuous calibration criteria: a
  posterior that just returns the prior has ~90% coverage of its 90% interval,
  so coverage alone can never demonstrate that a method learned anything.

* **summary + ridge** -- a degree-2 ridge regression on a handful of hand-written
  summary statistics of the observation. This is the cheapest thing a
  statistician would try. A neural amortized posterior that does not beat it has
  no claim to be useful, whatever its calibration looks like.

Both are deliberately dumb. That is the point: they are the floor a method has to
clear before any comparison against a "SOTA" baseline is worth reading.
"""
from __future__ import annotations

import numpy as np
import torch


def prior_mean_error(prob, m_true) -> np.ndarray:
    """MAE (% of prior range) of ignoring the data and predicting the prior mean."""
    mt = np.asarray(m_true)
    rng = (prob.prior.high - prob.prior.low).numpy()
    guess = (0.5 * (prob.prior.high + prob.prior.low)).numpy()
    return (np.abs(guess[None] - mt) / rng * 100).mean(0)


def summary_features(tokens: torch.Tensor) -> np.ndarray:
    """Generic per-feature summary statistics of a token set [B, T, F] -> [B, K].

    Deliberately model-agnostic: per token-feature mean, std, quantiles, mean
    square and mean |value|. For SDE tokens this automatically contains the
    quadratic-variation cue; for time series it contains level and spread.
    """
    x = tokens.numpy() if torch.is_tensor(tokens) else np.asarray(tokens)
    feats = [x.mean(1), x.std(1), (x ** 2).mean(1), np.abs(x).mean(1),
             np.quantile(x, 0.1, axis=1), np.quantile(x, 0.5, axis=1),
             np.quantile(x, 0.9, axis=1)]
    return np.concatenate(feats, axis=1)


class RidgeSummary:
    """Degree-2 ridge on summary statistics -- the 'a statistician would try this' control."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _design(self, tokens):
        f = summary_features(tokens)
        f = (f - self._mu) / self._sd
        quad = f[:, :, None] * f[:, None, :]
        iu = np.triu_indices(f.shape[1])
        return np.concatenate([np.ones((f.shape[0], 1)), f, quad[:, iu[0], iu[1]]], axis=1)

    def fit(self, tokens, m):
        f = summary_features(tokens)
        self._mu, self._sd = f.mean(0), f.std(0) + 1e-9
        X = self._design(tokens)
        Y = np.asarray(m)
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ Y)
        return self

    def predict(self, tokens):
        return self._design(tokens) @ self.W


def ridge_control_error(prob, n_train=4000, n_test=200, seed=0, alpha=1.0):
    """Train the ridge control on fresh simulations; return its MAE (% of range)."""
    g = torch.Generator().manual_seed(seed)
    m_tr, tk_tr = prob.simulate(n_train, generator=g)
    ridge = RidgeSummary(alpha).fit(tk_tr, m_tr.numpy())
    m_te = prob.prior.sample(n_test, generator=g)
    tk_te, _ = prob.observe(m_te, generator=g)
    pred = ridge.predict(tk_te)
    rng = (prob.prior.high - prob.prior.low).numpy()
    return (np.abs(pred - m_te.numpy()) / rng * 100).mean(0), ridge
