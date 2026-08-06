"""Distances between sample sets -- how we actually judge posterior convergence.

The training loss cannot do it: with independent coupling most of
||v - (z1 - z0)||^2 is the irreducible Var(z1 - z0 | z_t), so the loss can rise
while the posterior keeps improving. What we care about is whether the sampled
posterior matches the reference posterior, in full dimension, and that is a
two-sample distance.
"""
from __future__ import annotations

import numpy as np


def energy_distance(x, y) -> float:
    """Energy distance between two sample sets, x [n, d] and y [m, d].

        E = 2/(nm) sum ||xi - yj|| - 1/n^2 sum ||xi - xj|| - 1/m^2 sum ||yi - yj||

    Zero if and only if the two distributions coincide, defined in any dimension,
    needs no density estimate. Unlike a per-parameter width or a marginal rank
    test, it sees shape and dependence, not just the marginals.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def mean_dist(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).mean()

    return float(2 * mean_dist(x, y) - mean_dist(x, x) - mean_dist(y, y))


def energy_distance_scaled(x, y, scale) -> float:
    """Energy distance after dividing each coordinate by `scale` (e.g. the prior
    range), so parameters on different units contribute comparably."""
    s = np.asarray(scale, dtype=np.float64)
    return energy_distance(np.asarray(x) / s, np.asarray(y) / s)
