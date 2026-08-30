"""Priors over model parameters.

A prior must (a) draw parameter samples and (b) move parameters to/from the
*normalized* space the flow is trained in. We use a **probit** normalization: for
a uniform prior on [low, high], z = Phi^{-1}((m-low)/(high-low)) maps the prior
*exactly* onto the standard-normal base N(0, I) the flow starts from. This is the
right choice for Conditional Flow Matching — the base equals the prior marginal,
so the velocity field only has to learn the data-induced deviation from the prior
(easier to fit, better calibrated), and denormalization through Phi keeps every
posterior sample strictly inside the prior box.

(An affine standardization would map the prior to a *bounded uniform* while
the base stays Gaussian; that marginal mismatch forces the flow to transport
Gaussian tails into a flat box, piling mass at the boundaries and
miscalibrating the posterior — visible in SBC. The probit map removes the
mismatch.)
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch

_SQRT2 = math.sqrt(2.0)
_EPS = 1e-6


class BoxUniform:
    """Independent uniform prior on a box [low, high]^d."""

    def __init__(self, low: Sequence[float], high: Sequence[float],
                 names: List[str] = None):
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        assert self.low.shape == self.high.shape
        self.dim = int(self.low.numel())
        self.names = list(names) if names is not None else [f"m{i}" for i in range(self.dim)]
        self.span = self.high - self.low

    def sample(self, n: int, generator: torch.Generator = None) -> torch.Tensor:
        u = torch.rand(n, self.dim, generator=generator)
        return self.low + u * self.span

    # --- normalized (training) space: probit map, prior -> N(0, I) --------
    def normalize(self, m: torch.Tensor) -> torch.Tensor:
        u = ((m - self.low) / self.span).clamp(_EPS, 1.0 - _EPS)
        return _SQRT2 * torch.erfinv(2.0 * u - 1.0)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        u = 0.5 * (1.0 + torch.erf(z / _SQRT2))
        return self.low + u * self.span
