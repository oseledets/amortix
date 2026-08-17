"""Variable observation designs: p(m | any K points) machinery.

A *design* is which points of the underlying process are observed. The
canonical training mode for design amortization (measured across three zoos,
see CALIBRATION.md) is:

  * every training example keeps its FULL raw trajectory;
  * every optimizer step draws a FRESH design for each example in the batch
    (``fit(retokenize=...)``) -- one simulation serves unboundedly many
    designs and design memorization is impossible;
  * design sizes K follow the "mix" law: 50% log-uniform[k_min, k_max] +
    50% uniform over the dense half -- the measured cure for the
    dense-design width tail;
  * tokens are bare points [t/horizon, y, 0, 0, logK/logKmax, channel] --
    the logK slot is design metadata (normalized softmax attention is
    cardinality-blind without it), the last slot carries a channel/sensor
    id for multi-sensor (e.g. PDE) observations.

``DesignProblem`` is the base class: subclasses define ``prior``, a
``trajectories(m, generator) -> [B, n_steps+1, C]`` simulator and optionally
``obs_noise`` / ``LOGSD`` (additive Gaussian / multiplicative log-normal
observation noise). Everything else -- fresh-design retokenizer, evaluation
tokenizers, variable-design SBC -- is provided here.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .prior import BoxUniform


class DesignObserver:
    """Grid metadata + token contract for variable designs."""

    N_FEATURES = 6

    def __init__(self, dt_sim: float, n_steps: int, k_max: int,
                 n_channels: int = 1):
        self.dt_sim = float(dt_sim)
        self.n_steps = int(n_steps)
        self.horizon = dt_sim * n_steps
        self.k_max = int(k_max)
        self.n_tokens = int(k_max)
        self.n_channels = int(n_channels)


def sample_k(k_max: int, k_min: int, gen: torch.Generator) -> int:
    """The canonical "mix" design-size law."""
    if torch.rand((), generator=gen).item() < 0.5:
        return int(torch.randint(k_max // 2, k_max + 1, (1,), generator=gen))
    u = torch.rand((), generator=gen).item()
    return max(k_min, min(int(round(k_min * (k_max / k_min) ** u)), k_max))


class DesignProblem:
    """Base for design-amortized problems (see module docstring)."""

    prior: BoxUniform
    observer: DesignObserver
    k_min: int = 4
    #: True iff the OBSERVED series is Markov (fully observed diffusion, no
    #: observation noise): selects the pair-family embedding under
    #: embed="auto"; hidden states / obs noise / multi-sensor -> bare points.
    markov_observed: bool = False
    obs_noise: float = 0.0          # additive Gaussian on observed values
    LOGSD: float = 0.0              # multiplicative log-normal (assays)

    def trajectories(self, m: torch.Tensor,
                     generator: torch.Generator = None) -> torch.Tensor:
        """[B, n_steps + 1, n_channels] raw solution/sample paths."""
        raise NotImplementedError

    # --- fit() protocol: (m, raw) + per-step retokenizer -------------------
    def simulate(self, n: int, generator: torch.Generator = None):
        m = self.prior.sample(n, generator)
        return m, self.trajectories(m, generator)

    def _noisy(self, y: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        if self.LOGSD > 0:
            return y.clamp_min(1e-6) * torch.exp(
                self.LOGSD * torch.randn(y.shape, generator=gen))
        if self.obs_noise > 0:
            return y + self.obs_noise * torch.randn(y.shape, generator=gen)
        return y

    def sample_design(self, gen: torch.Generator, k: int):
        """k random (time index, channel) pairs, time-sorted."""
        tidx = torch.randint(1, self.observer.n_steps + 1, (k,), generator=gen)
        cidx = (torch.zeros(k, dtype=torch.long) if self.observer.n_channels == 1
                else torch.randint(0, self.observer.n_channels, (k,),
                                   generator=gen))
        order = torch.argsort(tidx)
        return tidx[order], cidx[order]

    def tokens_for(self, raw_i: torch.Tensor, tidx: torch.Tensor,
                   cidx: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        obs = self.observer
        y = self._noisy(raw_i[tidx, cidx], gen)
        t = tidx.float() * obs.dt_sim / obs.horizon
        z = torch.zeros_like(y)
        kf = torch.full_like(y, math.log(tidx.numel()) / math.log(obs.k_max))
        return torch.stack([t, y, z, z, kf, cidx.float()], dim=-1)

    def make_retokenizer(self, seed: int = 7001, vectorized: bool = True):
        """Fresh designs for a whole batch at every optimizer step.

        Two paths, drawing from the same design law. ``vectorized=True``
        builds the batch with whole-tensor operations instead of looping over
        the sets. Measured on one H200 host at the training batch of 64, with
        ``fit``'s thread cap in place: $0.45$ ms per step against $2.8$ ms for
        the loop, beside $30$ ms of forward and backward -- so the loop is a
        tenth of a step and the vectorized path is a percent of it.

        An earlier A/B found no end-to-end difference, but it was run on a host
        shared with other jobs and without the thread cap, where both paths
        were dominated by scheduler contention; that measurement should not be
        trusted. Keeping the randomness on a CPU generator is deliberate --
        results stay identical across devices -- and moving the gather to the
        GPU is measurably *worse* at this batch size ($2.2$ ms), since the
        launch overhead exceeds the work.

        ``vectorized=False`` reproduces the exact random stream of runs made
        before the fast path existed.
        """
        gen = torch.Generator().manual_seed(seed)
        obs = self.observer

        def retok_loop(raw, _g):
            B = raw.shape[0]
            tokens = torch.zeros(B, obs.k_max, 6)
            mask = torch.zeros(B, obs.k_max, dtype=torch.bool)
            for i in range(B):
                k = sample_k(obs.k_max, self.k_min, gen)
                tidx, cidx = self.sample_design(gen, k)
                tokens[i, :k] = self.tokens_for(raw[i], tidx, cidx, gen)
                mask[i, :k] = True
            return tokens, mask

        def retok_vec(raw, _g):
            B, T1, C = raw.shape
            K = obs.k_max
            dev = raw.device
            # design sizes: the "mix" law of sample_k, drawn for the batch
            dense = torch.randint(K // 2, K + 1, (B,), generator=gen)
            u = torch.rand(B, generator=gen)
            sparse = torch.round(
                self.k_min * (K / self.k_min) ** u).long().clamp(self.k_min, K)
            k_i = torch.where(torch.rand(B, generator=gen) < 0.5,
                              dense, sparse).to(dev)
            mask = torch.arange(K, device=dev)[None, :] < k_i[:, None]
            # observation times: k_i i.i.d. draws per set, sorted, padding last
            r = torch.rand(B, K, generator=gen).to(dev)
            r = torch.where(mask, r, torch.full_like(r, 2.0))
            r, _ = r.sort(dim=1)
            tidx = (r * obs.n_steps).long().clamp(0, obs.n_steps - 1) + 1
            tidx = torch.where(mask, tidx, torch.ones_like(tidx))
            if obs.n_channels == 1:
                cidx = torch.zeros(B, K, dtype=torch.long, device=dev)
            else:
                cidx = torch.randint(0, obs.n_channels, (B, K),
                                     generator=gen).to(dev)
            vals = raw.gather(1, tidx[..., None].expand(B, K, C))
            y = vals.gather(2, cidx[..., None]).squeeze(-1)
            y = self._noisy_vec(y, gen, dev)
            t = tidx.float() * obs.dt_sim / obs.horizon
            kf = (torch.log(k_i.float()) / math.log(obs.k_max))[:, None] \
                .expand(B, K)
            z = torch.zeros_like(y)
            tokens = torch.stack([t, y, z, z, kf, cidx.float()], dim=-1)
            return tokens * mask[..., None], mask

        return retok_vec if vectorized else retok_loop

    def _noisy_vec(self, y: torch.Tensor, gen: torch.Generator, dev) -> torch.Tensor:
        if self.LOGSD > 0:
            e = torch.randn(y.shape, generator=gen).to(dev)
            return y.clamp_min(1e-6) * torch.exp(self.LOGSD * e)
        if self.obs_noise > 0:
            e = torch.randn(y.shape, generator=gen).to(dev)
            return y + self.obs_noise * e
        return y


def sbc_design(post, prob: DesignProblem, n_sims: int = 400,
               n_post: int = 200, seed: int = 0, k_fixed: int = None):
    """SBC over random designs (mixed by default, or a fixed-K bucket).

    NOTE (measured): at these sizes SBC misses width ratios up to ~1.3 and
    its per-cell p-values flicker when residual biases sit at 0.1-0.2
    posterior-sd. Use an exact-reference probe for verdicts whenever a
    tractable likelihood exists; SBC is the screen, not the record.
    """
    from .diagnostics import sbc_uniformity
    gen = torch.Generator().manual_seed(seed)
    m_true = prob.prior.sample(n_sims, generator=gen)
    raw = prob.trajectories(m_true, generator=gen)
    obs = prob.observer
    tokens = torch.zeros(n_sims, obs.k_max, 6)
    mask = torch.zeros(n_sims, obs.k_max, dtype=torch.bool)
    for i in range(n_sims):
        k = k_fixed if k_fixed else sample_k(obs.k_max, prob.k_min, gen)
        tidx, cidx = prob.sample_design(gen, k)
        tokens[i, :k] = prob.tokens_for(raw[i], tidx, cidx, gen)
        mask[i, :k] = True
    draws = post.sample_batch(tokens, n=n_post, seed=seed, chunk=32, mask=mask)
    ranks = (draws.numpy() < m_true.numpy()[:, None, :]).sum(1).astype(np.int64)
    return sbc_uniformity(ranks, n_post)
