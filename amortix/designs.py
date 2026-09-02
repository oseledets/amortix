"""Variable observation designs: p(m | any K points) machinery.

A *design* is which points of the underlying process are observed. The
canonical training mode for design amortization (measured across the problem
zoo; see the technical report, report/techreport.pdf) is:

  * every training example keeps its FULL raw trajectory;
  * every optimizer step draws a FRESH design for each example in the batch
    (``fit(retokenize=...)``) -- one simulation serves unboundedly many
    designs and design memorization is impossible;
  * design sizes K follow the "mix" law: 50% log-uniform[k_min, k_max] +
    50% uniform over the dense half, which keeps dense designs well
    represented so that posterior widths stay calibrated at large K;
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
        (the default) builds the batch with whole-tensor operations instead
        of looping over the sets, which removes the per-set Python overhead
        from every optimizer step. Randomness stays on a CPU generator so
        results are identical across devices, and the gather is kept on the
        CPU: at training batch sizes the GPU launch overhead exceeds the
        work.

        ``vectorized=False`` reproduces the exact random stream of runs made
        before the vectorized path existed.
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


def tokens_from_data(prob: DesignProblem, times, values, channels=None):
    """Token tensor [K, 6] for measured data at arbitrary timestamps.

    times: array-like, in the same time units as the simulator grid
    (the observer horizon is dt_sim * n_steps); values: raw signal
    units; channels: integer channel per reading (default all 0).
    No noise is added -- the data are already measured. Rows are
    returned time-sorted, ready for FlowPosterior.sample.
    """
    obs = prob.observer
    t = torch.as_tensor(np.asarray(times), dtype=torch.float32).reshape(-1)
    y = torch.as_tensor(np.asarray(values), dtype=torch.float32).reshape(-1)
    if t.numel() != y.numel():
        raise ValueError(f"times ({t.numel()}) and values ({y.numel()}) "
                         f"must have the same length")
    if channels is None:
        c = torch.zeros(t.numel())
    else:
        c = torch.as_tensor(np.asarray(channels)).reshape(-1)
        if c.numel() != t.numel():
            raise ValueError("channels must have one entry per reading")
        if c.is_floating_point() and not torch.equal(c, c.round()):
            raise ValueError("channels must be integer channel ids, got "
                             f"{c.tolist()}")
        c = c.long()
        if c.numel() and (c.min() < 0 or c.max() >= obs.n_channels):
            raise ValueError(
                f"channels must lie in [0, n_channels={obs.n_channels}), "
                f"got {c.tolist()}")
        c = c.float()
    order = torch.argsort(t, stable=True)
    t, y, c = t[order], y[order], c[order]
    # same feature layout as DesignProblem.tokens_for:
    # [t/horizon, y, 0, 0, log(K)/log(k_max), channel]
    z = torch.zeros_like(y)
    kf = torch.full_like(y, math.log(t.numel()) / math.log(obs.k_max))
    tokens = torch.stack([t / obs.horizon, y, z, z, kf, c], dim=-1)
    validate_design_tokens(tokens, obs)
    return tokens


def validate_design_tokens(tokens, observer: DesignObserver, mask=None,
                           tol: float = 1e-4) -> None:
    """Check that ``tokens`` follow the DesignObserver layout; raise otherwise.

    tokens: [..., K, 6] with rows [t/horizon, value, 0, 0, log K/log k_max,
    channel]. mask: optional bool [..., K]; rows with mask False are padding
    and are not inspected. The bare-point embedding reads slots 0, 1, 4 and
    5 and ignores slots 2 and 3, so a token set that departs from the layout
    (a second reading packed into a reserved slot, a channel id outside the
    observer's range, a design-size entry that does not match the set) is
    otherwise conditioned on wrongly with no error raised. The checks:

      slot 0     normalized time in [0, 1]
      slot 1     finite value
      slots 2-3  exactly zero (reserved)
      slot 4     log K / log k_max in [0, 1], with K the number of valid
                 rows of the set, i.e. the design size the network is told
      slot 5     integer channel id in [0, n_channels)

    ``DesignProblem.tokens_for`` and ``tokens_from_data`` produce conforming
    tokens; FlowPosterior calls this check on the tokens it receives for a
    DesignProblem. ``tol`` absorbs float32 rounding of the normalized time
    and of the design-size entry.
    """
    tokens = torch.as_tensor(tokens)
    nf = observer.N_FEATURES
    if tokens.dim() < 2 or tokens.shape[-1] != nf:
        raise ValueError(
            f"design tokens must have shape [..., K, {nf}], got "
            f"{tuple(tokens.shape)}. {_LAYOUT_HINT}")
    K = tokens.shape[-2]
    sets = tokens.reshape(-1, K, nf)
    if mask is None:
        valid = torch.ones(sets.shape[:2], dtype=torch.bool, device=sets.device)
    else:
        valid = torch.as_tensor(mask, dtype=torch.bool, device=sets.device)
        if valid.shape != tokens.shape[:-1]:
            raise ValueError(f"mask shape {tuple(valid.shape)} does not match "
                             f"token sets {tuple(tokens.shape[:-1])}")
        valid = valid.reshape(-1, K)
    rows = sets[valid]                                   # [N, 6], valid only
    if rows.numel() == 0:
        return
    n_valid = valid.sum(1)                               # design size per set
    set_of_row = torch.nonzero(valid)                    # [N, 2] -> (set, row)

    def fail(slot: int, what: str, bad: torch.Tensor, detail: str = ""):
        i = int(torch.nonzero(bad)[0])
        b, r = (int(v) for v in set_of_row[i])
        raise ValueError(
            f"design token slot {slot} {what}: found {rows[i, slot].item():g} "
            f"(set {b}, row {r}){detail}. {_LAYOUT_HINT}")

    finite = torch.isfinite(rows)
    if not finite.all():
        bad_row = ~finite.all(1)
        first = int(torch.nonzero(bad_row)[0])
        fail(int(torch.nonzero(~finite[first])[0]), "must be finite", bad_row)
    t = rows[:, 0]
    fail_t = (t < -tol) | (t > 1.0 + tol)
    if fail_t.any():
        fail(0, "(time / horizon) must lie in [0, 1]", fail_t)
    for slot in (2, 3):
        nz = rows[:, slot] != 0
        if nz.any():
            fail(slot, "is reserved and must be exactly 0", nz)
    kf = rows[:, 4]
    too_many = n_valid[set_of_row[:, 0]] > observer.k_max
    if too_many.any():
        b = int(set_of_row[int(torch.nonzero(too_many)[0]), 0])
        fail(4, "(log K / log k_max) exceeds 1", too_many,
             f"; set {b} has K={int(n_valid[b])} valid rows, more than "
             f"observer.k_max={observer.k_max}")
    if observer.k_max > 1:
        expected = (torch.log(n_valid.clamp_min(1).float())
                    / math.log(observer.k_max))
        off = (kf - expected[set_of_row[:, 0]]).abs() > tol
        if off.any():
            b = int(set_of_row[int(torch.nonzero(off)[0]), 0])
            k_b = int(n_valid[b])
            fail(4, "(log K / log k_max) does not match the set", off,
                 f"; set {b} has K={k_b} valid rows, so the entry must be "
                 f"log({k_b})/log({observer.k_max}) = {float(expected[b]):.4f}"
                 " (padding rows must be marked with mask=False)")
    fail_kf = (kf < -tol) | (kf > 1.0 + tol)
    if fail_kf.any():
        fail(4, "(log K / log k_max) must lie in [0, 1]", fail_kf)
    c = rows[:, 5]
    fail_c = (c != c.round()) | (c < 0) | (c >= observer.n_channels)
    if fail_c.any():
        fail(5, "(channel id) must be an integer in "
                f"[0, n_channels={observer.n_channels})", fail_c)


_LAYOUT_HINT = (
    "DesignObserver tokens are [t/horizon, value, 0, 0, log K/log k_max, "
    "channel] with one reading per row; the embedding reads slots 0, 1, 4 "
    "and 5 only. Build tokens with DesignProblem.tokens_for or "
    "amortix.designs.tokens_from_data.")


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
