"""Transformer set-encoder for observations.

Follows the architecture choices of arXiv:2503.01375: bidirectional self-attention
with rotary position embeddings (RoPE), ReLU^2 feed-forward, and RMS normalization.
Attention pooling produces a single conditioning vector for the velocity field, and
`encode` exposes the per-token memory for cross-attention conditioning.
The encoder is permutation-equivariant up to RoPE and handles a variable number
of observation tokens via the padding mask.

The constructor flags (`rope`, `embed`, `input_norm`, `pool`, ...) exist because
the encoder was measured, not assumed, on the `linear_gaussian` testbed, where the
statistic it must extract is known exactly (see results/DEBUG_encoder.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class FeatureNorm(nn.Module):
    """Running standardization of the raw token features (masked BatchNorm-style).

    Observation tokens arrive in physical units -- on `linear_gaussian` the values
    span +-15 while the index feature spans [0, 1). A linear embed followed by a
    residual stream can rescale them, but only by fighting the initialization; and
    the pre-norm blocks see a stream whose scale is set by the largest feature.
    Standardizing first is free and makes the two features comparable.

    Statistics are collected over valid tokens during training and frozen at
    evaluation, so the encoding of one dataset never depends on its batch-mates.

    On by default: measured on `linear_gaussian` it is the only encoder change that
    improved the distance to the exact posterior (-17%), and it makes the sufficient
    statistic 2.5x more readable out of the token memory (results/DEBUG_encoder.md).
    """

    def __init__(self, n_features: int, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("var", torch.ones(n_features))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))

    def forward(self, x, mask=None):                  # x [B, T, F]
        if self.training and torch.is_grad_enabled():
            flat = x.reshape(-1, x.shape[-1]) if mask is None else x[mask]
            if flat.shape[0] > 1:
                m, v = flat.mean(0), flat.var(0, unbiased=False)
                with torch.no_grad():
                    self.n_seen += 1
                    # debiased warmup: the first batches count fully, so the
                    # statistics are usable immediately instead of after ~1/momentum
                    # steps -- small budgets (the gallery runs a few hundred) would
                    # otherwise train under a normalization that is still moving.
                    w = max(self.momentum, 1.0 / float(self.n_seen))
                    self.mean.lerp_(m.detach(), w)
                    self.var.lerp_(v.detach(), w)
        sd = torch.sqrt(self.var)
        # A feature that never varies (comp_id on a 1-D SDE, log dt with one
        # channel) has sd 0; dividing by it would be inf * 0 = NaN. Such a feature
        # carries no information, so pass the (zero) deviation through unscaled.
        scale = torch.where(sd > 1e-6 * (1.0 + self.mean.abs()),
                            1.0 / sd.clamp(min=1e-12), torch.ones_like(sd))
        return (x - self.mean) * scale


class MonotoneWarp(nn.Module):
    """Fully learnable monotone odd warp: linear part + octave bank of
    signed-log1p basis functions,

        w(v) = c0 * v + sum_k ck * sign(v) * log1p(|v| / s_k),

    with ck, c0 >= 0 via softplus (monotone by construction) and the scales
    s_k ALSO learnable (in log space, initialized on an octave grid
    2^-3..2^4). The family contains the identity, log-like compression at any
    scale, and their mixtures — and log shapes sit in the initialization basin
    at every octave simultaneously, so no hand-picked scale remains. That
    basin property is load-bearing: a learnable Yeo-Johnson warp, which also
    contains log but initializes at the identity, never moves off it (sigma
    bias +0.451 vs +0.483 raw)."""

    def __init__(self, n_scales: int = 8, lo: float = -3.0, hi: float = 4.0):
        super().__init__()
        octaves = torch.linspace(lo, hi, n_scales) * 0.6931471805599453
        self.log_s = nn.Parameter(octaves)
        self.raw_c = nn.Parameter(torch.zeros(n_scales))
        self.raw_c0 = nn.Parameter(torch.tensor(0.0))

    def forward(self, v):
        s = self.log_s.exp()
        basis = torch.sign(v)[..., None] * torch.log1p(v.abs()[..., None] / s)
        c = F.softplus(self.raw_c)
        c0 = F.softplus(self.raw_c0)
        return c0 * v + (basis * c).sum(-1)


class WarpDiffEmbed(nn.Module):
    """Learnable warped-increment token embedding for scale-varying series.

    Recomputes each token's increment IN a learnably warped coordinate,
        dw = w(x + dx) - w(x),
    and embeds [t, w(x), dw, dw^2, log10 dt, comp] instead of the raw token.
    The warp is a per-value bijection (no information loss). The essential
    point is WHERE the increment is taken: warping dx separately from x (a
    per-feature transform) leaves a residual bias that grows with training
    budget; differencing in the warped coordinate does not.

    Two warp families:
      kind="basis" (default, embed="wbasis"): MonotoneWarp — the fully
        learnable octave mixture, no scale hyperparameter at all.
      kind="slog" (embed="wdiff"): single signed-log1p with one learnable
        scale s; contains the hand-crafted log-price observer as s -> 0 and
        the identity as s -> inf.

    Measured on raw-price GBM (the worst floating-scale case in the gallery),
    sigma SBC at production budget (n_train=40000, steps=12000, 500x200):
    raw tokens p=0.002; per-feature slog p=0.014 (fails — its screen-level
    +0.024 sd residual grows with budget); wdiff p=0.876; wbasis p=0.283;
    fully-convolutional wconv (learnable taps, no hand square) p=0.130. The
    hand-crafted log-price observer reference is p=0.812. Structure buys
    calibration margin monotonically; every warp-then-difference variant
    passes. ou/cir verified unaffected.

    Assumes the 6-feature PathObserver token layout [t, x, dx, dx^2, res, cid].
    """

    def __init__(self, n_features: int, dim: int, kind: str = "basis"):
        super().__init__()
        if n_features != 6:
            raise ValueError(
                "warped-increment embed expects the 6-feature PathObserver "
                f"token layout [t, x, dx, dx^2, res, cid]; got {n_features}")
        self.kind = kind
        if kind == "basis":
            self.warp = MonotoneWarp(n_scales=8)
        elif kind == "slog":
            self.raw_s = nn.Parameter(torch.zeros(1))  # softplus -> s ~ 0.69
        else:
            raise ValueError(kind)
        self.norm = FeatureNorm(n_features)
        self.proj = nn.Linear(n_features, dim)

    def _w(self, v):
        if self.kind == "basis":
            return self.warp(v)
        s = F.softplus(self.raw_s) + 1e-6
        return torch.sign(v) * torch.log1p(v.abs() / s)

    def forward(self, tokens, mask=None):             # tokens [B, T, 6]
        t, x, dx, dx2, res, cid = tokens.unbind(-1)
        wx = self._w(x)
        dw = self._w(x + dx) - wx
        f = torch.stack([t, wx, dw, dw * dw, res, cid], dim=-1)
        return self.proj(self.norm(f, mask))


class PointEmbed(nn.Module):
    """Bare-point embedding for variable designs on NON-Markov-observed
    series (ODE + observation noise, hidden states, multi-sensor PDE): the
    likelihood factorizes over single points given the parameters, so the
    sufficient token is [t, w(y), design-metadata, channel] with a learnable
    monotone warp on the value; all temporal/spatial structure is learned by
    the t-RoPE attention. Measured winners on FHN / Hodgkin-Huxley / PK /
    Fisher-KPP (see CALIBRATION.md). Select with embed="wpoint"."""

    takes_mask = True

    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.warp = MonotoneWarp(8, lo=-3.0, hi=4.0)
        self.norm = FeatureNorm(4)
        self.proj = nn.Linear(4, dim)

    def forward(self, tokens, mask=None):
        t = tokens[..., 0]
        y = tokens[..., 1]
        meta = tokens[..., 4]
        chan = tokens[..., 5]
        f = torch.stack([t, self.warp(y), meta, chan], dim=-1)
        return self.proj(self.norm(f, mask))


class SetCondPairEmbed(nn.Module):
    """The measured-best universal embedding for variable designs on
    Markov-observed series: WarpPairEmbed's warped-increment features with
    (a) log-compression of the heavy-tail-prone channels (jump outliers
    become additive; near-identity on clean diffusions since log1p(x)≈x for
    small x) and (b) LEARNED set conditioning: a masked pool over the
    features computes a per-dataset context, an MLP maps it to per-channel
    scale/shift applied AFTER FeatureNorm (zero-init => exactly the robust
    base at start). Duel result on Merton (heavy tails, floating jump
    threshold): matches hand median-scaling (+0.13 vs +0.09 posterior-sd,
    sr 1.25 vs 1.33) with zero hand statistics. Three correctness rules,
    each paid for with a found bug: zero-init (right basin), norm BEFORE
    modulation (else the learned gain re-creates representation drift), and
    the mask must reach set-level statistics (padding otherwise dilutes the
    context differently in padded vs packed inference paths).
    Select with embed="wfilm"."""

    takes_mask = True

    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.warp_x = MonotoneWarp(8, lo=-3.0, hi=4.0)
        self.warp_t = MonotoneWarp(8, lo=-9.0, hi=0.0)
        self.norm = FeatureNorm(6)
        self.proj = nn.Linear(6, dim)
        self.cond = nn.Sequential(nn.Linear(12, 32), nn.GELU(),
                                  nn.Linear(32, 12))
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    def forward(self, tokens, mask=None):
        t = tokens[..., 0]
        x = tokens[..., 1]
        wx = self.warp_x(x)
        dw = torch.zeros_like(wx)
        dw[:, 1:] = wx[:, 1:] - wx[:, :-1]
        dt = torch.zeros_like(t)
        dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0.0)
        g = self.warp_t(dt)
        first = torch.zeros_like(t)
        first[:, 0] = 1.0
        dwc = torch.sign(dw) * torch.log1p(dw.abs())
        qv = torch.log1p(dw * dw)
        f = torch.stack([t, wx, dwc, qv, g, first], dim=-1)
        w = torch.ones_like(t) if mask is None else mask.float()
        wsum = w.sum(1, keepdim=True).clamp_min(1.0)
        fmean = (f * w[..., None]).sum(1) / wsum
        famean = (f.abs() * w[..., None]).sum(1) / wsum
        ctx = torch.cat([fmean, famean], dim=-1)
        fn = self.norm(f, mask)
        gb = self.cond(ctx)
        gamma, beta = gb[:, :6], gb[:, 6:]
        fn = fn * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        return self.proj(fn)


class WarpPairEmbed(nn.Module):
    """Bare-point token embedding for VARIABLE observation designs.

    Input contract: tokens are (t, x) pairs -- feature 0 = time normalized to
    ~[0,1], feature 1 = value -- sorted by t, valid-first when padded. K may
    differ per dataset (fit's (m, tokens, mask) protocol). The embedding forms
    consecutive differences in learnably warped coordinates of BOTH axes:

        dw_i = w_x(x_i) - w_x(x_{i-1})     (value axis, MonotoneWarp)
        g_i  = w_t(t_i - t_{i-1})          (time axis, log-like MonotoneWarp:
                                            dividing by the gap becomes a
                                            subtraction in warped coordinates)

    Token i -> [t_i, w_x(x_i), dw_i, dw_i^2, g_i, first_flag].

    Measured on design-amortized raw GBM (K ~ log-uniform[2,128] in training,
    eval at K=5/9/20/100 vs exact per-design posteriors, B=96): bias ~0 at
    every K with width ratios 1.02/1.02/1.02/1.27 (best of all arms; bare
    points without pair features are also unbiased but 1.05-1.95x wide).
    Combine with rope="time" -- ordinal RoPE encodes a lie (equal spacing) on
    irregular designs and its positional layout becomes a train/eval contract
    that is easy to violate (this exact bug produced spurious +1..+3 sd
    "biases" before being found).
    """

    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.warp_x = MonotoneWarp(8, lo=-3.0, hi=4.0)
        self.warp_t = MonotoneWarp(8, lo=-9.0, hi=0.0)   # gaps ~1e-3..1
        self.norm = FeatureNorm(6)
        self.proj = nn.Linear(6, dim)

    def forward(self, tokens, mask=None):
        t = tokens[..., 0]
        x = tokens[..., 1]
        wx = self.warp_x(x)
        dw = torch.zeros_like(wx)
        dw[:, 1:] = wx[:, 1:] - wx[:, :-1]
        dt = torch.zeros_like(t)
        dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0.0)
        g = self.warp_t(dt)
        first = torch.zeros_like(t)
        first[:, 0] = 1.0
        f = torch.stack([t, wx, dw, dw * dw, g, first], dim=-1)
        return self.proj(self.norm(f, mask))


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(seq_len).float()
    ang = torch.outer(pos, inv_freq)                 # [T, half]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)  # [T, head_dim]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: [B, H, T, Dh]; cos/sin: [T, Dh] (ordinal) or [B, T, Dh] (continuous)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    if cos.dim() == 3:                                # per-batch continuous phases
        return x * cos[:, None] + rot * sin[:, None]
    return x * cos[None, None] + rot * sin[None, None]


def time_rope_tables(t, head_dim, f_min=0.25, f_max=500.0):
    """Continuous RoPE phases from the actual observation times.

    Ordinal RoPE keys the slot index, which on an irregular design encodes a
    lie (equal spacing) and makes the positional layout a train/eval contract.
    Rotating by the physical time instead puts the TRUE relative gap
    t_i - t_j into the attention phases at a geometric bank of frequencies,
    and restores genuine permutation invariance: the phase follows the point,
    not its place in the list.

    t: [B, T] times normalized to ~[0, 1]. Frequencies span periods 1/f_max
    (grid resolution) .. 1/f_min (multiples of the horizon).
    """
    half = head_dim // 2
    k = torch.arange(half, dtype=torch.float32, device=t.device)
    freqs = f_min * (f_max / f_min) ** (k / max(half - 1, 1))
    ang = 2.0 * torch.pi * t[..., None] * freqs        # [B, T, half]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)    # [B, T, head_dim]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    return cos, sin


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin, mask):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)         # each [B, H, T, Dh]
        if cos is not None:                          # RoPE keys the *slot index*;
            q = apply_rope(q, cos, sin)              # for a set input that is not a
            k = apply_rope(k, cos, sin)              # physical coordinate (see below)
        # fused attention: no [B,H,T,T] score matrix is ever materialised, and the
        # bool mask (True = attend) is applied inside the kernel
        am = None if mask is None else mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)   # [B,H,T,Dh]
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h) ** 2                        # ReLU^2
        return self.fc2(h)


class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attention(dim, n_head)
        self.n2 = RMSNorm(dim)
        self.ffn = FFN(dim)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        x = x + self.ffn(self.n2(x))
        return x


class AttentionPool(nn.Module):
    """Pooling by multi-head attention (Set Transformer PMA): learned queries
    attend over the token set to produce the context vector. A strictly larger
    function class than mean-pool -- *when its parameters are trained*.

    `n_query > 1` gives Perceiver-style latents: several queries, concatenated and
    projected, so the summary is not forced through one attention read.

    Two measured caveats (results/DEBUG_encoder.md), both of which apply under the
    default `conditioning="xattn"`:

    - Nothing trains these parameters there. The velocity cross-attends to the token
      memory and the base head reads a *detached* context, so `pool()` sits on a
      dead branch of the graph and this module stays at its random initialization.
    - Even trained, the softmax weights are content-dependent, so an attention read
      is a *nonlinear* function of the token values. Where the statistic to extract
      is a linear functional of them, a masked mean is the exact mechanism and this
      has to approximate it.
    """

    def __init__(self, dim, n_head, n_query: int = 1, out_dim: int = None):
        super().__init__()
        self.n_head = n_head
        self.n_query = n_query
        self.head_dim = dim // n_head
        self.q = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(n_query * dim, out_dim or dim)

    def forward(self, x, mask):
        B, T, D = x.shape
        kv = self.kv(x).reshape(B, T, 2, self.n_head, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)              # each [B, H, T, Dh]
        q = (self.q.reshape(1, self.n_query, self.n_head, self.head_dim)
             .permute(0, 2, 1, 3).expand(B, -1, -1, -1))
        am = None if mask is None else mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)   # [B,H,Q,Dh]
        out = out.permute(0, 2, 1, 3).reshape(B, self.n_query * D)
        return self.proj(out)


class SetTransformer(nn.Module):
    """Encode a token set [B, T, F] -> token memory [B, T, dim] and context [B, ctx_dim].

    Options (see results/DEBUG_encoder.md for the measurements behind them):

    - `embed`: `"linear"` or `"mlp"`. A *linear* embed cannot make a
      token's value interact with its own coordinate feature: it maps
      `[value, index] -> W_v*value + W_i*index`, so the value's coefficient is the
      same for every token and the pooled summary can only see sums of values, not
      general linear functionals of them. The per-token MLP restores the product.
    - `input_norm`: standardize raw features (see `FeatureNorm`).
    - `rope`: rotary embeddings on q/k. The position they encode is the *slot index
      in the concatenated token list*, which for a set input is an artefact of
      packing, not a physical coordinate (tokens carry their own coordinates as
      features). It also breaks permutation invariance.
    - `pool`: `"attn"`, `"mean"` or `"sum"`; `n_query` latents and `pool_dim`
      width for the attention pool.
    - `final_norm`: RMSNorm on the token memory. It divides each token by its own
      norm, which is a *nonlinear*, saturating function of the token's magnitude.
    """

    def __init__(self, n_features: int, dim: int = 64, n_head: int = 4,
                 n_layer: int = 3, max_tokens: int = 512, pool: str = "attn",
                 rope: bool = True, n_query: int = 1, pool_dim: int = None,
                 embed: str = "linear", embed_hidden: int = None,
                 input_norm: bool = True, final_norm: bool = True):
        super().__init__()
        self.dim = dim
        self.pool_mode = pool
        self.use_rope = rope
        self.ctx_dim = (pool_dim or dim) if pool == "attn" else dim
        self.in_norm = FeatureNorm(n_features) if input_norm else None
        if embed in ("wdiff", "wbasis"):
            # the warp must see RAW features (its own FeatureNorm runs after
            # the warp), so the input normalization is disabled for it
            self.in_norm = None
            self.embed = WarpDiffEmbed(n_features, dim,
                                       kind="slog" if embed == "wdiff" else "basis")
        elif embed == "wpair":
            self.in_norm = None
            self.embed = WarpPairEmbed(n_features, dim)
        elif embed == "wfilm":
            self.in_norm = None
            self.embed = SetCondPairEmbed(n_features, dim)
        elif embed == "wpoint":
            self.in_norm = None
            self.embed = PointEmbed(n_features, dim)
        elif embed == "mlp":
            h = embed_hidden or 4 * dim
            self.embed = nn.Sequential(nn.Linear(n_features, h), nn.GELU(),
                                       nn.Linear(h, dim))
        else:
            self.embed = nn.Linear(n_features, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(dim) if final_norm else nn.Identity()
        self.attn_pool = (AttentionPool(dim, n_head, n_query, pool_dim)
                          if pool == "attn" else None)
        self.head_dim = dim // n_head
        if rope is True:
            cos, sin = rope_tables(max_tokens, self.head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def _rope(self, T):
        """RoPE tables for T positions, grown on demand.

        The table used to be capped at the observer's token count, so any input
        longer than that crashed -- which is why 'handles an arbitrary number of
        observations' had never actually been exercised."""
        if not self.use_rope:
            return None, None
        if T > self.cos.shape[0]:
            cos, sin = rope_tables(T, self.head_dim)
            self.cos, self.sin = cos.to(self.cos.device), sin.to(self.sin.device)
        return self.cos[:T], self.sin[:T]

    def encode(self, tokens, mask=None):
        """Per-token memory [B, T, dim] (no pooling) -- for cross-attention conditioning."""
        T = tokens.shape[1]
        raw_t = tokens[..., 0]               # feature 0 is normalized time (contract)
        if self.in_norm is not None:
            tokens = self.in_norm(tokens, mask)
        if (isinstance(self.embed, (WarpDiffEmbed, WarpPairEmbed))
                or getattr(self.embed, "takes_mask", False)):
            # any embedding with internal normalization or SET-level context
            # must see the mask: without it, padded zero-slots dilute
            # per-dataset statistics in the padded path but not in the packed
            # path, silently making the two inference paths inequivalent
            x = self.embed(tokens, mask)
        else:
            x = self.embed(tokens)
        if self.use_rope == "time":
            cos, sin = time_rope_tables(raw_t, self.head_dim)
        else:
            cos, sin = self._rope(T)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        return self.norm(x)

    def pool(self, x, mask=None):
        """Collapse token memory [B, T, dim] -> context [B, ctx_dim]."""
        if self.pool_mode == "attn":
            return self.attn_pool(x, mask)
        if mask is None:
            return x.sum(dim=1) if self.pool_mode == "sum" else x.mean(dim=1)
        w = mask.float()[..., None]
        s = (x * w).sum(dim=1)
        return s if self.pool_mode == "sum" else s / w.sum(dim=1).clamp(min=1.0)

    def forward(self, tokens, mask=None):
        return self.pool(self.encode(tokens, mask), mask)
