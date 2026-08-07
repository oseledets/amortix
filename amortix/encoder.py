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


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(seq_len).float()
    ang = torch.outer(pos, inv_freq)                 # [T, half]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)  # [T, head_dim]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: [B, H, T, Dh]; cos/sin: [T, Dh]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, None] + rot * sin[None, None]


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
        if embed == "mlp":
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
        if rope:
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
        if self.in_norm is not None:
            tokens = self.in_norm(tokens, mask)
        x = self.embed(tokens)
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
