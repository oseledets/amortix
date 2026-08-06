"""Transformer set-encoder for observations.

Follows the architecture choices of arXiv:2503.01375: bidirectional self-attention
with rotary position embeddings (RoPE), ReLU^2 feed-forward, and RMS normalization.
A masked mean-pool produces a single conditioning vector for the velocity field.
The encoder is permutation-equivariant up to RoPE and handles a variable number
of observation tokens via the padding mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


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
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,T,T]
        if mask is not None:
            keymask = (~mask)[:, None, None, :]      # [B,1,1,T]
            attn = attn.masked_fill(keymask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = attn @ v                                # [B,H,T,Dh]
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


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
    """Pooling by multi-head attention (Set Transformer PMA): a learned query
    attends over the token set to produce the context vector. Strictly more
    expressive than mean-pool -- it can keep per-parameter information instead of
    averaging it away, which matters for posterior calibration."""

    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        B, T, D = x.shape
        kv = self.kv(x).reshape(B, T, 2, self.n_head, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)              # each [B, H, T, Dh]
        q = self.q.reshape(1, self.n_head, 1, self.head_dim).expand(B, -1, -1, -1)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,1,T]
        if mask is not None:
            attn = attn.masked_fill((~mask)[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, D)                # [B, D]
        return self.proj(out)


class SetTransformer(nn.Module):
    """Encode a token set [B, T, F] -> context vector [B, dim]."""

    def __init__(self, n_features: int, dim: int = 64, n_head: int = 4,
                 n_layer: int = 3, max_tokens: int = 512, pool: str = "attn"):
        super().__init__()
        self.dim = dim
        self.pool_mode = pool
        self.embed = nn.Linear(n_features, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(dim)
        self.attn_pool = AttentionPool(dim, n_head) if pool == "attn" else None
        cos, sin = rope_tables(max_tokens, dim // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def encode(self, tokens, mask=None):
        """Per-token memory [B, T, dim] (no pooling) -- for cross-attention conditioning."""
        T = tokens.shape[1]
        x = self.embed(tokens)
        cos, sin = self.cos[:T], self.sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        return self.norm(x)

    def pool(self, x, mask=None):
        """Collapse token memory [B, T, dim] -> context [B, dim]."""
        if self.pool_mode == "attn":
            return self.attn_pool(x, mask)
        if mask is None:
            return x.mean(dim=1)
        w = mask.float()[..., None]
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    def forward(self, tokens, mask=None):
        return self.pool(self.encode(tokens, mask), mask)
