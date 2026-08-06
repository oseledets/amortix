"""Amortized posterior via Conditional Flow Matching.

Training (per arXiv:2503.01375): draw (m, observation) pairs from prior+simulator,
encode the observation to a context c, draw a base sample z0, form the linear path
z_t = (1-t) z0 + t z1 with z1 the normalized target parameter, and regress the
velocity field on the displacement z1 - z0:

    L = E || v_theta(z_t, t, c) - (z1 - z0) ||^2

**Data-dependent base (`base="data"`, default).** Flow matching can transport any
source to any target, but if the source spread is far from the (conditional)
target spread the deterministic ODE must do a large, stiff contraction/expansion
that a finite network underfits -- which shows up as a mis-calibrated posterior
(too wide or too narrow). So we *align* the source to the target per dataset: a
small head predicts a Gaussian base N(mu_hat(c), s_hat(c)^2) trained by Gaussian
NLL to match the posterior's mean and spread; the flow then only refines the
*shape*. The NLL keeps s_hat ~ the true posterior std (an ODE cannot create
spread from a point, so the base must seed it). `base="standard"` recovers the
plain N(0, I) source.

Inference: encode c, draw z0 from the (data-dependent) base, integrate
dz/dt = v_theta(z, t, c) from t=0 to 1, denormalize. One ODE solve per dataset.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from .encoder import SetTransformer, RMSNorm, FFN


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class VelocityNet(nn.Module):
    def __init__(self, dim: int, ctx_dim: int, t_dim: int = 64,
                 hidden: int = 256, depth: int = 3):
        super().__init__()
        self.t_dim = t_dim
        layers = [nn.Linear(dim + t_dim + ctx_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t, ctx):
        te = timestep_embedding(t, self.t_dim)
        return self.net(torch.cat([z, te, ctx], dim=-1))


def _modulate(x, shift, scale):
    # x [B, P, dim]; shift/scale [B, dim]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttention(nn.Module):
    """Parameter tokens (queries) attend to the observation token memory."""

    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.hd = dim // n_head
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def kv_from(self, mem):                      # mem [b, T, dim] -> k,v [b, H, T, hd]
        b, T, _ = mem.shape
        kv = self.kv(mem).reshape(b, T, 2, self.n_head, self.hd)
        k, v = kv.permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, k, v):                  # x [B,P,dim]; k,v [b,H,T,hd] (b broadcasts)
        B, P, D = x.shape
        q = self.q(x).reshape(B, P, self.n_head, self.hd).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)      # [B,H,P,T]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, P, D)
        return self.proj(out)


class CrossBlock(nn.Module):
    """adaLN-Zero cross-attention block (DiT-style): time modulates, params attend
    to the full observation memory -- no single-vector bottleneck."""

    def __init__(self, dim, n_head):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.cross = CrossAttention(dim, n_head)
        self.n2 = RMSNorm(dim)
        self.ffn = FFN(dim)
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)          # adaLN-Zero: blocks start as identity
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, k, v, temb):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(temb).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.cross(_modulate(self.n1(x), sh1, sc1), k, v)
        x = x + g2.unsqueeze(1) * self.ffn(_modulate(self.n2(x), sh2, sc2))
        return x


class CrossCondVelocity(nn.Module):
    """Velocity field conditioned on the observation tokens by cross-attention.

    One token per parameter (carrying that parameter's value) cross-attends to the
    encoder's per-observation memory at every block, with adaLN(t) modulation. The
    K,V of the memory depend only on the dataset, so they are computed once per
    dataset and reused across the whole ODE solve (`encode_memory`)."""

    def __init__(self, d_param, dim, n_head=4, n_layer=3, t_dim=64):
        super().__init__()
        self.d_param = d_param
        self.t_dim = t_dim
        self.param_emb = nn.Parameter(torch.randn(d_param, dim) * 0.02)
        self.val_in = nn.Linear(1, dim)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([CrossBlock(dim, n_head) for _ in range(n_layer)])
        self.out_norm = RMSNorm(dim)
        self.out = nn.Linear(dim, 1)

    def encode_memory(self, memory):             # -> list of (k,v) per block
        return [blk.cross.kv_from(memory) for blk in self.blocks]

    def forward(self, z, t, cache):
        tok = self.param_emb.unsqueeze(0) + self.val_in(z.unsqueeze(-1))  # [B,d,dim]
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))              # [B,dim]
        for blk, (k, v) in zip(self.blocks, cache):
            tok = blk(tok, k, v, temb)
        return self.out(self.out_norm(tok)).squeeze(-1)                   # [B,d]


class BaseHead(nn.Module):
    """Predicts a data-dependent Gaussian base N(mu, s^2) from the context."""

    def __init__(self, ctx_dim: int, dim: int):
        super().__init__()
        self.net = nn.Linear(ctx_dim, 2 * dim)
        self.dim = dim
        # init: mu ~ 0, log_s ~ 0  (base ~ N(0, I) at start of training)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, ctx):
        h = self.net(ctx)
        mu, log_s = h[:, :self.dim], h[:, self.dim:]
        log_s = log_s.clamp(-4.0, 2.0)
        return mu, torch.exp(log_s)


class FlowPosterior(nn.Module):
    def __init__(self, problem, dim_model: int = 64, n_head: int = 4,
                 n_layer: int = 3, hidden: int = 256, depth: int = 3,
                 pool: str = "attn", base: str = "data", conditioning: str = "xattn"):
        super().__init__()
        self.problem = problem
        self.prior = problem.prior
        self.d = self.prior.dim
        self.base = base
        self.conditioning = conditioning
        self.encoder = SetTransformer(
            n_features=problem.observer.N_FEATURES,
            dim=dim_model, n_head=n_head, n_layer=n_layer,
            max_tokens=problem.observer.n_tokens + 8, pool=pool,
        )
        if conditioning == "xattn":
            # dense conditioning: velocity cross-attends to the token memory
            self.velocity = CrossCondVelocity(self.d, dim_model, n_head=n_head, n_layer=depth)
        else:
            # concat conditioning: velocity MLP on [z, emb(t), pooled context]
            self.velocity = VelocityNet(self.d, dim_model, hidden=hidden, depth=depth)
        self.base_head = BaseHead(dim_model, self.d) if base == "data" else None

    # --- training --------------------------------------------------------
    def fit(self, n_train: int = 12000, epochs: int = 30, batch: int = 256,
            lr: float = 3e-4, base_weight: float = 1.0, seed: int = 0,
            verbose: bool = True):
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        if verbose:
            print(f"[fit] simulating {n_train} training trajectories...")
        m, tokens = self.problem.simulate(n_train, generator=gen)
        z1 = self.prior.normalize(m)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n_batches = max(1, n_train // batch)
        t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(n_train, generator=gen)
            running = 0.0
            for b in range(n_batches):
                idx = perm[b * batch:(b + 1) * batch]
                zb, tb = z1[idx], tokens[idx]
                bs = zb.shape[0]
                memory = self.encoder.encode(tb)
                ctx = self.encoder.pool(memory)
                eps = torch.randn(bs, self.d, generator=gen)
                if self.base == "data":
                    mu, s = self.base_head(ctx)
                    z0 = mu + s * eps
                    base_nll = (0.5 * ((zb - mu) ** 2 / (s ** 2) + 2 * torch.log(s))).mean()
                else:
                    z0 = eps
                    base_nll = torch.zeros(())
                t = torch.rand(bs, generator=gen)
                zt = (1 - t)[:, None] * z0 + t[:, None] * zb
                target = zb - z0
                cond = self.velocity.encode_memory(memory) if self.conditioning == "xattn" else ctx
                pred = self.velocity(zt, t, cond)
                cfm = ((pred - target) ** 2).mean()
                loss = cfm + base_weight * base_nll
                opt.zero_grad(); loss.backward(); opt.step()
                running += cfm.item()
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  cfm {running / n_batches:.4f}"
                      f"  ({time.time() - t0:.1f}s)")
        return self

    # --- inference -------------------------------------------------------
    @torch.no_grad()
    def sample(self, tokens: torch.Tensor, n: int = 2000, n_steps: int = 60,
               seed: int = 0) -> torch.Tensor:
        """Posterior samples for one observation. tokens: [T, F] or [1, T, F]."""
        if tokens.dim() == 2:
            tokens = tokens[None]
        gen = torch.Generator().manual_seed(seed)
        memory = self.encoder.encode(tokens)           # [1, T, dim]
        ctx = self.encoder.pool(memory)                # [1, dim]
        eps = torch.randn(n, self.d, generator=gen)
        if self.base == "data":
            mu, s = self.base_head(ctx)                 # [1, d] -> broadcast over n
            z = mu + s * eps
        else:
            z = eps
        # conditioning computed once per dataset and reused across the ODE solve
        cond = self.velocity.encode_memory(memory) if self.conditioning == "xattn" \
            else ctx.expand(n, -1)
        dt = 1.0 / n_steps
        for i in range(n_steps):                       # RK4 ODE integration
            t = torch.full((n,), i * dt)
            k1 = self.velocity(z, t, cond)
            k2 = self.velocity(z + 0.5 * dt * k1, t + 0.5 * dt, cond)
            k3 = self.velocity(z + 0.5 * dt * k2, t + 0.5 * dt, cond)
            k4 = self.velocity(z + dt * k3, t + dt, cond)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.prior.denormalize(z)
