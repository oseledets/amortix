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


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1000.0,
                       scale: float = 1000.0):
    """Sinusoidal embedding of the flow time t.

    `scale` maps our continuous t in [0,1] onto the range these frequencies were
    designed for. Without it every argument stays below 1 radian, so the sinusoids
    never complete a cycle and ~3/4 of the dimensions are effectively constant --
    the velocity field then receives almost no time signal.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = (t * scale)[:, None] * freqs[None]
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

    def forward_grouped(self, z, t, ctx):
        """z [B, G, d], t [B], ctx [B, ctx_dim] -> [B, G, d] (G samples per dataset)."""
        B, G, d = z.shape
        te = timestep_embedding(t, self.t_dim)[:, None].expand(B, G, -1)
        ctx = ctx[:, None].expand(B, G, -1)
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


class SelfAttention(nn.Module):
    """Attention among the parameter tokens of one posterior sample."""

    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.hd = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                        # x [N, P, dim]
        N, P, D = x.shape
        qkv = self.qkv(x).reshape(N, P, 3, self.n_head, self.hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) / (self.hd ** 0.5)).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(N, P, D))


class CrossBlock(nn.Module):
    """adaLN-Zero block: parameter tokens talk to each other (self-attention),
    then read the observation memory (cross-attention), with time modulation.

    The self-attention is essential, not decorative. Without it the velocity for
    parameter i depends only on z_i, so the field is factorized coordinate-wise --
    and an ODE with a coordinate-wise field maps independent coordinates to
    independent coordinates. Such a flow *cannot* produce a correlated posterior
    from an independent base, at any budget. Attention runs within one posterior
    sample only, so distinct samples stay independent.
    """

    def __init__(self, dim, n_head):
        super().__init__()
        self.n0 = RMSNorm(dim)
        self.selfattn = SelfAttention(dim, n_head)
        self.n1 = RMSNorm(dim)
        self.cross = CrossAttention(dim, n_head)
        self.n2 = RMSNorm(dim)
        self.ffn = FFN(dim)
        self.ada = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.ada.weight)          # adaLN-Zero: blocks start as identity
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, k, v, temb, n_param):
        """x [B, G*P, dim]; temb [B, dim]; P = n_param parameter tokens per sample."""
        B, GP, D = x.shape
        G = GP // n_param
        sh0, sc0, g0, sh1, sc1, g1, sh2, sc2, g2 = self.ada(temb).chunk(9, dim=-1)
        # --- parameter tokens attend to each other, within each sample ---
        h = _modulate(self.n0(x), sh0, sc0).reshape(B * G, n_param, D)
        h = self.selfattn(h).reshape(B, GP, D)
        x = x + g0.unsqueeze(1) * h
        # --- parameter tokens read the observation memory ---
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
            tok = blk(tok, k, v, temb, self.d_param)
        return self.out(self.out_norm(tok)).squeeze(-1)                   # [B,d]

    def forward_grouped(self, z, t, cache):
        """z [B, G, d], t [B], cache from a [B, T, dim] memory -> [B, G, d].

        G posterior samples per dataset are folded into the query axis, so the
        per-dataset K,V are used as-is (no replication): attention is [B,H,G*d,T].
        """
        B, G, d = z.shape
        tok = self.param_emb.view(1, 1, d, -1) + self.val_in(z.unsqueeze(-1))
        tok = tok.reshape(B, G * d, -1)                                   # [B,G*d,dim]
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))              # [B,dim]
        for blk, (k, v) in zip(self.blocks, cache):
            tok = blk(tok, k, v, temb, d)
        return self.out(self.out_norm(tok)).reshape(B, G, d)


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

    def nll(self, z1, ctx):
        """Gaussian NLL of the targets under the predicted base, and a sampler."""
        mu, s = self(ctx)
        nll = (0.5 * ((z1 - mu) ** 2 / s ** 2 + 2 * torch.log(s))).sum(-1).mean()
        return nll, (lambda eps: mu + s * eps), (mu, s)


class FullBaseHead(nn.Module):
    """Data-dependent Gaussian base with a **full covariance**: N(mu, L L^T).

    The diagonal head can only seed axis-aligned spread, so any posterior
    correlation has to be manufactured by the flow itself -- the suspected cause
    of the SBC failures on coupled parameters (Lotka-Volterra alpha/beta, SEIR
    beta2/gamma_d, CIR a/b, which enter the dynamics as products). Predicting a
    Cholesky factor lets the base match the posterior's covariance directly, so
    the flow only has to fix non-Gaussian shape.
    """

    def __init__(self, ctx_dim: int, dim: int):
        super().__init__()
        self.dim = dim
        self.n_off = dim * (dim - 1) // 2
        self.net = nn.Linear(ctx_dim, 2 * dim + self.n_off)
        nn.init.zeros_(self.net.weight)          # start at N(0, I)
        nn.init.zeros_(self.net.bias)
        idx = torch.tril_indices(dim, dim, offset=-1)
        self.register_buffer("off_i", idx[0], persistent=False)
        self.register_buffer("off_j", idx[1], persistent=False)

    def forward(self, ctx):
        h = self.net(ctx)
        B = h.shape[0]
        mu = h[:, :self.dim]
        log_d = h[:, self.dim:2 * self.dim].clamp(-4.0, 2.0)
        off = h[:, 2 * self.dim:]
        L = torch.zeros(B, self.dim, self.dim, dtype=h.dtype, device=h.device)
        L[:, range(self.dim), range(self.dim)] = torch.exp(log_d)
        if self.n_off:
            L[:, self.off_i, self.off_j] = off
        return mu, L

    def nll(self, z1, ctx):
        """Exact multivariate Gaussian NLL: 0.5||L^-1 (z1-mu)||^2 + log|det L|."""
        mu, L = self(ctx)
        u = torch.linalg.solve_triangular(L, (z1 - mu).unsqueeze(-1), upper=False)
        logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        nll = (0.5 * (u ** 2).sum((-2, -1)) + logdet).mean()
        return nll, (lambda eps: mu + torch.einsum("bij,bj->bi", L, eps)), (mu, L)


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
        if base == "data":
            self.base_head = BaseHead(dim_model, self.d)          # diagonal Gaussian
        elif base == "full":
            self.base_head = FullBaseHead(dim_model, self.d)      # full covariance
        else:
            self.base_head = None                                  # plain N(0, I)

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
                if self.base_head is not None:
                    base_nll, draw, _ = self.base_head.nll(zb, ctx)
                    # The base is trained by its NLL only. z0 is DETACHED from the
                    # CFM term: otherwise ||v - (z1 - z0)||^2 can be reduced by
                    # dragging z0 towards z1, i.e. the loss would pay the base to
                    # absorb the flow's job, collapsing the regression target to
                    # zero and leaving the velocity field with nothing to learn.
                    z0 = draw(eps).detach()
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
    def sample_batch(self, tokens: torch.Tensor, n: int = 1000, n_steps: int = 20,
                     seed: int = 0, chunk: int = 16, solver: str = "midpoint") -> torch.Tensor:
        """Posterior samples for a *batch* of observations. tokens [B, T, F] -> [B, n, d].

        All B datasets are encoded together and their ODEs solved jointly, which is
        what makes calibration studies (hundreds of datasets) tractable: one solve
        instead of B python-loop solves. `chunk` bounds peak attention memory.
        """
        gen = torch.Generator().manual_seed(seed)
        outs = []
        for start in range(0, tokens.shape[0], chunk):
            tb = tokens[start:start + chunk]
            B = tb.shape[0]
            memory = self.encoder.encode(tb)                  # [B, T, dim]
            ctx = self.encoder.pool(memory)                   # [B, dim]
            eps = torch.randn(B, n, self.d, generator=gen)
            if self.base == "full":
                mu, L = self.base_head(ctx)                   # [B,d], [B,d,d]
                z = mu[:, None] + torch.einsum("bij,bnj->bni", L, eps)
            elif self.base_head is not None:
                mu, s = self.base_head(ctx)                   # [B, d]
                z = mu[:, None] + s[:, None] * eps
            else:
                z = eps
            # conditioning computed once per dataset, reused across the whole solve
            grouped = self.velocity.forward_grouped
            cond = self.velocity.encode_memory(memory) if self.conditioning == "xattn" else ctx
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((B,), i * dt)
                if solver == "euler":                         # 1 eval / step
                    z = z + dt * grouped(z, t, cond)
                elif solver == "midpoint":                    # 2 evals / step
                    k1 = grouped(z, t, cond)
                    z = z + dt * grouped(z + 0.5 * dt * k1, t + 0.5 * dt, cond)
                else:                                         # rk4: 4 evals / step
                    k1 = grouped(z, t, cond)
                    k2 = grouped(z + 0.5 * dt * k1, t + 0.5 * dt, cond)
                    k3 = grouped(z + 0.5 * dt * k2, t + 0.5 * dt, cond)
                    k4 = grouped(z + dt * k3, t + dt, cond)
                    z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            outs.append(self.prior.denormalize(z))
        return torch.cat(outs, dim=0)

    def sample(self, tokens: torch.Tensor, n: int = 2000, n_steps: int = 20,
               seed: int = 0, solver: str = "midpoint") -> torch.Tensor:
        """Posterior samples for one observation. tokens: [T, F] or [1, T, F] -> [n, d]."""
        if tokens.dim() == 2:
            tokens = tokens[None]
        return self.sample_batch(tokens, n=n, n_steps=n_steps, seed=seed, solver=solver)[0]
