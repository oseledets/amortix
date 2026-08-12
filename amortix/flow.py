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

import contextlib
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SetTransformer, RMSNorm, FFN


class EMA:
    """Exponential moving average of the weights, evaluated instead of the raw ones.

    SGD on a noisy objective does not converge to a point, it *random-walks in a
    ball* around the optimum whose radius is set by lr x gradient noise. The CFM
    target is exceptionally noisy (most of ||v - (z1-z0)||^2 is the irreducible
    Var(z1-z0 | z_t)), so that ball is wide and any single iterate is a lottery
    ticket -- which is exactly the oscillating plateau seen on the monitored
    metric. Averaging the iterates cancels the walk (Polyak-Ruppert) and lands
    near the centre of the ball.

    `decay` is ramped in as min(decay, (1+n)/(10+n)) so the average is not held
    back by the (meaningless) initial weights during the first few hundred steps.
    """

    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.module = module
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone() for k, v in module.named_parameters()}

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, p in self.module.named_parameters():
            self.shadow[k].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextlib.contextmanager
    def averaged(self):
        """Temporarily swap the averaged weights into the module."""
        backup = {k: p.detach().clone() for k, p in self.module.named_parameters()}
        with torch.no_grad():
            for k, p in self.module.named_parameters():
                p.copy_(self.shadow[k])
        try:
            yield self.module
        finally:
            with torch.no_grad():
                for k, p in self.module.named_parameters():
                    p.copy_(backup[k])

    @torch.no_grad()
    def store_in_module(self):
        """Write the average into the module for good (end of training)."""
        for k, p in self.module.named_parameters():
            p.copy_(self.shadow[k])


def lr_at(step: int, total: int, lr: float, warmup: int = 0,
          schedule: str = "cosine", min_lr_frac: float = 0.0) -> float:
    """Learning rate at `step` (0-based) of a `total`-step run.

    Linear warmup then cosine decay to `min_lr_frac * lr`. Decaying the step size
    shrinks the noise ball the iterates wander in; warmup keeps the first updates
    from moving the (adaLN-zero initialised) network before Adam's second-moment
    estimate is meaningful.
    """
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        p = min(max(p, 0.0), 1.0)
        return lr * (min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))
    return lr                                        # schedule == "constant"


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1000.0,
                       scale: float = 1000.0):
    """Sinusoidal embedding of the flow time t.

    `scale` maps our continuous t in [0,1] onto the range these frequencies were
    designed for. Without it every argument stays below 1 radian, so the sinusoids
    never complete a cycle and ~3/4 of the dimensions are effectively constant --
    the velocity field then receives almost no time signal.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = (t * scale)[:, None] * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def pack_tokens(token_sets):
    """Pack observation sets of DIFFERENT lengths into (tokens, mask).

    `token_sets` is a list of [T_i, F] tensors. Returns a padded [B, Tmax, F]
    batch plus a boolean [B, Tmax] validity mask. Padding is never read: the mask
    reaches the encoder's self-attention, the pooling and the velocity's
    cross-attention, so a dataset's posterior does not depend on how much padding
    its neighbours in the batch needed.

    (This is padding-with-masking, not kernel-level packing. True varlen packing
    only pays off with a fused attention kernel; with a plain PyTorch matmul a
    flat packed sequence of length sum(T_i) would cost more, not less, than the
    padded [B, Tmax] layout.)
    """
    B = len(token_sets)
    Tmax = max(t.shape[0] for t in token_sets)
    n_feat = token_sets[0].shape[1]
    out = torch.zeros(B, Tmax, n_feat, dtype=token_sets[0].dtype)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    for i, t in enumerate(token_sets):
        out[i, :t.shape[0]] = t
        mask[i, :t.shape[0]] = True
    return out, mask


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

    def forward(self, z, t, ctx, mask=None):
        te = timestep_embedding(t, self.t_dim)
        return self.net(torch.cat([z, te, ctx], dim=-1))

    def forward_grouped(self, z, t, ctx, mask=None):
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

    def forward(self, x, k, v, mask=None):       # x [B,P,dim]; k,v [b,H,T,hd]
        B, P, D = x.shape
        q = self.q(x).reshape(B, P, self.n_head, self.hd).permute(0, 2, 1, 3)
        # The mask matters here, not only in the encoder: without it padded
        # observation slots leak straight into the velocity field.
        am = None if mask is None else mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)   # [B,H,P,hd]
        return self.proj(out.permute(0, 2, 1, 3).reshape(B, P, D))


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
        out = F.scaled_dot_product_attention(q, k, v)                 # [N,H,P,hd]
        return self.proj(out.transpose(1, 2).reshape(N, P, D))


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

    def forward(self, x, k, v, temb, n_param, mask=None):
        """x [B, G*P, dim]; temb [B, dim]; P = n_param parameter tokens per sample."""
        B, GP, D = x.shape
        G = GP // n_param
        sh0, sc0, g0, sh1, sc1, g1, sh2, sc2, g2 = self.ada(temb).chunk(9, dim=-1)
        # --- parameter tokens attend to each other, within each sample ---
        h = _modulate(self.n0(x), sh0, sc0).reshape(B * G, n_param, D)
        h = self.selfattn(h).reshape(B, GP, D)
        x = x + g0.unsqueeze(1) * h
        # --- parameter tokens read the observation memory ---
        x = x + g1.unsqueeze(1) * self.cross(_modulate(self.n1(x), sh1, sc1), k, v, mask)
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

    def forward(self, z, t, cache, mask=None):
        tok = self.param_emb.unsqueeze(0) + self.val_in(z.unsqueeze(-1))  # [B,d,dim]
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))              # [B,dim]
        for blk, (k, v) in zip(self.blocks, cache):
            tok = blk(tok, k, v, temb, self.d_param, mask)
        return self.out(self.out_norm(tok)).squeeze(-1)                   # [B,d]

    def forward_grouped(self, z, t, cache, mask=None):
        """z [B, G, d], t [B], cache from a [B, T, dim] memory -> [B, G, d].

        G posterior samples per dataset are folded into the query axis, so the
        per-dataset K,V are used as-is (no replication): attention is [B,H,G*d,T].
        """
        B, G, d = z.shape
        tok = self.param_emb.view(1, 1, d, -1) + self.val_in(z.unsqueeze(-1))
        tok = tok.reshape(B, G * d, -1)                                   # [B,G*d,dim]
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))              # [B,dim]
        for blk, (k, v) in zip(self.blocks, cache):
            tok = blk(tok, k, v, temb, d, mask)
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
        nll = (0.5 * ((z1 - mu) ** 2 / s ** 2 + 2 * torch.log(s))).mean()
        return nll, (lambda eps: mu + s * eps), (mu, s)


class FlowPosterior(nn.Module):
    def __init__(self, problem, dim_model: int = 64, n_head: int = 4,
                 n_layer: int = 3, hidden: int = 256, depth: int = 3,
                 pool: str = "attn", base: str = "standard", conditioning: str = "xattn",
                 embed: str = "auto", rope="auto"):
        """Non-default option values are kept as REPRODUCIBLE CONTROLS for the
        measurements in CALIBRATION.md, not as recommendations: base="data"
        loses to the plain N(0,I) source 1.75x on the exact-posterior testbed;
        conditioning="concat" underperforms dense cross-attention; ordinal
        rope=True on variable designs turns the token layout into a fragile
        train/eval contract. The defaults encode everything this repository
        has measured to be right."""
        super().__init__()
        self.problem = problem
        self.prior = problem.prior
        self.d = self.prior.dim
        self.base = base
        self.conditioning = conditioning
        from .designs import DesignProblem
        is_design = isinstance(problem, DesignProblem)
        if embed == "auto":
            # SDE problems tokenize [t, x, dx, dx^2, res, cid] (PathObserver);
            # for them the default is the universal learnable warped-increment
            # embedding, which handles floating multiplicative scale with no
            # hand-crafted observer transforms (see WarpDiffEmbed).
            # Variable-design problems (DesignProblem) default to the
            # set-conditioned pair embedding, the measured best across the
            # design zoos (see SetCondPairEmbed / CALIBRATION.md). Other
            # observers (ODE/time-series, custom layouts) get the plain
            # linear embedding.
            from .sde import PathObserver
            if is_design:
                # class rule, verified from both sides (CALIBRATION.md):
                # consecutive-pair structure is sufficient iff the observed
                # process is Markov; otherwise bare points are the correct
                # (and measured-best) token.
                embed = ("wfilm" if getattr(problem, "markov_observed", False)
                         else "wpoint")
            elif isinstance(problem.observer, PathObserver):
                embed = "wbasis"
            else:
                embed = "linear"
        self.embed_mode = embed
        if rope == "auto":
            # ordinal RoPE is only meaningful when every dataset shares one
            # fixed token layout. rope="time" (continuous phases from the
            # actual observation times, feature 0) is the honest choice for
            # variable/irregular designs.
            rope = "time" if is_design else True
        self.encoder = SetTransformer(
            n_features=problem.observer.N_FEATURES,
            dim=dim_model, n_head=n_head, n_layer=n_layer,
            max_tokens=problem.observer.n_tokens + 8, pool=pool,
            embed=embed, rope=rope,
        )
        if conditioning == "xattn":
            # dense conditioning: velocity cross-attends to the token memory
            self.velocity = CrossCondVelocity(self.d, dim_model, n_head=n_head, n_layer=depth)
        else:
            # concat conditioning: velocity MLP on [z, emb(t), pooled context]
            self.velocity = VelocityNet(self.d, dim_model, hidden=hidden, depth=depth)
        if base == "data":
            self.base_head = BaseHead(dim_model, self.d)          # diagonal Gaussian
        else:
            self.base_head = None                                  # plain N(0, I)

    # --- training --------------------------------------------------------
    def fit(self, n_train: int = 12000, epochs: int = 30, batch: int = 64,
            lr: float = 3e-4, base_weight: float = 1.0, seed: int = 0,
            token_dropout: float = 0.0, steps: int = None,
            schedule: str = "cosine", warmup: int = 300, min_lr_frac: float = 0.0,
            ema_decay: float = 0.999, grad_clip: float = 10.0,
            betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
            k_pairs: int = 4, t_strat: bool = True,
            monitor=None, monitor_every: int = 500, verbose: bool = True,
            device: str = "auto", retokenize=None):
        """Train the amortized posterior.

        device: "auto" uses CUDA when available (simulation stays on CPU; the
        model and training batches move to the device), otherwise stays where
        the model is. RNG is kept on CPU generators for cross-device
        reproducibility of the data stream; per-batch noise is copied over.

        retokenize: callable (raw_batch, generator) -> (tokens, mask). When
        given, problem.simulate(n) must return (m, raw) with the RAW
        trajectories, and every optimizer step draws FRESH observation designs
        from them: one simulation serves unboundedly many designs, each
        trajectory is seen with a different subset every step (kills the
        design<->dataset co-adaptation of the fixed-design protocol), and the
        subset is drawn BEFORE tokenization, so consecutive-difference
        embeddings (wpair) stay exact. This is the training mode for
        all-subset amortization / downstream optimal experimental design.

        What actually governs convergence here is the number of OPTIMIZER STEPS,
        not the number of simulations or epochs. Holding n_train=6000 and
        epochs=20 fixed on the linear-Gaussian testbed and changing only the batch
        size:

            batch 1024 ->  100 steps -> posterior 5.46x too wide
            batch  256 ->  460 steps -> posterior 4.17x too wide
            batch   64 -> 1860 steps -> posterior 1.21x too wide

        The old default of 256 was starving the model: a "3000 sims / 8 epochs"
        configuration is 88 steps, which is an untrained network, and results
        measured there say nothing about the method. Hence batch=64 by default.

        `steps` overrides `epochs` and is the honest way to state a budget: the
        number of epochs is then whatever it takes to run that many updates.

        Do not judge convergence by the loss. Most of ||v - (z1 - z0)||^2 is the
        irreducible Var(z1 - z0 | z_t); the loss can rise while the posterior
        keeps improving. Track the distribution (examples/budget_curve.py).

        token_dropout: fraction of observation tokens randomly masked out each
        batch, which is what makes the set encoder's size-invariance real rather
        than nominal -- a model trained at one fixed count has never been asked
        to generalise over count.

        monitor: callable(self) -> float, evaluated every `monitor_every`
        optimizer steps and appended to `self.history`. This is where the real
        convergence signal goes: the distance between the sampled posterior and a
        reference posterior, in full dimension (see amortix.metrics). The training
        loss cannot serve -- most of ||v - (z1 - z0)||^2 is the irreducible
        Var(z1 - z0 | z_t), so it can rise while the posterior keeps improving.

        --- optimization (measured on `linear_gaussian`, exact posterior known;
        results/DEBUG_optimization.md) -------------------------------------

        At a constant learning rate the monitored distance to the exact posterior
        stops improving after ~2000 steps and then *oscillates* between 0.007 and
        0.025 forever: the iterates random-walk in a ball around the optimum whose
        radius is set by lr x gradient noise, and the CFM gradient is almost all
        noise. Three knobs shrink that ball (4000 steps, 2 seeds, distance to the
        exact posterior; baseline 0.0178):

        `ema_decay` -- Polyak average of the weights, evaluated AND shipped
            instead of the raw iterate. Alone: 0.0044 (4.1x). It does not just
            lower the number, it makes the curve monotone -- the "plateau" was
            never a plateau, the model was improving underneath the noise.
            0.99 and 0.999 are equivalent; 0.9995 is slightly worse.
        `schedule="cosine"` + `warmup` -- decays the step size to ~0 so the walk
            contracts. Alone: 0.0075 (2.4x). It costs distance travelled, so at a
            short budget it is worse *combined* with EMA than EMA alone; at 12000
            steps it wins (0.0036 vs 0.0040) and, unlike a constant rate, the last
            check is the best one -- the run actually settles.
        `k_pairs` -- how many (t, z0) CFM pairs are drawn per *encoded batch*. The
            observation memory is tiled, not re-encoded, so all k pairs push
            gradients through one encoder pass and the noisiest term of the
            gradient is averaged k times. With cosine+EMA at 4000 steps,
            k=1/4/16 -> 0.0084 / 0.0044 / 0.0025 for 1.0x / 1.9x / 4.5x the
            wallclock per step (pairs 2..k cost ~23% of the first). Saturates
            around k=16; k=64 is not worth it.

        Combined at 12000 steps, on 32 datasets that took part in no decision:
        0.0023 +- 0.0005 (8 runs) against a Monte-Carlo floor of 0.00062 -- 5x
        better than the same budget without any of this (0.0114, 3 runs).
        Refining the ODE solver does not move it (midpoint/20 = RK4/50 to 0.5%),
        and neither does doubling the budget, so what is left is the model.

        `grad_clip` bounds a genuine blow-up (the Gaussian NLL of the base head
        can spike) and must therefore sit ABOVE the working gradient norm, or it
        silently becomes a learning-rate cut instead of a safety net: measured
        median norms are 1.5-3.1 with 90th percentiles 4.3-6.4 across
        linear_gaussian / ou / cir / stoch_lv, so the customary `1.0` fires on
        73-100% of steps. Clipping at 1.0 / 0.5 / not at all / not at all with
        lr halved all land inside the same seed band here (per-run 0.0017-0.0030),
        so the default 10.0 -- above every observed norm (max 8.1) -- is chosen on
        mechanism, not on the metric.
        `betas`, `eps`, `weight_decay` are the usual Adam knobs.
        `t_strat` puts one t in each 1/k-wide bin instead of k independent
        uniforms -- measured neutral here, kept because it cannot cost anything.
        """
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        if verbose:
            print(f"[fit] simulating {n_train} training trajectories "
                  f"({max(1, n_train // batch)} batches/epoch)...")
        sim = self.problem.simulate(n_train, generator=gen)
        data_mask = None
        raw = None
        if retokenize is not None:
            if token_dropout:
                raise ValueError("token_dropout is meaningless with retokenize")
            m, raw = sim[0], sim[1]
            tokens = None
        elif len(sim) == 3:
            # variable-design problems: simulate() -> (m, tokens, mask) with
            # valid tokens first and padding masked out. Design amortization
            # then lives in the DATA (every dataset its own K), which keeps
            # embeddings that difference consecutive tokens correct.
            m, tokens, data_mask = sim
            if token_dropout:
                raise ValueError(
                    "token_dropout cannot be combined with a variable-design "
                    "problem (its mask already defines the design; dropping "
                    "more tokens would break embeddings that difference "
                    "consecutive observations)")
        else:
            m, tokens = sim
        z1 = self.prior.normalize(m)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else None
        dev = torch.device(device) if device else next(self.parameters()).device
        self.to(dev)
        z1 = z1.to(dev)
        if tokens is not None:
            tokens = tokens.to(dev)
        if data_mask is not None:
            data_mask = data_mask.to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=tuple(betas), eps=eps,
                               weight_decay=weight_decay)
        n_batches = max(1, n_train // batch)
        if steps is not None:                       # budget stated in optimizer steps
            epochs = max(1, int(round(steps / n_batches)))
        total_steps = epochs * n_batches
        # a fixed warmup must never eat a short budget: at 4000+ steps this is the
        # requested 300, at 500 steps it is 50
        warmup = min(warmup, max(1, total_steps // 10))
        ema = EMA(self, ema_decay) if ema_decay else None
        k = max(1, int(k_pairs))
        self.history = []
        step_now = 0
        t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(n_train, generator=gen)
            running = 0.0
            for b in range(n_batches):
                for group in opt.param_groups:
                    group["lr"] = lr_at(step_now, total_steps, lr, warmup,
                                        schedule, min_lr_frac)
                idx = perm[b * batch:(b + 1) * batch]
                zb = z1[idx]
                bs = zb.shape[0]
                if retokenize is not None:
                    tb, mb = retokenize(raw[idx], gen)
                    tb, mb = tb.to(dev), mb.to(dev)
                else:
                    tb = tokens[idx]
                    mb = None if data_mask is None else data_mask[idx]
                if token_dropout == "design":
                    # design amortization: every example keeps K_i ~ log-uniform
                    # in [2, T] randomly chosen tokens, so ONE network learns
                    # p(m | any observation design) -- 5 points or 100, dense or
                    # sparse -- instead of the single design the observer emits.
                    T = tb.shape[1]
                    u = torch.rand(bs, generator=gen)
                    ki = (2.0 * (T / 2.0) ** u).long().clamp(2, T)     # [bs]
                    r = torch.rand(bs, T, generator=gen)
                    kth = r.sort(dim=1).values.gather(1, (ki - 1)[:, None])
                    keep = r <= kth                                     # K_i kept
                    # COMPACT kept tokens to the front (stable order): the
                    # encoder uses RoPE over token positions, and inference
                    # packs a K-point design contiguously from position 0 --
                    # training must present the same positional geometry, or
                    # sparse designs are out-of-distribution at eval.
                    order = torch.argsort(~keep, dim=1, stable=True)
                    tb = tb.gather(1, order[..., None].expand_as(tb).to(dev))
                    mb = keep.gather(1, order).to(dev)
                elif token_dropout > 0:
                    keep = torch.rand(tb.shape[:2], generator=gen) >= token_dropout
                    keep[:, 0] = True                          # never drop everything
                    mb = keep.to(dev)
                memory = self.encoder.encode(tb, mb)
                ctx = self.encoder.pool(memory, mb)
                eps_z = torch.randn(k, bs, self.d, generator=gen).to(dev)
                if self.base_head is not None:
                    # The base reads a DETACHED context. Otherwise its NLL, which is
                    # a much stronger signal than the CFM term, trains the shared
                    # encoder 20-176x harder than the flow does, and the token
                    # memory ends up shaped as a good linear readout for a diagonal
                    # Gaussian instead of as information for the velocity field.
                    base_nll, draw, _ = self.base_head.nll(zb, ctx.detach())
                    # The base is trained by its NLL only. z0 is DETACHED from the
                    # CFM term: otherwise ||v - (z1 - z0)||^2 can be reduced by
                    # dragging z0 towards z1, i.e. the loss would pay the base to
                    # absorb the flow's job, collapsing the regression target to
                    # zero and leaving the velocity field with nothing to learn.
                    z0 = torch.cat([draw(eps_z[j]) for j in range(k)], dim=0).detach()
                else:
                    z0 = eps_z.reshape(k * bs, self.d)
                    base_nll = torch.zeros((), device=dev)
                # k independent (t, z0) pairs share ONE encoder pass. The memory is
                # tiled, not re-encoded, so gradients from all k pairs accumulate
                # into the same encoder graph.
                zk = zb.repeat(k, 1)
                if t_strat and k > 1:
                    # one t per stratum [j/k, (j+1)/k) per dataset: the k pairs then
                    # cover the whole path instead of clumping at random
                    t = (torch.arange(k, dtype=torch.float32).repeat_interleave(bs)
                         + torch.rand(k * bs, generator=gen)) / k
                else:
                    t = torch.rand(k * bs, generator=gen)
                t = t.to(dev)
                zt = (1 - t)[:, None] * z0 + t[:, None] * zk
                target = zk - z0
                if self.conditioning == "xattn":
                    mem_k = memory if k == 1 else memory.repeat(k, 1, 1)
                    cond = self.velocity.encode_memory(mem_k)
                    vmask = None if mb is None else (mb if k == 1 else mb.repeat(k, 1))
                else:
                    cond = ctx if k == 1 else ctx.repeat(k, 1)
                    vmask = None
                pred = self.velocity(zt, t, cond, vmask)
                cfm = ((pred - target) ** 2).mean()
                loss = cfm + base_weight * base_nll
                opt.zero_grad(); loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                opt.step()
                if ema is not None:
                    ema.update()
                running += cfm.item()
                step_now += 1
            # the LAST epoch is always measured: with a decaying schedule the final
            # stretch is where the model settles, and a run whose last check sits
            # 1500 steps short of the end understates exactly what the decay buys
            if monitor is not None and (ep == epochs - 1 or
                                        step_now // monitor_every >
                                        (step_now - n_batches) // monitor_every):
                with torch.no_grad():
                    # the EMA weights are what will be shipped, so they are what
                    # gets measured
                    with (ema.averaged() if ema is not None else contextlib.nullcontext()):
                        val = float(monitor(self))
                self.history.append(dict(step=step_now, cfm=running / n_batches, metric=val))
                if verbose:
                    print(f"  step {step_now:6d}  cfm {running / n_batches:.4f}"
                          f"  dist-to-reference {val:.4f}  ({time.time() - t0:.0f}s)")
            elif verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  step {step_now:6d}  "
                      f"cfm {running / n_batches:.4f}  ({time.time() - t0:.1f}s)")
        if ema is not None:
            ema.store_in_module()
        return self

    # --- inference -------------------------------------------------------
    @torch.no_grad()
    def sample_batch(self, tokens, n: int = 1000, n_steps: int = 20,
                     seed: int = 0, chunk: int = 16, solver: str = "midpoint",
                     mask: torch.Tensor = None) -> torch.Tensor:
        """Posterior samples for a *batch* of observations. tokens [B, T, F] -> [B, n, d].

        All B datasets are encoded together and their ODEs solved jointly, which is
        what makes calibration studies (hundreds of datasets) tractable: one solve
        instead of B python-loop solves. `chunk` bounds peak attention memory.
        """
        if isinstance(tokens, (list, tuple)):                 # variable-length input
            tokens, mask = pack_tokens([torch.as_tensor(t) for t in tokens])
        gen = torch.Generator().manual_seed(seed)
        dev = next(self.parameters()).device
        outs = []
        for start in range(0, tokens.shape[0], chunk):
            tb = tokens[start:start + chunk].to(dev)
            mb = None if mask is None else mask[start:start + chunk].to(dev)
            B = tb.shape[0]
            memory = self.encoder.encode(tb, mb)              # [B, T, dim]
            ctx = self.encoder.pool(memory, mb)               # [B, dim]
            eps = torch.randn(B, n, self.d, generator=gen).to(dev)
            if self.base_head is not None:
                mu, s = self.base_head(ctx)                   # [B, d]
                z = mu[:, None] + s[:, None] * eps
            else:
                z = eps
            # conditioning computed once per dataset, reused across the whole solve
            grouped = self.velocity.forward_grouped
            cond = self.velocity.encode_memory(memory) if self.conditioning == "xattn" else ctx
            vmask = mb if self.conditioning == "xattn" else None
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=dev)
                if solver == "euler":                         # 1 eval / step
                    z = z + dt * grouped(z, t, cond, vmask)
                elif solver == "midpoint":                    # 2 evals / step
                    k1 = grouped(z, t, cond, vmask)
                    z = z + dt * grouped(z + 0.5 * dt * k1, t + 0.5 * dt, cond, vmask)
                else:                                         # rk4: 4 evals / step
                    k1 = grouped(z, t, cond, vmask)
                    k2 = grouped(z + 0.5 * dt * k1, t + 0.5 * dt, cond, vmask)
                    k3 = grouped(z + 0.5 * dt * k2, t + 0.5 * dt, cond, vmask)
                    k4 = grouped(z + dt * k3, t + dt, cond, vmask)
                    z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            outs.append(self.prior.denormalize(z.cpu()))
        return torch.cat(outs, dim=0)

    def sample(self, tokens: torch.Tensor, n: int = 2000, n_steps: int = 20,
               seed: int = 0, solver: str = "midpoint") -> torch.Tensor:
        """Posterior samples for one observation. tokens: [T, F] or [1, T, F] -> [n, d]."""
        if tokens.dim() == 2:
            tokens = tokens[None]
        return self.sample_batch(tokens, n=n, n_steps=n_steps, seed=seed, solver=solver)[0]
