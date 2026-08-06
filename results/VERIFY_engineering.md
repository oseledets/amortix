# VERIFY — engineering audit (everything except the CFM mathematics)

Scope as assigned: (1) training-loop mechanics, (2) randomness hygiene, (3) numerical
precision, (4) token feature scaling, (5) train/inference consistency, (6) the observers.
The CFM objective itself is another reviewer's.

**No source file was modified.** Every probe is a replica of the shipped code path or a
hot-swap of a module with the same interface; the only file written is this one.

**Instruments.**
* `amortix/problems/linear_gaussian.py` — exact posterior in closed form
  (`exact_posterior`, verified to return the full 800 requested draws on every call).
  Used for every width/correlation number. Its unconstrained posterior sd is
  `[0.376, 0.430, 0.264, 0.278]` against a prior sd of 1.732.
* `amortix/problems/ou.py` + the exact OU transition likelihood evaluated on the
  **union of the observer's time indices** (73 of 501 path points), on a 181×181
  (θ, σ) grid — a gold posterior conditioned on exactly the data the network sees.
* Configuration throughout: `base="data"`, `conditioning="xattn"`, `dim_model=64`,
  `n_layer=3`, `depth=3`, Adam `lr=3e-4`, budget `n_train=12000 / epochs=30`
  unless stated otherwise.

---

## Summary, ranked by impact

| # | finding | status | causes a too-wide / under-correlated posterior? |
|---|---|---|---|
| **E0** | `AttentionPool` — the whole pooling module, 12 738 parameters — receives **exactly zero gradient** and never changes. `ctx.detach()` is the only edge into it in the default `conditioning="xattn"` config. The base head, which sets the posterior's spread, therefore reads a **randomly-initialised** projection | CONFIRMED | **yes, directly — this is E1's root cause** |
| **E1** | The data-dependent base head is **not converged**: it is 89 % worse than the best *linear* probe on its own frozen input, and its σ̂ is **2.1–3.3× the true posterior sd** — the exact opposite of what its docstring claims | CONFIRMED | **yes, directly** |
| **E2** | Token features are unnormalised: at init the informative channels carry **0.05–0.7 %** of the first-layer variance while a 1-bit channel indicator carries **43–91 %**; on SEIR the *measured value* carries **0.22 %** | CONFIRMED | plausibly (weak encoder ⇒ E1 gets worse) |
| **E3** | Training pairs are simulated **once** before the epoch loop and reused for all epochs | CONFIRMED; gap measured **+5.1 %** | marginally, at small `n_train` |
| **E4** | The encoder gets **exactly zero** gradient at the first optimiser step, and ~8× less than the velocity for the first ~10 epochs (adaLN-Zero × detached base context) | CONFIRMED | indirectly |
| **E5** | No gradient clipping, no LR schedule, no weight decay, one LR for three components with very different conditioning; the base head's Adam update/weight ratio is 1.00 at ep 0 and still 0.15 at ep 29 | CONFIRMED | indirectly — this is E1's mechanism |
| **E6** | fp32 `erf` saturates for \|z\| ≳ 5.4, so `denormalize` returns the prior-box edge **exactly**; `normalize` clamps training targets at \|z\| ≤ 4.751 | CONFIRMED | no (narrows, if anything) |
| **E7** | The `cid` / channel feature is identically zero in 6 of 9 problems — a permanently dead input dimension | CONFIRMED | no |
| **E8** | OU's fast and slow channels share 3 time indices, so 2 of 74 tokens are redundant | CONFIRMED | no |
| **E9** | Latent trap: in `fit`, `torch.manual_seed(seed)` and `torch.Generator().manual_seed(seed)` are the **same stream** | CONFIRMED (inert today) | no |

**Clean bills — verified numerically, not assumed.** Chunk-size invariance of
`sample_batch`; independence of base draws across chunks and datasets; `rand`/`randn`
seed reuse in SBC; `velocity.forward` vs `forward_grouped` bit-identity; no
cross-sample leakage in the grouped path; padding-mask correctness; RoPE table growth;
fp32 vs fp64 attention/pooling/ODE; `eval()` being a genuine no-op. §7.

---

## 1. Training-loop mechanics (`FlowPosterior.fit`)

### E1 — the base head is not converged, and it seeds 2–3.3× too much spread. **CONFIRMED**

`flow.py` docstring: *"The NLL keeps s_hat ~ the true posterior std (an ODE cannot
create spread from a point, so the base must seed it)."* Measured, that is false by a
factor of 2–3.3.

`BaseHead` is a **single `nn.Linear(ctx_dim, 2*dim)`** reading a **detached** context.
After the shipped 12000/30 run on `linear_gaussian`, on 8000 freshly simulated pairs:

```
                                          RMS(z1 - mu_hat)   mean s_hat
shipped linear head                            0.6494        [0.585 0.700 0.633 0.543]
best linear probe on the SAME frozen ctx       0.3425        --
2x128 SiLU MLP on the SAME frozen ctx          0.3143        [0.302 0.344 0.257 0.237]
```

The shipped head is **1.90× worse than the closed-form optimum of its own model class
on its own input**. It is an optimisation failure, not a capacity limit: a ridge
solution on the identical frozen context halves its residual.

Consequence, measured against the exact posterior on 60 held-out datasets (z-space sd):

```
true conditional posterior sd   [0.266 0.267 0.187 0.156]
shipped linear base head        [0.557 0.687 0.621 0.516]   =  2.10x 2.58x 3.32x 3.31x
MLP head on the same frozen ctx [0.308 0.337 0.254 0.186]   =  1.16x 1.27x 1.36x 1.19x
```

A Gaussian NLL fitted with a biased mean inflates the scale to cover the bias:
`s² → Var + bias²`. Here `0.649² = 0.42`, of which the true conditional variance is
only `0.266² ≈ 0.07` — **83 % of the base's variance is the head's own mean error.**

So the data-dependent base does not do the job its docstring assigns it. It hands the
flow a source that is 2–3.3× too wide, i.e. exactly the *"large, stiff contraction that
a finite network underfits"* the feature was introduced to avoid. With
`base="standard"` the source is 3.7–5.5× too wide; the flow's job is barely easier
with `base="data"` than without it, which is visible end-to-end:

```
linear_gaussian, 12000/30, 40 datasets x 800 draws, vs the EXACT posterior
base        std ratio   corr err   |corr| flow   |corr| exact   base spread / true
data          1.035       0.227        0.243         0.425          2.4-3.7x
standard      1.065       0.239        0.247         0.425          3.7-5.5x
```

The flow *does* absorb most of the width error (final std ratio 1.03–1.07), but it
delivers only **57 % of the true parameter correlation**. That is the dominant
remaining defect on the reference instrument, and it is consistent with the repo's own
token-matched MCMC comparison (`results/VS_MCMC_ou.md`: corr(θ,σ) amortised +0.129 vs
MCMC +0.289).

**Patch.** In `flow.py`:

```python
class BaseHead(nn.Module):
    def __init__(self, ctx_dim, dim, hidden=128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)   # keep the N(0, I) start
        nn.init.zeros_(self.net[-1].bias)
```

and in `fit`, give the head its own parameter group with a larger LR, because it is the
one component whose loss is a proper scoring rule and whose input is frozen:

```python
head = [p for n, p in self.named_parameters() if n.startswith("base_head")]
rest = [p for n, p in self.named_parameters() if not n.startswith("base_head")]
opt = torch.optim.Adam([{"params": rest, "lr": lr},
                        {"params": head, "lr": 10 * lr}])
```

The same change applies to `FullBaseHead`.

### E3 — the training set is simulated once and memorised. **CONFIRMED, effect small**

`fit` calls `self.problem.simulate(n_train, generator=gen)` **before** the epoch loop;
the same `n_train` pairs are reshuffled and replayed for every epoch. There is no
regeneration and (by default) no token dropout, so the only stochasticity per epoch is
the base draw `eps` and the flow time `t`.

Measured CFM loss on the training set vs on 3000–4000 freshly simulated pairs, same
`(eps, t)` seed:

```
budget            train CFM   fresh CFM     gap
12000 / 30 data     0.2384      0.2507     +5.1 %
12000 / 30 standard 0.3746      0.3913     +4.5 %
```

At the canonical budget the memorisation gap is ~5 %, so this is **not** the cause of a
wide posterior. It becomes a real risk at the `examples/recover.py` default
(`n_train=8000, epochs=25`) and at any longer schedule; see §1 table in the appendix
for the `n_train` sweep.

**Patch.** Cheap and strictly better for CPU-cheap simulators — regenerate periodically:

```python
def fit(..., resim_every: int = 0):
    ...
    for ep in range(epochs):
        if resim_every and ep and ep % resim_every == 0:
            m, tokens = self.problem.simulate(n_train, generator=gen)
            z1 = self.prior.normalize(m)
```

Also report the fresh-pair loss alongside the training loss in the `verbose` line —
the current print reports only the training CFM, so memorisation is invisible.

### E4 — the encoder is frozen at step 0 and starved for ~10 epochs. **CONFIRMED**

Two mechanisms compound:

1. `CrossBlock.ada` is zero-initialised (adaLN-Zero), so at step 0 all nine
   modulation outputs are 0 and every residual branch — including the
   cross-attention that is the encoder's *only* consumer — is multiplied by 0.
2. The base head reads `ctx.detach()`, so the NLL never reaches the encoder either.

Result: with `base="data"` the encoder's gradient at the very first batch is
**exactly 0.0**, while the velocity's is 12.1. Per-component gradient norms
(first batch of each logged epoch, `linear_gaussian`, `base="data"`):

```
epoch   |g| encoder   |g| velocity   |g_nll| encoder   |g_nll| base_head
    0     0.000e+00     1.209e+01       0.000e+00        3.757e-01
    5     1.092e-01     8.966e-01       0.000e+00        2.633e-01
   10     1.098e+00     1.595e+00       0.000e+00        3.051e-01
   15     1.727e+00     2.226e+00       0.000e+00        2.343e-01
   29     1.494e+00     1.419e+00       0.000e+00        1.708e-01
```

`|g_nll| encoder ≡ 0` at every epoch is the detach, working as documented. The
`|g_cfm| encoder = 0` at epoch 0 is the adaLN-Zero gate; the encoder is literally not
being trained until `ada` moves off zero, and at epoch 5 it is still receiving 8× less
gradient than the velocity. Nothing is *permanently* frozen — but roughly the first
sixth of the budget is spent training a velocity field that reads an untrained context.

**Patch.** Zero-initialise only the *gate* chunks, not the shift/scale chunks, or
initialise the gates to a small non-zero value (e.g. `1e-2`) so the encoder receives
signal from step 0:

```python
nn.init.zeros_(self.ada.weight)
nn.init.zeros_(self.ada.bias)
with torch.no_grad():                       # gates g0, g1, g2 are chunks 2, 5, 8
    for c in (2, 5, 8):
        self.ada.bias[c * dim:(c + 1) * dim] = 1e-2
```

### E5 — no clipping, no schedule, one LR for three very differently-conditioned components. **CONFIRMED**

`opt = torch.optim.Adam(self.parameters(), lr=lr)`; `loss.backward(); opt.step()`.
There is no `clip_grad_norm_`, no LR decay, no weight decay. Adam's
update-to-weight ratio per component (‖Δθ‖/‖θ‖ over each logged interval):

```
epoch    encoder   velocity   base_head
    0    5.19e-02   8.11e-02   1.00e+00
    5    1.26e-01   1.40e-01   8.69e-01
   15    8.25e-02   8.89e-02   4.63e-01
   29    1.27e-02   2.30e-02   1.47e-01
```

The base head starts at exactly zero, so its first epoch moves it by 100 % of its own
norm, and at the end of training it is still moving 15 % per 4 epochs — it never
settles. That is E1's mechanism: a constant LR with no decay on a zero-initialised
linear head whose input is a moving target for 30 epochs.

**Patch.** Add `torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)` before
`opt.step()` and a cosine schedule
(`torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*n_batches)`, stepped per
batch). Both are one line and neither changes the method.

### Minor, same section

* `n_batches = max(1, n_train // batch)` silently drops the last partial batch. With
  `n_train=12000, batch=256` that is 224 pairs per epoch — harmless, because `perm` is
  redrawn each epoch, but worth a comment.
* `running += cfm.item()` is the only quantity printed; `base_nll` is accumulated but
  never reported, so an NLL that diverges is invisible.
* `base_weight` multiplies a per-dimension mean NLL against a per-dimension mean squared
  error. The two are on comparable scales here (CFM 0.24 vs NLL −0.06 at the end), so
  no term is irrelevant — but the NLL's gradient reaches only `base_head`, so
  `base_weight` is effectively just a base-head LR multiplier. Say so in the docstring.

---

## 2. Randomness hygiene — essentially clean

### Verified negatives

* **`sample_batch` is exactly chunk-invariant.** `gen` is created once outside the
  chunk loop and `torch.randn(B, n, d, generator=gen)` is row-major, so dataset *i*
  always consumes stream positions `[i·n·d, (i+1)·n·d)` regardless of `chunk`.
  Measured over `chunk ∈ {1, 4, 8, 16, 40, 64}` on 40 datasets × 500 draws: identical
  posterior means and stds to 4 decimals, `max |diff|` = 1.9e-6 (fp32 matmul
  reassociation, not RNG). **No chunk-boundary noise repetition**: 0 duplicate rows,
  max off-diagonal correlation between per-dataset `eps` blocks 0.072 over 2000 dims
  (MC noise floor 1/√2000 = 0.022 → this is a 40×40 max-of-1560, consistent with noise).
* **Draws are exchangeable within a dataset.** First vs second half of 4000 draws:
  KS p = 0.16, 0.21, 0.48, 0.51 per dimension.
* **The SBC shared seed is harmless.** `run_sbc` seeds the data generator and
  `sample_batch` with the same integer. On CPU, `torch.rand` and `torch.randn` from the
  same seed are *not* related: corr(u_k, e_k) = −0.0008, corr(u_k, |e_k|) = 0.0002, and
  the Box–Muller pairing hypothesis is rejected (corr 0.0034, max abs difference 5.80).
  Cross-dataset draw correlation in SBC: 0.043 (MC floor 0.035).

### E9 — a latent seed-reuse trap. **CONFIRMED, currently inert**

`fit` does both `gen = torch.Generator().manual_seed(seed)` **and**
`torch.manual_seed(seed)`. Those two streams are *identical*:

```
default RNG after torch.manual_seed(0):  [ 1.5410 -0.2934 -2.1788  0.5684 -1.0845]
Generator().manual_seed(0):              [ 1.5410 -0.2934 -2.1788  0.5684 -1.0845]
IDENTICAL: True
```

Today nothing in the training loop draws from the default generator (every `randn`,
`rand` and `randperm` passes `generator=gen`), so the duplication is inert. But the
moment anyone adds dropout, a random augmentation, or a `torch.randn` without the
`generator=` kwarg, its draws will be **bit-identical** to the base noise `eps` or to
the simulator's Brownian increments. That is precisely the class of defect this
codebase has already been bitten by.

**Patch.** Either drop `torch.manual_seed(seed)` (module init happens in `__init__`,
before `fit`, so it buys nothing) or offset it: `torch.manual_seed(seed + 10_000_019)`.

---

## 3. Numerical precision — fp32 is adequate, with one saturation edge

### The probit map

`BoxUniform.normalize` clamps `u` to `[1e-6, 1−1e-6]`, capping training targets at
`|z1| ≤ 4.751`. fp32 vs fp64 `erfinv` over 800 000 prior draws:

```
median |z32 - z64|   3.6e-08
|z| > 0.67 (top 50 %)   1.9e-07
|z| > 2.57 (top  1 %)   4.5e-06
|z| > 3.28 (top 0.1 %)  3.1e-05
|z| > 3.87 (top 0.01%)  1.9e-04
worst case (|z| = 4.75) 3.9e-03
```

Round-trip `denormalize(normalize(m))` error: `5.2e-06` absolute = `8.7e-07` of the
prior range. Irrelevant against a posterior sd of 2–25 % of the range.

### E6 — fp32 `erf` saturation at the box edges. **CONFIRMED**

`denormalize` uses `0.5*(1 + erf(z/√2))`. In fp32 that returns exactly 1.0 for
`z ≳ 5.4`:

```
z = 5.0   1 - u  fp32 2.98e-07   fp64 2.87e-07
z = 5.5   1 - u  fp32 0.00e+00   fp64 1.90e-08   <- m == prior.high exactly
z = 6.0   1 - u  fp32 0.00e+00   fp64 9.87e-10
```

So any ODE output past ±5.4 collapses onto the box edge, producing a point mass there.
Frequency measured on the reference model (see §appendix, probe C) — it is rare, and it
truncates rather than widens, so it cannot explain a wide posterior; it *can* distort
SBC ranks at the extremes (ties at rank 0 / rank L). Note the asymmetry: training
targets are capped at 4.751 but nothing caps the sampler's output.

**Patch.** Clamp the ODE output before denormalising, at the same place the targets are
clamped:

```python
Z_MAX = math.sqrt(2) * torch.erfinv(torch.tensor(1.0 - 2 * _EPS)).item()   # 4.751
outs.append(self.prior.denormalize(z.clamp(-Z_MAX, Z_MAX)))
```

### Everything else in fp32

* Attention / pooling, fp32 vs fp64 on the same weights: encoder memory relative error
  **3.2e-07**, pooled context **2.3e-07**. `scaled_dot_product_attention` upcasts the
  softmax internally; no precision problem.
* Gaussian NLL: `BaseHead` clamps `log_s` to `[-4, 2]`, so `s ∈ [0.018, 7.39]` and
  `s²` never underflows.
* `FullBaseHead` is the one place with a conditioning hazard: the diagonal is clamped
  (`exp(log_d)`, floor 0.0183) but the **off-diagonals are unclamped**, so `L` can be
  made arbitrarily ill-conditioned while `log|det L|` — which only sees the diagonal —
  stays finite and the NLL never penalises it. With random weights of scale 0.5,
  `cond(L)` reaches 2.3e8. It is not the default base, but it is the one the HANDOFF
  nominates as the fix for correlated posteriors, so it should be hardened before use
  (e.g. scale the off-diagonals by `tanh` × the diagonal, or add a small ridge).

---

## 4. Token feature scaling — the informative channels are invisible at init

### E2. **CONFIRMED**

`SetTransformer.embed = nn.Linear(n_features, dim)` with PyTorch's default
`U(±1/√n_features)` init and **no input normalisation**. Per-feature share of the
first-layer pre-activation variance, `std(x_f · W[:, f])²` normalised to sum to 1,
measured on 256 real datasets per problem:

```
problem            t/H      x       dx      dx^2   log10 dt   cid
ou                3.00%  12.33%   3.95%   3.13%    77.59%   0.00%
gbm               2.02%  43.87%   0.52%   0.70%    52.89%   0.00%
cir               3.92%  32.77%   0.41%   0.05%    62.86%   0.00%
double_well       2.12%  13.82%   1.36%   0.59%    82.11%   0.00%
stoch_lv          2.46%  43.94%   0.32%   0.15%    43.84%   9.29%
sindy_sde         3.83%   4.55%   0.93%   0.15%    90.55%   0.00%

problem                    f0            f1              f2
linear_gaussian       y  98.58%    index 1.42%          --
seir                  t  33.15%    value 0.22%     channel 66.63%
fhn                   t  19.45%    v    80.55%     channel  0.00%
```

Read that table against what each feature is supposed to do:

* `dx²` is the **quadratic-variation cue — the only direct evidence about the diffusion
  coefficient** — and it carries **0.05 % (CIR), 0.15 % (sindy_sde, stoch_lv), 0.59 %
  (double_well), 0.70 % (GBM)** of the first-layer signal.
* `log10 dt` takes exactly **two distinct values** (one per channel): it is a one-bit
  channel indicator. It carries **43–91 %** of the signal — 600× more than `dx²` on CIR.
* On SEIR the **observed value itself** carries 0.22 %, against 66.6 % for the channel
  indicator and 33.2 % for time. The network's first layer is almost entirely a
  representation of *which series and when*, not *what was measured*.
* `cid` is identically zero in ou / gbm / cir / double_well / sindy_sde, and the FHN
  channel feature is identically zero. Six of nine problems feed the encoder a
  constant. (E7.)

This is a pure initialisation-scale problem — the weights can in principle rescale —
but it means the encoder starts training with the diffusion evidence three orders of
magnitude below the channel tag, and the only gradient that could fix it is the CFM
loss, which (E4) is itself gated off for the first epochs. A too-wide posterior for
`sigma`-type parameters is exactly the symptom this predicts.

**Patch.** Standardise the token features once, from the training simulation, and store
the statistics on the model so inference uses the same map:

```python
# in fit(), right after m, tokens = self.problem.simulate(...)
f = tokens.reshape(-1, tokens.shape[-1])
self.register_buffer("tok_mu", f.mean(0))
self.register_buffer("tok_sd", f.std(0).clamp_min(1e-6))
tokens = (tokens - self.tok_mu) / self.tok_sd
# and in encode paths / sample_batch:
tokens = (tokens - self.tok_mu) / self.tok_sd
```

A constant feature standardises to exactly 0 (the `clamp_min` keeps it finite), which
also disposes of E7 for free.

---

## 5. Train/inference consistency — clean

Every difference between the training path and `sample_batch` was checked numerically:

| aspect | training path | inference path | verdict |
|---|---|---|---|
| velocity call | `velocity(zt, t, cond, mb)` | `velocity.forward_grouped(z, t, cond, mb)` | **bit-identical** for G=1: `max\|Δ\| = 0.0` |
| cross-sample independence | n/a | G samples folded into the query axis | **no leakage**: perturbing sample 1 changes sample 0 by exactly 0.0 |
| token construction | `problem.simulate` | `problem.observe` | same `tokens_from_traj` / `tokens_from_solution` |
| conditioning | `encode_memory(memory)` per batch | `encode_memory(memory)` per chunk | same function, same inputs |
| base | `base_head.nll(zb, ctx.detach())` → `draw(eps)` | `base_head(ctx)` → `mu + s·eps` | same `(mu, s)`; the detach only affects gradients |
| mask | `mb` (None unless `token_dropout>0`) | `mb` from `pack_tokens` or `None` | padded+masked call is **bit-identical** to the unpadded call (`max\|Δ\| = 0.0` with 40 junk tokens appended) |
| `no_grad` | absent | `@torch.no_grad()` on `sample_batch` | no forward-path difference |
| `train()` / `eval()` | never set | never set | **genuine no-op**: the module tree contains no `Dropout`, no `BatchNorm`, no `LayerNorm` — only `Linear`, `RMSNorm`, `SiLU`, attention. `train()` vs `eval()` output difference: exactly 0.0 |

One asymmetry worth documenting rather than fixing: with `token_dropout > 0` the model
trains on a random subset of tokens but is evaluated on all of them — the intended
direction, but it means the *effective* number of observations differs between train
and test, and the RoPE positions the surviving tokens occupy do not shift, so dropout
teaches robustness to missing values, not to a shorter sequence.

`SetTransformer._rope` grows the table in place on a longer input: verified with a
740-token input against a 114-entry table — no crash, table grows to 740, output
finite. Note the growth permanently mutates the module, so a model that has once seen a
long input keeps the larger buffer; since the buffers are `persistent=False` this does
not affect checkpoints.

---

## 6. The observers

Checked for every problem in `GALLERY`: token tensor shape equals
`(B, observer.n_tokens, N_FEATURES)`; all indices in range; no non-finite values; all
tokens within a set distinct.

```
problem        n_tokens  channels (every/count)  max index / n_steps  clamped  t-feature span
ou                74     1/50, 20/24              50, 480 / 500          0     [0, 0.098] / [0, 0.920]
gbm              100     1/60, 10/40              60, 400 / 500          0     [0, 0.118] / [0, 0.780]
cir               74     1/50, 20/24              50, 480 / 500          0     [0, 0.098] / [0, 0.920]
double_well      116     1/80, 25/36              80, 900 / 1000         0     [0, 0.079] / [0, 0.875]
stoch_lv         180     1/50, 15/40 (x2 dims)    50, 600 / 600          0     [0, 0.082] / [0, 0.975]
sindy_sde        125     1/80, 20/45              80, 900 / 1000         0     [0, 0.079] / [0, 0.880]
```

* **Index range / off-by-one: clean.** `idx = arange(count+1) * every` then
  `clamp(max=n_steps)`; `count+1` points give exactly `count` increments, and no
  problem currently over-runs (`clamped = 0` everywhere). The clamp is a silent
  failure mode if one ever does — an over-running channel emits `dx = 0` tokens that
  look like observations — so it should assert rather than clamp:
  `assert c.count * c.every <= self.n_steps`.
* **E8 — channel overlap.** The two channels always start at index 0, so their index
  sets intersect. No `(start, end)` increment pair is duplicated outright (checked for
  all problems) — this is double *representation* of shared time points, not double
  *counting* of increments — but it means the advertised token count overstates the
  number of independent transitions:

  ```
  problem       tokens/dim   distinct time points   independent transitions   shared indices           redundancy
  ou                74            73 of 501                  72                {0,20,40}                   2  (3 %)
  gbm              100            95 of 501                  94                {0,10,...,60}               6  (6 %)
  cir               74            73 of 501                  72                {0,20,40}                   2  (3 %)
  double_well      116           114 of 1001                113                {0,25,50,75}                3  (3 %)
  stoch_lv          90            88 of 601                  87                {0,15,30,45}                3  (3 %)
  sindy_sde        125           122 of 1001                121                {0,20,40,60,80}             4  (3 %)
  ```
* **`n_paths` concatenation is correct**: `_tokens_for` calls `simulate_paths`
  `n_paths` times off the same generator (independent noise, independent `x0`) and
  concatenates on the token axis. Two caveats: (i) RoPE gives each replicate a
  different position block, so replicate paths are *not* exchangeable to the encoder;
  (ii) `observe()` returns only the **last** path as `traj`, so with `n_paths > 1`
  every `sota()` baseline silently sees 1/`n_paths` of the data. Default is 1.
* **The time feature is meaningful but nearly degenerate on the fast channel**: it
  spans 8–12 % of the horizon there, so 50–80 of the tokens differ in `t/H` by
  ~0.002 each. Combined with E2 (`t/H` = 2–4 % of first-layer variance) the fast
  channel's ordering information is carried almost entirely by RoPE, not by the
  feature.
* **`log10 dt` and `cid` are constants per channel**, i.e. the 6-feature token really
  encodes 4 varying numbers plus a 1-bit channel tag (and, in 6 of 9 problems, one
  identically-zero column — E7).
* **Advertised information is present**: for every SDE problem the `dx` and `dx²`
  columns reproduce the raw path increments exactly, and the `x` column the pre-step
  state (verified by reconstructing the observed subsequence from the tokens).

---

## 7. Context — the "too wide" reading is partly an artefact of the comparison

Already documented as `results/CRITIC_problems.md` §1 (`sota()` is fed 5–10× more data
than the network), but the magnitude in *posterior* terms had not been measured. Using
the exact OU transition likelihood on a flat box prior, with the true (θ, σ) = (1.5, 0.8),
averaged over 12 simulated paths:

```
                            full 500-step path   the 74 token increments   ratio
posterior sd theta                18.59 %                19.85 %           1.07x
posterior sd sigma                 2.00 %                 5.55 %           2.77x   (% of prior range)
theta-sigma correlation           +0.149                 +0.349
```

So a σ posterior 2.8× wider than the classical MLE's sampling sd is **the correct
answer** for the data the network is given, not a defect. Conversely the correlation
the network must reproduce is *larger* on the token view (+0.35) than on the full path
(+0.15) — which is why under-correlation, not width, is the sharper symptom here.

Any future "too wide" claim must be made against a token-matched reference:
`amortix/mcmc.py` already builds one (`observed_indices`, "73 of 501 path points"), and
`examples/vs_mcmc.py` uses it. The `sota()` comparisons in `examples/gallery.py` and
`examples/scoreboard.py` do not.

---

## 8. Appendix — probes and raw output

### A. Observer audit (all 9 gallery problems)

Token tensors have the advertised shape, no clamped indices, no non-finite values, and
every token in a set is distinct. Reconstructing the `x`, `dx`, `dx²`, `t/H` and `cid`
columns directly from the raw trajectory reproduces the emitted tokens **exactly**:

```
          ou: tokens cover  74/74   max reconstruction error 0.00e+00
         gbm: tokens cover 100/100  max reconstruction error 0.00e+00
         cir: tokens cover  74/74   max reconstruction error 0.00e+00
 double_well: tokens cover 116/116  max reconstruction error 0.00e+00
    stoch_lv: tokens cover 180/180  max reconstruction error 0.00e+00
   sindy_sde: tokens cover 125/125  max reconstruction error 0.00e+00
```

Raw feature statistics (256 datasets each) that produce the §4 table:

```
ou       t/H  mean +0.182 sd 0.250 | x  sd 0.633 | dx sd 0.299 | dx^2 mean 0.0895 sd 0.300 | log10dt sd 0.609 | cid sd 0
gbm      t/H  mean +0.191 sd 0.220 | x  sd 1.164 (max 39.4) | dx sd 0.217 | dx^2 sd 1.372 (max 148) | log10dt sd 0.490 | cid sd 0
cir      t/H  mean +0.182 sd 0.250 | x  sd 0.421 | dx sd 0.104 | dx^2 mean 0.0109 sd 0.042 | log10dt sd 0.609 | cid sd 0
sindy    t/H  mean +0.184 sd 0.248 | x  sd 0.372 | dx sd 0.154 | dx^2 sd 0.063 | log10dt sd 0.625 | cid sd 0
seir     t    sd 0.293 | value mean 0.0154 sd 0.031 | channel sd 0.500
fhn      t    sd 0.301 | v sd 0.832 | channel sd 0  (identically zero)
```

Note GBM's heavy tail: `x` reaches 39.4 and `dx²` reaches 148 on a prior-drawn path,
into an un-normalised `nn.Linear` — another argument for E2's patch.

### B. Randomness probes

```
=== chunk-size invariance of sample_batch (40 datasets x 500 draws) ===
 chunk=  1  mean [-0.0809  0.0020 -0.0373 -0.0945]  std [1.7517 1.7661 1.6271 1.6178]
 chunk=  4  mean [-0.0809  0.0020 -0.0373 -0.0945]  std [1.7517 1.7661 1.6271 1.6178]
 chunk=  8  ... identical ...
 chunk= 64  mean [-0.0809  0.0020 -0.0373 -0.0945]  std [1.7517 1.7661 1.6271 1.6178]
 max|diff vs chunk=16| <= 1.9073e-06  (fp32 matmul reassociation)

=== per-dataset eps blocks ===
 chunk=4  max |off-diag corr| 0.0721 over 2000 dims; exact duplicate rows: 0
 chunk=16 identical

=== exchangeability within one dataset (4000 draws, first half vs second) ===
 KS p-values per dim: 0.161  0.212  0.484  0.509

=== torch.manual_seed(s) vs Generator().manual_seed(s) ===
 IDENTICAL streams: True
=== but torch.rand and torch.randn from the same seed are unrelated ===
 corr(u_k, e_k) = -0.0008   corr(u_k,|e_k|) = 0.0002   Box-Muller pairing corr = 0.0034
```

### C. Precision probes

```
=== forward vs forward_grouped (training path vs inference path) ===
 max |forward - forward_grouped| = 0.000e+00
 cross-sample leakage (perturb sample 1 -> change sample 0) = 0.000e+00

=== masking ===
 padded (+40 junk tokens) with mask vs unpadded: max |diff| = 0.0
 variable-length list input: works, dataset 0 matches the plain call exactly
 RoPE growth: T=740 against a 114-entry table -> table grows to 740, output finite

=== fp32 vs fp64 on identical weights ===
 encoder memory   relative error 3.2e-07
 pooled context   relative error 2.3e-07

=== train() vs eval() ===
 max |diff| = 0.0 ; module tree contains no Dropout / BatchNorm / LayerNorm
```

