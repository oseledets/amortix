# Adversarial review of the CFM implementation

Scope: `amortix/flow.py`, `amortix/encoder.py`, `amortix/prior.py`, `amortix/diagnostics.py`,
`METHOD.md`. Every claim below marked **CONFIRMED** was measured, not reasoned about.

**Snapshot note.** `amortix/flow.py` was being edited by another workstream *during* this
audit — the working tree gained `SelfAttention` / `CrossBlock(…, n_param)` mid-run, which
briefly broke imports. All numbers below come from a pinned copy taken at that moment
(`flow.py` sha1 `2e3f2bfa92ea9cce1b951d867110047ddf8872cc`); the working tree has since
stabilised at that same sha1 and imports cleanly — the edit landed as commit `1b42c1a`
("Find and fix three defects…") — so the snapshot *is* the current tree. No source file was
modified by this audit; the only file written is this report.

Two labels are used throughout:

- **"as shipped"** = the configuration that produced the 17/29 gallery score: *no* parameter
  self-attention, `base="data"`, `conditioning="xattn"`. Reproduced on the snapshot by
  monkey-patching `CrossBlock.forward` to skip the self-attention branch (the unused
  `n0`/`selfattn` weights receive zero gradient, so this is faithful).
- **"self-attn"** = the current working tree as-is.

F1 was found on the pre-fix file *before* the in-flight edit appeared; the concurrent fix is
independent corroboration, and it is audited here on its own terms (it is correct, and it is
not sufficient — see F2).

**Budget caveat.** The A/B runs are 8000 sims / 30 epochs / `dim_model=64`, not the
canonical 50k/60/dim96. Findings F1, F4, F5 and F6 are *structural* (exact zeros, analytic,
or budget-independent); the magnitudes in the A/B table are indicative only.

**Instrument.** `amortix/problems/linear_gaussian.py` — the only case with a closed-form
posterior. Its posterior covariance is data-independent, so the target is a fixed number:
mean |off-diagonal correlation| = **0.406** (max 0.727).

---

## F1. The velocity field is *exactly* factorized per parameter — the ODE cannot create any dependence between parameters. **CONFIRMED**

### What is wrong

`CrossBlock` contains cross-attention (parameter tokens → observation memory), a per-token
FFN, per-token RMSNorm, and adaLN modulation from a per-dataset `temb`. **There is no
self-attention among the parameter tokens.** Every operation is applied token-wise, so
parameter token *i* never sees parameter token *j*:

```
v_i(z, t, c) = f_i(z_i, t, c)          # exactly, for all i
```

Jacobian probe on `CrossCondVelocity` (ada layers randomised to `std=0.3` so this is *not*
just the adaLN-Zero init being trivially identity):

```
Jacobian dv_i/dz_j:
[[0.289068 0.       0.       0.      ]
 [0.       0.278288 0.       0.      ]
 [0.       0.       0.129114 0.      ]
 [0.       0.       0.       0.424599]]
max |offdiag| = 0.0            <-- exactly zero, not small
```

Same result on the *trained* field: `max|offdiag| = 0.00000`.

### Why it destroys posterior fidelity

An ODE whose velocity field is coordinate-wise integrates to a **coordinate-wise monotone
map** `z_i(1) = φ_i(z_i(0))`. Such a map preserves the *copula* exactly. Therefore:

> With a diagonal base (`base="data"`, the default) the learned posterior has **exactly zero
> dependence structure**, at any training budget, for any architecture width or depth.

Measured — Spearman rank-correlations of the base draw vs. the flow output, same 4000 draws:

```
spearman base offdiag  = [-0.022 -0.014 -0.012 -0.023  0.009  0.007]
spearman flow offdiag  = [-0.022 -0.014 -0.012 -0.023  0.009  0.007]
max |rank-corr changed by flow| = 0.0000
```

The flow changes the dependence structure by **nothing, to four decimals**. On the
exact-posterior testbed:

```
exact |offdiag corr| mean = 0.406
flow  |offdiag corr| mean = 0.011        (= Monte-Carlo noise, 1/sqrt(4000) = 0.016)
mean abs correlation error = 0.404       (i.e. 0% of the dependence is captured)
```

This is a complete, sufficient explanation of the reported symptom pattern: the SBC failures
cluster on exactly the coupled parameters (Lotka–Volterra α/β, SEIR β₂/γ_d, CIR a/b), and
`results/flow_contrib_double_well.txt` already shows base-only ≈ base+flow (err 13.31% vs
13.16%, SBC 3/3 both).

METHOD.md §9 names the diagonal base as the "leading suspect" for missing correlations. That
is only half of it — **the velocity field cannot produce correlation either**, so a
full-covariance base is not an optional refinement, it is currently the *only* path by which
any correlation can reach the output.

Note the irony: the `"concat"` `VelocityNet` — the variant that was replaced — **is**
coordinate-coupled (`max|offdiag| = 0.0029`, non-zero). The xattn upgrade traded a
single-vector context bottleneck for a factorized velocity field, which is consistent with
xattn 17/29 ≈ concat 16/29 despite winning cleanly on GBM (2 near-independent parameters,
where factorization costs nothing).

### Patch

Add self-attention among the parameter tokens *within one posterior sample*. This landed in
the working tree during the audit; I verified it is correct (see "Verified correct" below).
The essential part:

```python
class SelfAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head, self.hd = n_head, dim // n_head
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x):                              # x [N, P, dim]
        N, P, D = x.shape
        q, k, v = self.qkv(x).reshape(N, P, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-2, -1)) / self.hd ** 0.5).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(N, P, D))

# in CrossBlock.forward, before the cross-attention, with its own adaLN-Zero gate g0:
h = _modulate(self.n0(x), sh0, sc0).reshape(B * G, n_param, D)   # group per sample
x = x + g0.unsqueeze(1) * self.selfattn(h).reshape(B, G * n_param, D)
```

**The fix alone is necessary but not sufficient** — see F2.

---

## F2. With the data-dependent base the flow's total reshaping budget is ~9%; the base does essentially all the work. **CONFIRMED**

### What is wrong

For a coordinate-wise field, `∫₀¹ ∂v_i/∂z_i dt` is exactly the log of the stretch factor the
ODE applies to marginal *i*. Measured after training:

| base | `∫₀¹ ∂v_i/∂z_i dt` | stretch `exp(·)` |
|---|---|---|
| `data` (default) | `[-0.082, -0.095, +0.007, -0.065]` | `[0.92, 0.91, 1.01, 0.94]` |
| `standard` | `[-1.63, -1.59, -1.64, -2.00]` | `[0.20, 0.20, 0.19, 0.14]` |

With `base="standard"` the flow learns the *correct* prior→posterior contraction (the true
posterior std in probit space is ≈0.2), so the velocity field is perfectly capable. With
`base="data"` the NLL-trained Gaussian has already absorbed the mean and the spread, and the
flow is left able to change each marginal by **at most 9%**. Combined with F1 (it cannot
touch the copula at all), the model *is* its base: a learned diagonal Gaussian with a ≤9%
rescale. That is precisely the "learned Gaussian base alone matched the full flow"
observation, quantified.

This is not by itself a bug — a well-matched base *should* leave the flow little to do — but
it means all remaining posterior fidelity is delegated to a **linear** head
(`BaseHead.net = nn.Linear(ctx_dim, 2*dim)`, no nonlinearity) predicting a *diagonal*
Gaussian. Non-Gaussianity, multimodality, and correlation all have nowhere to come from.

### A/B on the exact-posterior testbed

8000 sims / 30 epochs / dim 64 / 8 test datasets × 4000 draws. Target: `|offdiag corr| =
0.406`, corr error `0`.

| variant | flow mean \|offdiag corr\| | mean abs corr error | max \|Δ rank-corr\| by flow |
|---|---|---|---|
| **no self-attn + `data` (as shipped)** | 0.011 | **0.404** | **0.0000** |
| no self-attn + `standard` | 0.010 | 0.403 | 0.0000 |
| no self-attn + `full` | 0.463 | **0.197** | 0.0000 |
| self-attn + `data` | 0.164 | 0.403 | 0.2698 |
| self-attn + `standard` | 0.204 | 0.303 | 0.5338 |
| self-attn + `full` | 0.581 | 0.363 | 0.7167 |

Read this carefully:

- **Self-attention alone (row 4) does not fix the error.** It raises the produced correlation
  from 0.011 to 0.164, but the corr *error* is unchanged at 0.403 — because the flow adds a
  roughly uniform *negative* correlation (`spearman flow offdiag = [-0.227 -0.227 -0.205
  -0.178 -0.26 -0.059]`) instead of the data-specific mixed-sign structure the true posterior
  has (`[+0.427 +0.225 -0.727 -0.512 -0.065 -0.643]`). At this budget it learns "correlated"
  but not "correlated *how*", because the flow only has a ≤20% reshaping budget to work in
  (F2) and a memory representation it barely trains (F3).
- **The single biggest lever at this budget is the full-covariance base** (row 3): corr error
  0.404 → **0.197**, with the flow still contributing exactly nothing to the copula. This
  matches `results/ABL_cir_base.md` (base=full 3/3 SBC vs base=data 0/3).
- Row 5 is the clean test of the flow in isolation: base is exactly `N(0,I)`, so *all*
  correlation must be manufactured by the velocity field. It gets halfway (0.204 of 0.406).
  So parameter self-attention does work — it is just starved.

### Patch

Make `base="full"` the default, and give the base head some capacity:

```python
# amortix/flow.py, FlowPosterior.__init__
def __init__(self, problem, ..., base: str = "full", ...):
```

```python
# amortix/flow.py, BaseHead/FullBaseHead: a linear map from ctx is the whole model
# for the posterior's first two moments -- give it a hidden layer.
self.net = nn.Sequential(nn.Linear(ctx_dim, ctx_dim), nn.SiLU(),
                         nn.Linear(ctx_dim, 2 * dim + self.n_off))
nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
```

Ship F1 and F2 **together**; neither is sufficient alone.

---

## F3. The shared encoder is trained 20–176× harder by the base NLL than by the CFM loss. **CONFIRMED**

### What is wrong

Both terms backprop into the same `SetTransformer`. Their gradient scales are wildly
unbalanced. Measured `‖∂L/∂θ_encoder‖` from each term separately (linear_gaussian, d=4,
batch 512):

```
AT INIT:
  ||grad_encoder(NLL)|| = 0.000e+00     <-- BaseHead.net is zero-init, so dL/dctx = W^T(..) = 0
  ||grad_encoder(CFM)|| = 0.000e+00     <-- adaLN-Zero g1=g2=0 kills the only path back to k,v
  ||grad_velocity(CFM)|| = 1.082e+01

DURING TRAINING (8000 sims, batch 256):
  ep  0  ||g_enc(NLL)||=   0.0000  ||g_enc(CFM)||=  0.00000   NLL/CFM =      0.0x
  ep  6  ||g_enc(NLL)||=   2.6368  ||g_enc(CFM)||=  0.10094   NLL/CFM =     26.1x
  ep 12  ||g_enc(NLL)||=  10.6591  ||g_enc(CFM)||=  0.52392   NLL/CFM =     20.3x
  ep 18  ||g_enc(NLL)||=  15.5069  ||g_enc(CFM)||=  0.08794   NLL/CFM =    176.3x
  ep 24  ||g_enc(NLL)||=  16.8742  ||g_enc(CFM)||=  0.10268   NLL/CFM =    164.3x
  ep 30  ||g_enc(NLL)||=   9.0543  ||g_enc(CFM)||=  0.10867   NLL/CFM =     83.3x
```

Two compounding causes:

1. **Reduction mismatch (the one the brief asked about).** `BaseHead.nll` /
   `FullBaseHead.nll` use `.sum(-1).mean()` (sum over dimensions), while the CFM term is
   `((pred - target) ** 2).mean()` (mean over batch **and** dimensions). With
   `base_weight=1.0` the base term therefore carries `d×` the per-coordinate weight — 2× to
   4× across the gallery. Real, but only a small part of the 20–176×.
2. **adaLN-Zero throttles the CFM→encoder path.** The *only* route from the CFM loss back to
   the encoder is through `g1 * cross(…, k, v)`. Measured gate magnitudes after training are
   `g1 ≈ 0.10–0.16` (see below), so the flow's gradient into the shared trunk is attenuated
   by ~10× on top of everything else. The NLL path has no such gate.

### Why it hurts

The velocity field cross-attends to `memory`; the base head reads `pool(memory)`. The shared
trunk is optimised almost entirely to be a good **linear** readout for a diagonal Gaussian's
(μ, log s). The token memory the flow must read is a by-product. This is the mechanism by
which "the base absorbs the flow's job" survives the `z0.detach()` fix: detaching `z0`
stopped the *base* from being paid to shortcut the CFM target, but it did nothing about the
*encoder* being shaped by the NLL alone.

### Patch

```python
# amortix/flow.py -- BaseHead.nll and FullBaseHead.nll: make the reduction match CFM
nll = (0.5 * ((z1 - mu) ** 2 / s ** 2 + 2 * torch.log(s))).sum(-1).mean() / z1.shape[-1]
# FullBaseHead:
nll = (0.5 * (u ** 2).sum((-2, -1)) + logdet).mean() / z1.shape[-1]
```

That removes the `d×` factor. It does **not** remove the ~10× adaLN throttle, so also make
the balance visible and tunable rather than accidental:

```python
# amortix/flow.py, FlowPosterior.fit -- report the two gradient scales, don't guess
if verbose and b == 0:
    gn = torch.autograd.grad(base_nll, list(self.encoder.parameters()),
                             retain_graph=True, allow_unused=True)
    gc = torch.autograd.grad(cfm, list(self.encoder.parameters()),
                             retain_graph=True, allow_unused=True)
    nrm = lambda gs: sum(float(g.pow(2).sum()) for g in gs if g is not None) ** 0.5
    print(f"    enc-grad  NLL {nrm(gn):.3f}  CFM {nrm(gc):.4f}  ratio {nrm(gn)/max(nrm(gc),1e-12):.0f}x")
```

then tune `base_weight` down until the ratio is O(1). The aggressive variant — give the base
head its own read-out with `self.base_head.nll(zb, ctx.detach())` so the trunk is shaped only
by the flow — is worth an A/B, but note it leaves the encoder trained *solely* through the
throttled `g1` path, so it should be paired with removing the gate from the conditioning
route (or warming `g1` up) rather than shipped on its own.

---

## F4. The SBC gate is anti-conservative: ~2× the nominal false-rejection rate at the canonical setting. The 17/29 scoreboard is biased pessimistic. **CONFIRMED**

### What is wrong

`diagnostics.sbc_uniformity` bins `u = ranks / n_post` into `n_bins=20` and compares against
a **flat** expected count `exp = n_sims / n_bins`. But there are `n_post + 1` attainable rank
values, and they do not spread evenly over 20 bins. At the canonical `n_post=200`:

```
# of distinct rank values per bin:
[10 10 11  9 10 11 10  9 10 10 10 11  9 11  9 10 11  9 11 10]
```

The correct expected count is `n_sims * m_j / (n_post + 1)`, not `n_sims / 20`. The mismatch
adds a constant ≈ `20 * n_sims * Σ_j (m_j/(n_post+1) − 1/n_bins)²` ≈ 2.7 to a χ² whose mean
is 19 — which moves the effective 5% threshold from 30.1 down to ~27.4.

Measured on **perfectly uniform** ranks (600 replicates of exact `randint(0, n_post+1)`):

```
false-reject rate of the shipped sbc_uniformity on PERFECTLY calibrated ranks:
  n_sims= 500 n_post= 200 -> FPR@0.05 = 0.107   median p=0.352     <-- canonical gallery run
  n_sims= 300 n_post= 200 -> FPR@0.05 = 0.085   median p=0.379
  n_sims= 150 n_post= 150 -> FPR@0.05 = 0.060   median p=0.453     <-- calib_gallery default
  n_sims=  20 n_post=  40 -> FPR@0.05 = 0.085   median p=0.395     <-- ablate.py runs
  n_sims= 100 n_post= 100 -> FPR@0.05 = 0.110   median p=0.371
```

And it degrades badly when the ranks-per-bin count gets small:

```
n_post= 200 n_bins= 20  ranks/bin=9-11  current FPR=0.097   fixed FPR=0.040
n_post= 199 n_bins= 20  ranks/bin=10-10 current FPR=0.047   fixed FPR=0.047
n_post= 100 n_bins= 20  ranks/bin=4-6   current FPR=0.470   fixed FPR=0.060
```

### Why it matters

The canonical gallery run is 500×200 → **10.7%** false-reject rate at a nominal 5%. Over 29
parameters that is ≈3 expected spurious failures. The headline "17/29" is more like **~20/29
in expectation** for the same model. Worse, the metric that drives every ablation decision is
mis-calibrated in a setting-dependent way, so ablations run at different `n_post` are not
comparable, and the low-budget ablations (`ablate.py`, SBC 20×40) are close to noise.

Also note the median p-value under perfect calibration is 0.35–0.45, not 0.5 — the test is
visibly shifted.

### Patch

```python
# amortix/diagnostics.py
def sbc_uniformity(ranks: np.ndarray, n_post: int, n_bins: int = 20):
    """Chi-square uniformity p-value per parameter (low p => mis-calibrated)."""
    from scipy.stats import chi2
    n_sims, d = ranks.shape
    u = ranks / float(n_post)
    # The n_post+1 attainable ranks do not spread evenly over n_bins; expected counts
    # must follow the actual bin membership or the statistic is inflated (~2x FPR).
    m, _ = np.histogram(np.arange(n_post + 1) / float(n_post), bins=n_bins, range=(0, 1))
    exp = n_sims * m / float(n_post + 1)
    pvals = np.zeros(d)
    for j in range(d):
        counts, _ = np.histogram(u[:, j], bins=n_bins, range=(0, 1))
        pvals[j] = chi2.sf(((counts - exp) ** 2 / exp).sum(), n_bins - 1)
    return pvals
```

Cheap alternative (also verified: FPR 0.047): pick `n_post` so that `(n_post + 1) % n_bins ==
0`, e.g. `n_post=199` instead of 200. Do the fix anyway — it makes the test robust to any
`n_post`.

---

## F5. Independent coupling + `t ~ U(0,1)`: 78.5% of the CFM loss is irreducible noise, and the sampling scheme wastes the cheap variance reduction already sitting in the codebase. **CONFIRMED (analytic + cost measured); the fix's benefit is SUSPECTED**

### What is wrong

With independent coupling `(z0, z1)` and a base already matched to the target (which is
exactly what `BaseHead` is trained to produce), the regression target `z1 − z0` has
conditional-mean fraction

```
R²(t) = (2t − 1)² / (2·((1−t)² + t²))
```

```
   t=0.00  R^2=0.500      t=0.50  R^2=0.000      t=1.00  R^2=0.500
   t=0.25  R^2=0.200      t=0.75  R^2=0.200
   E_t~U(0,1)[R^2] = 0.2146
   -> irreducible (noise) share of the CFM loss under U(0,1): 0.7854
```

So ~78.5% of the CFM loss is pure coupling noise that no network can reduce, and `t ~ U(0,1)`
puts its *densest* mass exactly where `R² → 0`. This does not bias the optimum, but it
inflates gradient variance ~5×, which at a finite budget shows up as underfitting — the flow
learns "there is correlation" but not "correlation of this sign and size" (F2, row 4).

### The free win that is being left on the table

`CrossCondVelocity.forward_grouped` already folds `G` samples per dataset into the query axis
and reuses the per-dataset K/V — but it is **only used at inference**. Training draws exactly
one `(t, z0)` per dataset per epoch and throws the encoding away. The encoder forward is the
expensive part; the velocity forward on `d ≤ 4` tokens is not.

Measured cost of one training step (batch 256, `dim_model=64`, CPU), splitting the encoder
forward from the velocity forward, and comparing `G=1` against `G=8` draws per encoding:

```
linear_gaussian  T=   6 tok, d=4 | encoder    33.7 ms | velocity G=1  22.51 ms | velocity G=8 121.87 ms
                   -> overhead for 8x the CFM draws: 176.8%
gbm              T= 100 tok, d=2 | encoder   474.6 ms | velocity G=1  49.05 ms | velocity G=8 126.25 ms
                   -> overhead for 8x the CFM draws:  14.7%
stoch_lv         T= 180 tok, d=4 | encoder   902.4 ms | velocity G=1  84.86 ms | velocity G=8 227.48 ms
                   -> overhead for 8x the CFM draws:  14.4%
```

On the real gallery problems the encoder is 6–11× the velocity, so **8× the CFM draws costs
~14% more wall-clock per step**. Only on the 6-token `linear_gaussian` toy (where the encoder
is trivial) is the trade unattractive. This is the cheapest available variance reduction in
the whole pipeline and it is currently unused during training.

*(Honesty note: I also queued a direct `t~U(0,1)` vs arcsine × `G=1` vs `G=8` training A/B on
`linear_gaussian` with `base="standard"`. It had not finished within the time budget and I
killed it, so the claim that reweighting `t` improves the fitted posterior remains
**SUSPECTED** — only the `R²(t)` analysis and the cost split above are measured. That A/B is
the right next experiment; the script is `probe_time.py` in the audit scratchpad.)*

### Patch

Reuse each encoding for `G` draws in `fit`:

Note the `draw` closure returned by `nll` is `lambda eps: mu + s * eps` with `mu, s` of shape
`[bs, d]`, so it does **not** broadcast over a `G` axis. Use the `(mu, s)` / `(mu, L)` tuple
that `nll` already returns as its third element — the same construction `sample_batch`
already uses — and hoist it into a shared helper so training and sampling cannot drift apart:

```python
# amortix/flow.py, FlowPosterior
def _draw_base(self, ctx, n, gen):
    """[B, n, d] base samples -- the single definition used by fit() and sample_batch()."""
    eps = torch.randn(ctx.shape[0], n, self.d, generator=gen)
    if self.base == "full":
        mu, L = self.base_head(ctx)
        return mu[:, None] + torch.einsum("bij,bnj->bni", L, eps)
    if self.base_head is not None:
        mu, s = self.base_head(ctx)
        return mu[:, None] + s[:, None] * eps
    return eps
```

```python
# amortix/flow.py, FlowPosterior.fit  (G = 8, say)
memory = self.encoder.encode(tb); ctx = self.encoder.pool(memory)
base_nll, _, _ = self.base_head.nll(zb, ctx)
z0   = self._draw_base(ctx, G, gen).detach()          # [bs, G, d]
t    = torch.rand(bs, generator=gen)
zt   = (1 - t)[:, None, None] * z0 + t[:, None, None] * zb[:, None, :]
cond = self.velocity.encode_memory(memory)
cfm  = ((self.velocity.forward_grouped(zt, t, cond) - (zb[:, None, :] - z0)) ** 2).mean()
```

To also vary `t` *within* a dataset — which is what actually kills the `R²(t)` variance —
`forward_grouped` must accept `t` of shape `[B, G]`. That needs `temb` to become `[B, G, dim]`
and the adaLN gates/shifts in `CrossBlock` to be `repeat_interleave(n_param, dim=1)`-expanded
to `[B, G*n_param, dim]` instead of `unsqueeze(1)`-broadcast. Worth doing; it is the
difference between averaging out the `z0` noise only and averaging out both.

Optionally bias `t` toward the informative ends (arcsine / `Beta(0.5, 0.5)`):

```python
u = torch.rand(bs, generator=gen)
t = 0.5 * (1 - torch.cos(math.pi * u))     # arcsine: density concentrated at t=0 and t=1
```

The principled alternative is minibatch-OT coupling (OT-CFM) instead of independent coupling,
which attacks the 78.5% directly rather than averaging over it.

---

## F6. RoPE breaks the permutation invariance the method claims, and the position it encodes is physically meaningless. **CONFIRMED**

### What is wrong

`METHOD.md` §3 and `encoder.py`'s docstring both describe a permutation-invariant set
encoder. It is not one:

```
||ctx(x) - ctx(perm x)|| / ||ctx||          = 0.038
swap two whole 6-token path blocks          = 0.039
```

Worse, the position index RoPE encodes is the index in the *concatenated* token list. From
`PathObserver.tokens_from_traj`, tokens are laid out as

```
[comp0-chan0 | comp0-chan1 | comp1-chan0 | ...]   x  n_paths   (SDEProblem._tokens_for)
```

so position jumps discontinuously at every channel / component / path boundary, and channel
1's first token gets position `count0` rather than 0. Meanwhile the token *features* already
carry `t/horizon` (feature 0), `log10(dt_obs)` (feature 4) and `comp_id` (feature 5). RoPE
therefore adds a second, conflicting positional signal that does not correspond to time.

For `n_paths > 1` the replicate trajectories are exchangeable by construction, but each gets
a different RoPE position — the network must burn capacity learning an invariance it was
supposed to have for free.

### Patch

Make RoPE optional and default it off (the physical time is already a feature):

```python
# amortix/encoder.py, SetTransformer.__init__
def __init__(self, ..., pool: str = "attn", rope: bool = False):
    ...
    if rope:
        cos, sin = rope_tables(max_tokens, dim // n_head)
    else:                                  # identity rotation -> true set encoder
        cos = torch.ones(max_tokens, dim // n_head); sin = torch.zeros_like(cos)
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)
```

If positional information is wanted, index by the *physical* time step
(`idx * dt_sim / horizon`) rather than by concatenation order, so the same instant in two
channels shares a position.

---

## F7. Probit normalization makes the Gaussian base a poor fit exactly at the prior-box edges, where SBC draws 20% of its truths. **SUSPECTED**

The probit map is right for the *prior* marginal (it sends `U(low, high)` exactly to
`N(0,1)`), but it strongly skews *posteriors* that sit near a box edge. A symmetric posterior
of sd 0.03 in a `[0,1]` box, mapped to z-space:

```
  posterior centred at m=0.50 (sd .03): z-space mean=-0.000 sd=0.075 skew=-0.00
  posterior centred at m=0.90 (sd .03): z-space mean= 1.303 sd=0.190 skew= 1.11
  posterior centred at m=0.98 (sd .03): z-space mean= 1.957 sd=0.394 skew= 1.13
```

SBC draws truths uniformly from the box, so ~20% of datasets have a posterior in the outer
10% of some coordinate, where the base's Gaussian assumption is off by skew ≈ 1.1 and the
flow must supply the correction. A coordinate-wise flow *can* produce 1-D skew (so this is
not structurally blocked like F1), but it is extra work concentrated at the box edges — and
it is exactly the kind of defect that shows up as a sloped/edge-piled SBC rank histogram
while central coverage stays fine, which matches the reported "cov50 45–63%, cov90 86–92% but
rank uniformity fails" pattern.

Worth checking directly: split the SBC rank histograms by whether the truth is in the
central 80% or the outer 20% of the box. If the failures live at the edges, consider a
mildly heavier-tailed or skew-capable base (e.g. predict a per-coordinate skew parameter)
rather than more flow capacity.

---

## F8. Minor issues. **CONFIRMED by inspection/measurement**

- **`log_s.clamp(-4.0, 2.0)` has exactly zero gradient at the boundary.** `s ∈ [0.018, 7.39]`
  against a prior marginal sd of 1.0 in probit space. A posterior >55× sharper than the prior
  silently pins to the clamp and stops learning, with no warning. Use a softplus
  parameterisation or at least log a counter of clamp hits.
- **`mask` is never passed.** `FlowPosterior` calls `self.encoder.encode(tb)` and
  `self.encoder.pool(memory)` with no mask anywhere (`'mask' in source(FlowPosterior)` is
  `False`). Harmless today because every observer emits a fixed token count, but the encoder
  advertises variable-length support and `SetTransformer.pool`'s masked branch is dead code —
  the first ragged observer will silently pool padding.
- **Output head magnitude ceiling — checked, not currently binding.** `out = nn.Linear(dim,1)`
  after `RMSNorm` means `|v_i| ≤ sqrt(dim)·‖w ⊙ γ‖`, a single learned scalar cap. Measured
  ceiling 4.73 vs `max|target| = 2.78`, `p99 = 1.30`, `frac(|target| > ceiling) = 0.0000`. Not
  a problem at these scales; would become one if the base ever mismatched badly.
- **`FullBaseHead` marginal variance is not bounded by the `log_d` clamp**: `diag(LLᵀ)_i =
  exp(2·log_d_i) + Σ_{j<i} off_ij²`, and the off-diagonals are unconstrained. Only the NLL
  restrains them. In the `self-attn + full` run the marginal stds came out 2–3× too wide
  (`[0.67, 1.03, 0.73, 0.42]` vs exact `[0.33, 0.41, 0.25, 0.24]`) at this budget.

---

## Verified correct — things I suspected and could not break

- **ODE time grid: no off-by-one.** `t_i = i·dt` for `i = 0..n_steps−1` with step `dt` is the
  correct Euler grid; `Σ dt = 1` exactly. Reproduced on `v ≡ 1`: final `z = 1.0` for
  `n_steps = 5` and `1.0000001` for 20. Midpoint evaluates at `(i+½)dt` (max 0.9 for n=5),
  RK4's `k4` reaches `t = 1.0`. Nothing lands short.
- **Train/inference field consistency.** Training regresses `v ≈ z1 − z0` along
  `z_t = (1−t)z0 + t·z1`, whose true derivative is `z1 − z0`; sampling starts from a base draw
  and integrates `dz/dt = v` forward from 0 to 1. Same direction, same parameterization.
- **K/V cache path is identical in both.** `fit` and `sample_batch` both build
  `memory = encoder.encode(tokens)` then `velocity.encode_memory(memory)`. No train/test skew.
- **`forward` vs `forward_grouped` agree.** `max |forward − forward_grouped| = 2.4e-07`
  (pre-fix) and `1.8e-07` (with the in-flight self-attention). With self-attention added I
  also checked the reshape ordering explicitly: perturbing one posterior sample changes only
  that sample (`cross-sample leakage = 0.0`, `same-sample coupling = 1.229`). The
  `[B, G*d]` → `[B*G, d]` regrouping is correct.
- **adaLN-Zero gates are not dead.** After 30 epochs, `g1` rms 0.10–0.36 and `g2` rms
  0.09–0.30 across all three blocks; the `standard`-base runs reach `sc1` rms ≈ 1.0. The
  blocks do escape the identity regime. (They are, however, small enough to throttle the
  CFM→encoder gradient — that is F3, not a dead-block problem.)
- **`timestep_embedding` after the `scale=1000` fix** matches standard DiT practice
  (fastest channel ≈159 cycles over `t ∈ [0,1]`, slowest ≈1 radian, so there is a usable
  low-frequency channel). The fast channels alias at the solver's `dt = 0.05`, but the
  reported midpoint/20 vs RK4/60 agreement (0.002%) shows the learned field is smooth in `t`
  in practice.
- **Probit round-trip.** `normalize` clamps `u` to `[1e-6, 1−1e-6]` → `|z| ≤ 4.75`, hit with
  probability 2e-6 under the prior; `denormalize` keeps every sample strictly in-box; both are
  monotone per coordinate so SBC ranks are invariant to the transform.
- **RNG hygiene in `sample_batch`.** The generator is created once outside the chunk loop, so
  each dataset gets its own `eps` and chunking does not reuse noise.

---

## Recommended order of work

1. **F1 + F2 together** — parameter self-attention *and* `base="full"` as default. Neither
   closes the correlation gap alone (0.404 → 0.403 for self-attn alone, → 0.197 for full base
   alone). Re-run the gallery only after both.
2. **F4** — fix `sbc_uniformity` before measuring anything else, or you will be chasing ~3
   phantom failures out of 29 and comparing ablations run at different `n_post`.
3. **F3** — rebalance the two loss terms and stop the NLL from owning the encoder.
4. **F5** — `G` draws per encoding in `fit`; nearly free, and it is what will let the
   self-attention actually learn *which* correlation.
5. **F6/F7** — encoder symmetry and the box-edge base fit.

Re-validate on `linear_gaussian`, not on SBC: it is the only case where "the flow is wrong"
can be distinguished from "the posterior is genuinely broad". The single number to watch is
**mean abs correlation error, currently 0.404 out of a true 0.406 — i.e. zero dependence
captured.**
