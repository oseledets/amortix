# Encoder debug on `linear_gaussian`: does the sufficient statistic survive?

**The question.** The amortized posterior on `linear_gaussian` is displaced from the
exact one, and an oracle test attributed about half of that displacement to the
encoder. The task is trivial in principle — the posterior mean is an exactly linear
function of the six observed values, `mu = Sigma Aᵀ y / sigma²`, and the encoder sees
those six values as six tokens `[y_i, i/6]` — so an encoder that cannot produce it
would be the finding.

**Answer.** The statistic survives the encoder's **token memory** with room to spare:
a linear probe recovers `mu` from it to 0.08 exact-posterior sd with the current
encoder, 0.03 sd once the token features are standardized — and §2 shows a location
error that small is worth *nothing* on the target metric. It does **not** survive the
**pooled context** (1.1 sd), for a mechanical reason: under the default
`conditioning="xattn"` the pooling module sits on a dead branch of the graph and
never receives a gradient. But repairing the pooled context does **not** improve the
posterior — every change that made the context more linearly readable left the energy
distance flat or made it worse.

What *did* close a large part of the gap is **standardizing the raw token features**:
energy distance to the exact posterior **0.00427 → 0.00184** at a matched budget over
two seeds (floor 0.00069) — it removes 68 % of the excess over the floor. And it
does so *without* adding information — by §2 the extra statistic it exposes is worth
only 0.0002 of that. The encoder's remaining job was never to carry the sufficient
statistic; it was to present it in a scale the velocity can read.

Reproduce with `examples/encoder_ab.py` (`--floor`, `--probe`, or a training A/B via
`--cfg`); `input_norm=True` is now the default, so `--cfg '{"input_norm":false}'` is
the old encoder.

---

## 1. Where the information is

Ridge probe from the encoder's output to `mu`, fitted on 3000 datasets and scored on
3000 held out, on **trained** models (one seed each, the eight configurations of §7).
Reported as **residual RMSE in units of the exact posterior sd**. R² is the wrong
unit here — the prior sd is 1.73 against a posterior sd of 0.26–0.43, so R² = 0.99
already means half a posterior sd of location error — but it is quoted because the
task asked for it.

| encoder (trained) | ctx R² (per parameter) | ctx err | memory err | ratio |
|---|---|---|---|---|
| **default** | 0.965 / 0.942 / 0.943 / 0.969 | **1.137** | **0.084** | 13.5× |
| no final RMSNorm | 0.942 / 0.945 / 0.963 / 0.980 | 1.061 | 0.055 | 19.3× |
| mean pool | 0.976 / 0.966 / 0.982 / 0.991 | 0.742 | 0.088 | 8.4× |
| input standardization | 0.993 / 0.991 / 0.994 / 0.995 | 0.436 | 0.033 | 13.2× |
| + MLP embed | 0.993 / 0.994 / 0.996 / 0.997 | 0.373 | 0.030 | 12.4× |
| + MLP embed + mean pool | 0.994 / 0.993 / 0.996 / 0.998 | 0.365 | 0.038 | 9.6× |
| + MLP embed + mean pool, no RoPE | 0.982 / 0.987 / 0.992 / 0.994 | 0.545 | 0.043 | 12.7× |
| + MLP embed + mean pool, no final RMSNorm | 0.988 / 0.994 / 0.995 / 0.997 | 0.410 | 0.032 | 12.8× |

*(err = residual RMSE / exact posterior sd, averaged over the four parameters.)*

The pooled context is 8–19× worse than the memory it is computed from, in every
configuration. Sanity check: the same probe from the raw `y` gives R² = 1.000000 and
error 0.000, as it must.

The same probe on **untrained** encoders — which, per §3, is the state the pooling
never leaves:

| encoder (random init) | ctx | memory |
|---|---|---|
| default (attention pool, RoPE, linear embed) | 3.195 | 0.286 |
| + input standardization | 1.362 | 0.024 |
| + input standardization + MLP embed | 1.385 | 0.011 |
| + input standardization + MLP embed + mean pool | 1.266 | 0.011 |
| + input standardization, no final RMSNorm | 1.318 | **0.002** |

## 2. What those numbers are worth: calibrating the metric

Take the *exact* posterior, displace it rigidly by `d` posterior sd, and measure
`energy_distance_scaled` against the undisplaced exact posterior (16 datasets, 400 vs
400 draws — so the `d = 0` row is the Monte-Carlo floor):

| pure location shift | energy distance | | pure width error | energy distance |
|---|---|---|---|---|
| 0.00 sd | 0.00068 | | ×1.00 | 0.00068 |
| 0.05 sd | 0.00073 | | ×1.02 | 0.00068 |
| 0.10 sd | 0.00092 | | ×1.05 | 0.00074 |
| 0.15 sd | 0.00106 | | ×1.10 | 0.00096 |
| 0.20 sd | 0.00144 | | ×1.20 | 0.00185 |
| 0.30 sd | 0.00284 | | | |
| 0.44 sd | 0.00509 | | | |

Read §1 through this table:

- the trained model's measured distance (§7: 0.0029) corresponds to ≈ **0.30
  posterior sd** of pure location error, which matches the independently reported
  0.2–0.44 sd displacement;
- the memory's own extraction error — 0.084 sd, or 0.033 sd standardized — is worth
  **0.0009**, resp. **0.0007**: the Monte-Carlo floor. A downstream that used the
  memory perfectly would be *at* the floor;
- the reported 1.04 width ratio falls between the ×1.02 and ×1.05 rows, i.e. **at the
  floor**. At this accuracy the metric is a location metric; width is already free.

**So the encoder is not information-limited.** Everything below follows from that.

## 3. The pooling never trains

`flow.py` trains the base head on a **detached** context:

```python
memory = self.encoder.encode(tb, mb)
ctx    = self.encoder.pool(memory, mb)
base_nll, draw, _ = self.base_head.nll(zb, ctx.detach())   # <- detached
```

Under `conditioning="xattn"` (the default) nothing else consumes `ctx` — the velocity
cross-attends to `memory`. So `pool()` is on a dead branch. Measured after
`fit(steps=300)`, relative parameter change ‖Δθ‖/‖θ‖:

| parameter | change |
|---|---|
| `encoder.embed.weight` | 2.3e-02 |
| `encoder.blocks.0.attn.qkv.weight` | 1.5e-01 |
| `encoder.blocks.0.ffn.fc2.weight` | 2.2e-01 |
| `encoder.attn_pool.q` | **1.4e-07** |
| `encoder.attn_pool.kv.weight` | **1.6e-07** |
| `encoder.attn_pool.proj.weight` | **1.6e-07** |

and directly, `encoder.attn_pool.q.grad is None`. The base head — one
`Linear(64, 8)` — therefore reads a *random projection of a random-weighted attention
read* of the memory, and that vector sets the posterior's starting centre and spread.

This corrects a piece of project folklore: `METHOD.md` §3 credits attention pooling
with fixing calibration relative to mean-pooling. That A/B predates
`conditioning="xattn"`; under `concat` the velocity consumes `ctx`, so the pool *was*
trained then. Under today's default it is not.

## 4. Why a frozen pool cannot carry a linear statistic (and a mean can)

Hand the pooling an *ideal* memory — token `i` carrying its own standardized value in
its own channel, so the statistic is present and needs only a linear read of the
token sum:

| pooling of the ideal memory | probe error (posterior sd) |
|---|---|
| mean | **0.0000** |
| frozen attention pool | 0.107 |
| mean, after the encoder's final RMSNorm | 2.542 |
| frozen attention pool, after the final RMSNorm | 2.545 |

A mean is *linear* in the memory, so a linear base head composes with it exactly. A
softmax read is *nonlinear and content-dependent* — its weights move with the very
values being summed — and with frozen random weights nothing corrects that. The last
two rows are a separate effect: the final `RMSNorm` divides each token by its own
norm, applying a saturating, magnitude-dependent gain to the channel whose magnitude
carries the answer.

Widening the frozen pool does not rescue it (probe at initialization, which is the
state it stays in):

| pooling | ctx dim | ctx probe |
|---|---|---|
| attention, 1 query (default) | 64 | 1.369 |
| attention, 4 queries | 64 | 1.323 |
| attention, 8 queries | 64 | 1.398 |
| attention, 4 queries, `pool_dim=256` | 256 | 1.110 |
| mean | 64 | 1.222 |
| sum | 64 | 1.222 |

## 5. Why a linear embed cannot form the statistic

A pooled set summary is `Σ_i f(token_i)`, and the target is `Σ_i c_i y_i` with `c_i`
depending on the token's **index**. A linear embed gives
`f = W_v y_i + W_p (i/6) + b`, whose value coefficient `W_v` is the same for every
token: the summary can only see `Σ_i y_i`, never a general linear functional. The
value has to be *multiplied* by a function of its own coordinate, which needs a
per-token nonlinearity — an MLP embed, or, indirectly, attention plus RoPE. This is
why dropping RoPE *on its own* makes the encoder worse (§6): with a linear embed,
RoPE's per-slot rotation is one of the few mechanisms available for making a token's
contribution depend on which token it is.

## 6. Encoder capacity, measured without the flow

Train `encoder → pool → Linear(ctx, 4)` by MSE onto `mu`. No flow, no base head, no
ODE — and, unlike the real xattn setup, the pooling *is* trained here. 4000 steps,
2 seeds, mean over the four parameters of residual RMSE / posterior sd:

| config | error (posterior sd) |
|---|---|
| input standardization | **0.105** |
| MLP embed | 0.137 |
| input standardization + MLP embed | 0.136 |
| no final RMSNorm | 0.158 |
| **default** | **0.163** |
| attention pool, 4 queries | 0.169 |
| no RoPE | 0.205 |
| sum pool | 0.208 |
| mean pool | 0.212 |

The default's 0.163 posterior-sd residual is the same order as the ~0.22 sd of
location error the oracle test attributed to the encoder, so this screen is measuring
the right thing. The pooling rows do not transfer to the real setup (here the pool is
trained; there it is frozen).

Permutation invariance, separately: the relative change of the context under a
permutation of the six tokens is **1.96 %** with RoPE and **0.00 %** without — RoPE
is the only thing breaking the set property the method advertises.

## 7. Energy distance to the exact posterior

Protocol: 32 held-out datasets, 400 posterior draws each against 400 exact draws,
`energy_distance_scaled` by the prior range, averaged over the 32. Training:
`n_train=20000`, batch 64, 9360 steps (30 epochs), one encoder change at a time, with
`flow.py` **pinned to a single snapshot** for every row (that file was being rewritten
by the optimization workstream while these ran; the snapshot includes its cosine
schedule, warmup, EMA and `k_pairs=4`). The reported figure averages the late
checkpoints (steps ≥ 4500) plus the final one; the spread across those four
checkpoints within a run is 0.0002–0.0005.

**Monte-Carlo floor on this eval set: 0.00069.**

| # | encoder | energy distance | vs default |
|---|---|---|---|
| 1 | default **+ input standardization** | **0.00237** | −17 % |
| 2 | default + input standardization + MLP embed | 0.00284 | ±0 |
| 3 | **default** (attention pool, RoPE, linear embed) | 0.00285 | — |
| 4 | default, no final RMSNorm | 0.00353 | +24 % |
| 5 | input std. + MLP embed + **mean pool** | 0.00375 | +32 % |
| 6 | default + **mean pool** | 0.00388 | +36 % |
| 7 | input std. + MLP embed + mean pool, no RoPE | 0.00403 | +41 % |
| 8 | input std. + MLP embed + mean pool, no final RMSNorm | 0.00632 | +122 % |

One seed per row (seed 0) — this table ranks eight *different* encoders rather than
replicating one, and §8 repeats the decisive comparison over two seeds at a matched
budget. Rows 1 vs 3 (0.00237 vs 0.00285) are within what a single seed can resolve;
the group separation in (b) is not.

Two things to read off this table.

**(a) The metric does not follow the probe.** Rows 1–8 span 8–19× in context
readability and 3× in memory readability, and the ordering of the energy distance is
unrelated to either. The best-probed encoder (row 5: ctx 0.365, memory 0.038) is 32 %
*worse* than the default, whose context probe is 3× poorer. Making the pooled context
linearly readable is not the lever — which is what §2 predicts, since the information
loss in the memory is worth ~0.0002 and the differences here are ~0.001–0.003.

**(b) Mean pooling costs about 0.0012, consistently.** Every mean-pool row (5, 6, 7,
8: 0.0038–0.0063) is worse than every attention-pool row (1, 2, 3, 4: 0.0024–0.0035),
across four different trunks. This survived every explanation I tested: at
initialization a mean beats the frozen attention read on the linear statistic
(1.22 vs 1.37), on the base head's actual mean target — the posterior mean of
`probit(m)` under the *box-truncated* posterior — (0.269 vs 0.293 of its sd) and on
its log-sd target (0.630 vs 0.650); and the two contexts have the same
signal-to-offset ratio (0.66 vs 0.71). So the cost is not context readability, not
the nonlinearity of the base head's target and not conditioning; it appears during
training, and I have no verified mechanism for it. Recorded as an unexplained
negative result: **do not switch the pooling on the strength of the probe.**

The leading untested hypothesis is the context's *magnitude*: a mean over
RMS-normalised tokens has ‖mean‖ ≈ 0.68 against the attention read's 0.21, and the
base head is a zero-initialised `Linear` whose second output is `log s`, so the same
weight step moves the predicted base *spread* three times as far. A noisy base spread
is expensive in a way a noisy base mean is not — the flow can translate a posterior
but an ODE cannot create spread from a point. Two cheap falsifiers for whoever
follows: `pool="sum"` (6× the magnitude of `mean`) should then be *worse* than
`mean`, and a mean followed by a fixed normalisation should recover the attention
pool's score.

**(c) The plateau moved while this was measured.** On the *same* 12-dataset reference
set used by `examples/train_monitored.py`, the unchanged default encoder now scores
**0.0035** where it scored ≈ 0.0087 before. The whole 2.5× came from the parallel
optimization work in `flow.py` (cosine + warmup + EMA + `k_pairs`), none of it from
the encoder. The premise of this investigation — that half the location error
is the encoder's — was measured at the old operating point. At the new one the
encoder's share is, by §2, at most ~0.0002 of the 0.0022 that still separates the
model from the floor.

## 8. Two-seed check on the change that mattered

Same protocol at a 4680-step budget — a complete, fully cosine-annealed run at half
the budget, so both arms are converged training runs — two seeds per arm, both arms
on identical code:

| encoder | seed 0 | seed 1 | **mean** | excess over floor | ctx probe | memory probe |
|---|---|---|---|---|---|---|
| default | 0.00514 | 0.00340 | **0.00427** | 0.00358 | 0.768 / 0.850 | 0.087 / 0.092 |
| **+ input standardization** | 0.00170 | 0.00197 | **0.00184** | 0.00115 | 0.374 / 0.404 | 0.021 / 0.024 |

**−57 % on the metric, −68 % on the excess over the Monte-Carlo floor, with the two
arms' seeds disjoint** (0.0034–0.0051 against 0.0017–0.0020). It is also worth more
than doubling the training budget: standardized at 4680 steps (0.00184) beats the
unstandardized default at 9360 steps (0.00285).

Note what this does *not* say. By the §2 calibration the memory's information content
went from "worth 0.0009 if used perfectly" to "worth 0.0007" — 0.0002 of the 0.0024
that was actually gained. Standardization is therefore not paying off by carrying
*more* of the statistic; it pays off by presenting it in a form the velocity's
cross-attention can use. The encoder's job here is conditioning, not information.

## 9. Recommended encoder

```python
SetTransformer(..., input_norm=True)     # the change; everything else as before
#              pool="attn", rope=True, embed="linear", final_norm=True
```

`input_norm=True` is now the default in `amortix/encoder.py`. It is the only change
that improved the target metric — **−57 % over two seeds at a matched budget** (§8) —
and it improves everything else that was measured: in the same two-seed A/B the
memory probe by 4× and the context probe by 2×, and the flow-free capacity screen by
1.6× (§6). Nothing else earned a change of default:

| considered | verdict |
|---|---|
| MLP embed (`embed="mlp"`) | better probe, no metric gain — kept as an option |
| mean / sum pooling | better probe, **worse** metric (§7b) — not adopted |
| Perceiver latents (`n_query`), wider `pool_dim` | no material effect on the frozen pool |
| dropping RoPE (`rope=False`) | worse with a linear embed, no gain with an MLP embed; the only route to true permutation invariance, so it stays available |
| dropping the final RMSNorm (`final_norm=False`) | 12× better memory probe at init, but +24 % on the metric — not adopted |

Why standardization matters beyond this testbed: the observers emit tokens in
physical units. `linear_gaussian` mixes a value spanning ±15 with an index in [0,1);
`PathObserver` mixes `t/horizon ~ 1`, `Δx` and a quadratic-variation cue `Δx²` that
can be `1e-4`, `log₁₀(dt_obs) ~ -2` and a 0/1 `comp_id`. The pre-norm blocks see a
residual stream whose scale is set by the largest feature. `FeatureNorm` keeps running
per-feature statistics over *valid* tokens only (mask-aware), freezes them at
evaluation so one dataset's encoding never depends on its batch-mates, debiases the
warmup so small budgets are not trained under a moving normalization, and passes
degenerate constant features (`comp_id` on a 1-D SDE) through unscaled instead of
dividing by zero.

**Scope.** Everything here is measured on `linear_gaussian` only. The default change
is safe by construction elsewhere (`uv run pytest -q`: 36 passed), but the gallery
numbers should be re-measured before being quoted.

## 10. The one thing the encoder cannot fix from inside

The pooling can be made parameter-free, which is what would make it immune to
receiving no gradient — but it cannot be made *trained* from inside `encoder.py`,
because the gradient is cut in `flow.py`. If the owner of that file wants a trained
pooling back while keeping the property that motivated the detach (the base-head NLL
must not reshape the shared trunk), the change is to move the cut one step earlier:

```python
# amortix/flow.py, FlowPosterior.fit
ctx = self.encoder.pool(memory.detach(), mb)      # trunk still protected...
base_nll, draw, _ = self.base_head.nll(zb, ctx)   # ...pooling now trained
```

That trains `AttentionPool` and the base head on the NLL while the trunk still sees
only the CFM gradient — the stated intent of the current code, which the present form
does not achieve. Given §7 it should be expected to help the *context probe* and it is
an open question whether it helps the metric; it is worth one A/B, not more.

Two further notes for whoever picks this up:

- `FlowPosterior.__init__` forwards `pool=` to the encoder with its own default
  `"attn"`, so the encoder's `pool` default is dead code as far as the library's own
  entry point is concerned. Changing the pooling means changing that signature (or
  passing `FlowPosterior(prob, pool=...)`).
- The remaining 0.0012 above the floor (§8, standardized arm) is **not** encoder
  information loss. On the calibration of §2 it is ~0.2 posterior sd of location
  error, while the statistic sits in the memory at 0.02 sd — a hundredth of what is
  being lost. It goes missing between the memory and the samples: in the base head's
  read of a frozen-pool context, in the velocity's cross-attention read of the
  memory, or in the ODE. That is where the next factor is.
