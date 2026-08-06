# Calibration: diagnosis, fix, and gallery-wide verification

The amortized posterior must be **calibrated** (its credible intervals must have
the right coverage), not just accurate. We test this with Simulation-Based
Calibration (SBC, Talts et al. 2018): the rank of the true parameter among L
posterior draws must be uniform. A single coverage number can look fine while SBC
fails — so SBC is the real test (`amortix/diagnostics.py`).

## Diagnosis

A first OU run (mean-pool encoder, standard N(0,I) base) reported cov90 ≈ 90% but
**failed SBC**: μ was under-confident (cov50 = 85%, err 2% ≪ std 11%), θ/σ biased.
Root causes:
1. **Source/target spread mismatch.** Flow matching can map any source to any
   target, but if the base spread (N(0,I), std 1) is far from the conditional
   posterior spread (narrow for well-identified params), the deterministic ODE
   must do a large stiff contraction a finite net underfits → mis-dispersion.
2. **Mean-pool encoder** bottlenecks the observation summary.
3. (Minor) affine normalization mapped the prior to a *bounded uniform* while the
   base stayed Gaussian — a marginal mismatch.

## Fix (all in the core, toggleable)

- **probit normalization** (`prior.py`): `z = Φ⁻¹((m−low)/(high−low))` maps the
  uniform prior *exactly* onto the N(0,I) base; posterior samples stay in-box.
- **attention pooling** (`encoder.py`, `pool="attn"`): a learned query (Set
  Transformer PMA) replaces mean-pool, keeping per-parameter information.
- **data-dependent base** (`flow.py`, `base="data"`): a head predicts
  N(μ̂(d), ŝ(d)²) trained by Gaussian NLL to match the posterior's mean/spread, so
  the flow only refines the *shape*. Aligning the source spread to the target per
  dataset is the key fix. (NLL keeps ŝ ~ posterior std: an ODE can't create spread
  from a point, so the base must seed it.)

Caveat: the data-base has more parameters, so it needs an adequate budget —
at 8k/25 it was *worse* than the standard base; the payoff appears at ≥12k/40.

## Gallery-wide verification (attn-pool + data-base, n_train=12000, epochs=40, SBC 300×150)

| case | dim | calib-err | SBC pass | mean cov50 | mean cov90 |
|---|---|---|---|---|---|
| ou | 3 | 4.0pp | 1/3 | 58% | 93% |
| seir | 5 | 2.9pp | 3/5 | 48% | 87% |
| gbm | 2 | 3.7pp | 1/2 | 57% | 93% |
| cir | 3 | 8.0pp | 1/3 | 64% | 91% |
| double_well | 3 | 2.7pp | 2/3 | 50% | 88% |
| stoch_lv | 4 | 1.5pp | 3/4 | 52% | 89% |
| fhn | 4 | 4.1pp | 1/4 | 49% | 88% |
| sindy_sde | 5 | 3.6pp | 4/5 | 49% | 88% |
| **total** | | **3.8pp avg** | **16/29** | ~52% | ~90% |

Per-parameter SBC-p (p>0.05 = calibrated):
```
        ou: theta 0.69  mu 0.00  sigma 0.00
      seir: beta1 0.00  beta2 0.31  alpha 0.03  gamma_r 0.70  gamma_d 0.73
       gbm: mu 0.25  sigma 0.00
       cir: a 0.27  b 0.00  sigma 0.03
double_well: theta1 0.24  theta2 0.25  sigma 0.02
  stoch_lv: alpha 0.33  beta 0.08  delta 0.85  gamma 0.00
       fhn: a 0.00  b 0.00  eps 0.12  I 0.03
 sindy_sde: c0 0.06  c1 0.05  c2 0.13  c3 0.17  sigma 0.30
```

## Read

- **Coverage is calibrated across the gallery** (cov50 48–64% vs target 50,
  cov90 87–93% vs target 90; mean calib-err 3.8pp). SBC pass rose from 4/11 at a
  lean budget to 16/29 at 12k/40 — confirming the fix needs adequate training.
- **Residual 1 — diffusion σ.** σ fails SBC in ou/gbm/cir/double_well (p≈0) but
  **passes in sindy_sde (p=0.30)**, which uses 80 fast-channel tokens over 1000
  steps vs 50–60 for the others. ⇒ σ mis-calibration is a *high-frequency data
  budget* issue (quadratic variation resolution), not a method defect — fixable
  by widening the fast channel.
- **Residual 2 — weakly identified params** (fhn a/b/I with only v observed; ou μ;
  cir b; seir β1/α): coverage is fine but the rank distribution is slightly
  non-uniform (residual bias/skew) — the hardest posteriors, limited by
  information/capacity.
- **cir** is the weakest case (calib-err 8pp, cov50 64% = mild over-dispersion).

Reproduce: `uv run python examples/calib_gallery.py --n_train 12000 --epochs 40`
(writes `CALIB_GALLERY_RESULTS.md` + `.json`).

---

## Round 2: dense conditioning (`xattn`) — the current state

Collapsing the whole dataset into one context vector was the next suspect (same
failure mode as conditioning an image model on a single global-pooled vector).
`CrossCondVelocity` replaces it: one token per parameter cross-attends to the
**full observation token memory** at every block, with adaLN(t) modulation.

**Controlled A/B on GBM** (same budget 12k/40) — dense conditioning clearly wins:

| conditioning | calib-err | SBC-pass | mu | sigma |
|---|---|---|---|---|
| concat | 4.3pp | 0/2 | 0.002 | 0.004 |
| **xattn** | **1.7pp** | **1/2** | **0.356** | 0.038 |

**Scaled up** (GBM, 50k/60, dim 96, SBC 500×200): **2/2 pass**, mu 0.165,
**sigma 0.571** — the σ residual disappears entirely with dense conditioning plus
an adequate budget. (An earlier hypothesis that σ needed more high-frequency
tokens was *refuted*: widening the fast channel 60→240 did nothing.)

**Controlled A/B on a coupled case** (double-well, 20k/50, **3 seeds**):

| conditioning | calib-err | mean SBC-pass |
|---|---|---|
| concat | 2.5±0.6pp | 1.3/3 |
| **xattn** | **2.1±0.6pp** | **1.7/3** |

`xattn` is not worse on coupled dynamics (slightly better, within seed noise) —
so the gallery-canon dip for double-well was seed noise. σ is marginal for *both*
modes here (p≈0.06), i.e. budget-limited, not conditioning-limited. **`xattn` is
therefore the default.**

## Canonical gallery run (xattn, 50k/60, SBC 500×200)

| case | dim | calib-err | SBC pass | cov50 | cov90 |
|---|---|---|---|---|---|
| ou | 3 | 7.6pp | 2/3 | 63% | 92% |
| seir | 5 | 2.8pp | 3/5 | 48% | 88% |
| gbm | 2 | 1.6pp | 2/2 | 53% | 92% |
| cir | 3 | 7.5pp | 1/3 | 61% | 91% |
| double_well | 3 | 2.4pp | 1/3 | 48% | 89% |
| stoch_lv | 4 | 4.3pp | 1/4 | 45% | 86% |
| fhn | 4 | 1.9pp | 3/4 | 48% | 88% |
| sindy_sde | 5 | 1.3pp | 4/5 | 49% | 90% |
| **total** | | **3.7pp** | **17/29** | ~51% | ~90% |

**Honest read.** Despite a much better architecture and 4× the budget, the
aggregate (17/29, 3.7pp) is essentially unchanged from the concat 12k/40 canon
(16/29, 3.8pp). Per case it is mixed — gbm 2/2, fhn 3/4, ou 2/3 improved; cir,
double_well, stoch_lv sit at 1/3–1/4. The systematic failures (p≈0) cluster on:

- **strongly correlated** parameters — stoch-LV α/β, SEIR β₂/γ_d enter as products;
- **multimodal** ones — bistable double-well θ₂;
- **weakly identified** ones — CIR a/b, FHN I (recovery variable `w` unobserved).

Coverage remains usable everywhere (cov50 45–63%, cov90 86–92%); it is strict
rank-uniformity that fails. **Leading suspect:** the data-dependent base is a
*diagonal* Gaussian and cannot seed posterior **correlations**, so the flow must
build all correlation and multimodality from a factorized start. Next candidates:
a correlated / low-rank / mixture base, and more ODE steps for complex posteriors.

Reproduce: `uv run python examples/calib_gallery.py --n_train 50000 --epochs 60
--n_sims 500 --n_post 200 --out results/CALIB_GALLERY_XATTN.md`
