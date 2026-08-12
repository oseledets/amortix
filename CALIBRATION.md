# Calibration — the current state

Accuracy is half the claim. The other half is whether the posterior's *width and
shape* are right, which is what Simulation-Based Calibration (SBC, Talts et al.
2018) tests: if the amortized posterior is the true posterior, the rank of the
true parameter among L posterior draws must be uniform.

Reproduce:

```bash
uv run python examples/calib_gallery.py --n_train 40000 --steps 12000 \
    --n_sims 500 --n_post 200
```

## The board

Measured on the current defaults: raw observers everywhere (no hand-crafted
observer transforms) + the universal learnable warped-increment embedding
(`embed="auto"` → `wbasis` for every PathObserver/SDE case, plain linear for
ODE and custom observers).

| case | dim | embed | calib-err | SBC pass | cov50 | cov90 |
|---|---|---|---|---|---|---|
| linear_gaussian | 4 | linear | 0.9pp | **4/4** | 51% | 91% |
| ou | 2 | wbasis | 1.0pp | **2/2** | 52% | 90% |
| seir | 5 | linear | 1.6pp | **5/5** | 49% | 89% |
| gbm | 2 | wbasis | 1.2pp | **2/2** | 50% | 90% |
| cir | 3 | wbasis | 1.0pp | **3/3** | 51% | 90% |
| double_well | 3 | wbasis | 1.8pp | **3/3** | 49% | 89% |
| stoch_lv | 4 | wbasis | 1.7pp | 3/4 | 51% | 89% |
| fhn | 4 | linear | 1.4pp | **4/4** | 50% | 90% |
| sindy_sde | 5 | wbasis | 1.2pp | **5/5** | 49% | 90% |

**31/32 parameters pass**, mean calibration error **1.3pp**. The previous
boards read 29/32 with two hard failures (`gbm` sigma p=0.00, then after its
fix `sindy_sde` sigma p=0.00 and `stoch_lv` alpha p=0.00). All three are now
calibrated; the single remaining miss is `stoch_lv` beta at p=0.04 — exactly
at the threshold, alongside delta's history of hovering at 0.05.

Per-parameter p-values (p > 0.05 passes):

```
linear_gaussian: m1 0.33   m2 0.17   m3 0.70   m4 0.12
             ou: theta 0.69  sigma 0.60
           seir: beta1 0.18  beta2 0.36  alpha 0.50  gamma_r 0.95  gamma_d 0.07
            gbm: mu 0.41   sigma 0.30
            cir: a 0.80    b 0.33    sigma 0.96
    double_well: theta1 0.68  theta2 0.58  sigma 0.56
       stoch_lv: alpha 0.14  beta 0.04  delta 0.30  gamma 0.67
            fhn: a 0.99    b 0.80    eps 0.38  I 0.45
      sindy_sde: c0 0.47   c1 0.19   c2 0.37   c3 0.11   sigma 0.56
```

## One mechanism explained every hard failure: floating input scale

The campaign that produced this board started from a single symptom — `gbm`
sigma failing SBC (p=0.00) with a genuine +0.48 posterior-sd centre bias,
confirmed against the exact closed-form GBM posterior. The cause was not the
flow, the base, the solver, or the attention: it was the **conditioning of the
encoder's input**. Raw price tokens force the network to divide `dx²` by `x²`
to recover the quadratic-variation cue, and `x`'s scale floats
multiplicatively across the horizon and across datasets, so that division has
no fixed scale to normalize against. The same mechanism, undiagnosed, was
behind `sindy_sde` sigma (polynomial SDE, drifting state scale) and
`stoch_lv` alpha (positive populations with a wide dynamic range) — both
previously attributed to other causes, both fixed by the same embedding
change, no per-problem code touched.

## What fixed it: a universal learnable embedding, not feature engineering

The resolution is architectural and task-agnostic (`WarpDiffEmbed` in
`amortix/encoder.py`, default via `embed="auto"`): recompute each token's
increment *in* a learnably warped coordinate,

    dw = w(x + dx) − w(x),

where `w` is a fully learnable monotone warp (octave bank of signed-log1p
basis functions — mixture coefficients, scales, and linear part all trained;
monotone by construction, hence bijective, hence zero information loss).
Nothing in it knows about GBM, prices, or logs.

How the alternatives measured on raw-price GBM (signed sigma bias vs the
exact posterior, K=96, screen budget; production = 40k sims/12k steps,
SBC 500×200):

| embedding | learnable? | screen bias | production SBC sigma |
|---|---|---|---|
| raw tokens + linear | — | +0.483 | p=0.002 ✗ |
| multi-scale RFF of x | no (fixed) | +0.397 | — |
| Yeo-Johnson warp | λ per feature | +0.451 | — |
| RevIN instance-norm | affine | +1.23 (sr 4.07) | — |
| NALU log-arithmetic | yes | +0.058 | — |
| per-feature signed-log | scale | +0.024 | p=0.014 ✗ (residual grows with budget) |
| **wdiff** (warp-then-difference, single-scale) | scale | −0.010 | **p=0.876 ✓** |
| **wbasis** (fully learnable warp family) | everything | +0.033 | **p=0.283 ✓** |
| wconv (learnable taps, no hand square) | everything+ | +0.164 | p=0.130 ✓ |
| wpoint (point tokens only; attention learns all temporal structure) | everything++ | +0.058, sr 1.19 | p=0.074 ✓ |
| hand-crafted log-price observer (reference) | — | +0.000 | p=0.812 ✓ |

The last four rows form a complete structure-removal ladder, and it is
monotone: p = 0.876 → 0.283 → 0.130 → 0.074 as the hand differencing, the
hand square, and finally the increment window itself are handed over to
learning. Every rung passes; each removed prior is paid for out of the
optimizer's budget (wpoint at screen budget shows it as a 19%-too-wide
posterior — an information-extraction cost, not a bias). The one prior that
cannot be removed is the pointwise monotone warp: without it the same
architecture fails at p=0.002 no matter how much temporal machinery is
learnable.

Three transferable lessons:

1. **Where the increment is taken matters more than which nonlinearity is
   applied.** Warping `dx` separately from `x` (per-feature slog) leaves a
   residual that *grows* with training budget; differencing in the warped
   coordinate does not. `slog(dx) ≠ Δslog(x)`.
2. **The hypothesis class is not enough — the transform must be in the
   initialization basin.** Yeo-Johnson contains log one scalar away (λ=0) and
   gradient descent never moved λ off its identity init. The octave-bank warp
   works because log-like shapes are present at init at every scale
   simultaneously.
3. **Structure buys calibration margin monotonically** (p = 0.876 → 0.283 →
   0.130 as structure is removed), but every warp-then-difference variant
   passes — the load-bearing choice is the difference-in-warped-coordinate
   structure, the package analogue of convolution's locality prior.

RevIN is the cautionary tale on the other side: per-instance normalization
*erases the estimand* when the parameter of interest is itself the scale of
the increments (sr = 4.07, tercile bias ±5 sd). Universality of a module is
not a property of its genericity but of what it preserves.

The hand-crafted `LogPathObserver` is kept in `amortix/problems/gbm.py`
purely as the exact-sufficient-statistic reference to benchmark learned
representations against — it is no longer used by any default.

## Design amortization: p(m | any observation design)

A fixed observer bakes one experimental design into the model. The package
now also supports the honest general contract — the input is K bare
(t_i, x_i) pairs, K arbitrary — via three pieces: a problem may return
`(m, tokens, mask)` from `simulate()` (every dataset its own design),
`embed="wpair"` (consecutive differences in learnably warped coordinates of
both the value AND the time axis), and `rope="time"` (attention phases from
the physical time instead of the slot index).

Measured on GBM trained at K ~ log-uniform[2,128] random-time designs and
evaluated against the exact per-design posterior (B=96 per bucket, sigma
bias / width ratio):

| arm | K=5 | K=9 | K=20 | K=100 |
|---|---|---|---|---|
| bare points, ordinal RoPE, spread-position training (bug) | +1.09 / 0.94 | +1.90 / 1.13 | +3.30 / 1.42 | +1.59 / 2.08 |
| bare points, ordinal RoPE, contiguous | −0.01 / 1.05 | +0.01 / 1.17 | −0.10 / 1.13 | +0.03 / 1.73 |
| bare points, rope="time" | +0.02 / 1.01 | +0.02 / 1.10 | +0.05 / 1.14 | +0.15 / 1.95 |
| wpair, ordinal RoPE | +0.04 / 1.03 | +0.05 / 1.04 | +0.04 / 1.04 | +0.04 / 1.27 |
| **wpair + rope="time"** | +0.01 / **1.02** | +0.04 / **1.02** | +0.04 / **1.02** | +0.08 / **1.27** |

Two further fully-universal arms were built to try to kill the wpair stencil
(both permutation-invariant, no adjacency, no Markov assumption): a learnable
per-head additive time-kernel bias in the attention logits
(ALiBi/T5-relative generalized to continuous irregular time), and a
graph-transformer edge-value attention whose *messages* carry
gap-modulated warped increments on ALL pairs. Widths (sigma, same budget):

| bare-point arm | K=5 | K=9 | K=20 | K=100 |
|---|---|---|---|---|
| time-kernel logit bias | 1.03 | 1.14 | 1.31 | 2.51 |
| edge-value messages | 1.03 | 1.15 | 1.29 | 2.45 |

The failure anatomy was then pinned down by two independent probes, and it is
entirely prosaic. A ridge-regression forensics pass on the trained encoders
(pooled context → the exact sufficient statistics) showed the posterior-mean
information largely present in every model (R² 0.83–0.94 on E[sigma|D]) but
**the number of observations poorly encoded exactly where widths failed**:
R²(log N) = 0.985 (wpair) / 0.957 (ordinal) / 0.712 (time-kernel) / 0.498
(edge-value). Softmax attention is normalized — it computes means, and means
erase cardinality — while the posterior width is governed by N (the chi²
degrees of freedom). Adding a single design-metadata feature log K to the
bare-point arm confirmed the mechanism causally: widths 1.95→1.49 (K=100),
1.14→1.08, 1.10→1.05, bias unchanged ≈0. The model can also recover the
count BY ITSELF via register tokens (2 learned tokens appended to every set:
an ordinary token's softmax weight on a register scales as ~1/K, so the
normalization itself becomes the counter) — that fixes sparse/mid designs
completely (1.10→1.06, 1.14→1.06) but only partially at K=100 (1.95→1.71),
because the register signal's amplitude decays as 1/K exactly where count
matters most; explicit log K keeps the signal-to-noise flat across K.

**The class boundary, tested from both sides.** On a complex nonlinear ODE
with observation noise and a hidden state (FitzHugh-Nagumo, only noisy v
observed) the ordering REVERSES, as the likelihood structure demands: with
observation noise the observed series is not Markov and the likelihood
factorizes over single points given m, so bare points are the sufficient
token there. Measured (design-amortized, K ~ log-uniform[4,64], SBC):
bare points pass 4/4 on mixed designs while wpair drops eps (p=0.028) and an
extra parameter at dense K — the SDE-world dominance of wpair (widths 1.02 vs
1.1–2.5) is gone entirely. Both arms share dense-K failures on the
sharpest parameters (I, eps at K≥12–50) — arm-independent, the same
training-design-tail signature as GBM/OU, i.e. a data-distribution effect.
Selection rule, derived and now verified on both sides: **wpair iff the
observed process is Markov; bare points otherwise.**

**Class transfer.** wpair is derived from the Markov factorization of the
likelihood (consecutive-pair transition densities at sorted times), so it
must transfer across the whole Markov-diffusion class with no changes — and
it does: the identical module trained on design-amortized OU (additive noise,
mean reversion, stationary start; nothing multiplicative) against an MCMC
reference with exact per-gap transitions gives biases ≈ 0 and widths
0.96–1.01 at K=5..20 for both parameters (theta 1.000 even at K=100); the
only deviation is sigma at K=100 (1.34) — the same training-design-tail
signature as GBM's 1.27, shared across processes, i.e. a data-distribution
effect, not process specificity.

Lessons: (1) the huge "biases" of the first arm were NOT a capability limit —
they were a positional train/eval mismatch (ordinal RoPE turns the token
layout into a contract; masked subsampling trained on spread-out positions
while inference packs contiguously; `token_dropout="design"` now compacts,
and `rope="time"` removes the contract altogether — attention phases follow
the point, not its place, restoring true permutation invariance for pointwise
embeddings, verified to float noise). (2) The width gap of every bare-point
variant is mostly **cardinality blindness** of normalized attention, fixed by
letting the set model know its own input size (metadata, not process
knowledge). This also explains the once-mysterious advantage of ordinal RoPE
over the metric arms at dense designs (1.73 vs ~2.5): with contiguous
packing, the maximal position IS K — ordinal phases smuggle the count.
wpair's gap features encode local density, hence count, hence its clean
widths; its consecutive-pair stencil is an efficiency optimization at fixed
budget, not a necessity. (3) The residual gap after the count fix (1.49 vs
wpair's 1.27 at K=100) is consistent with the training design distribution
(log-uniform puts ~6% of mass at K≥100) and budget, not with any identified
architectural limit. (4) The exact per-design reference used for all these
numbers was independently validated against adaptive MCMC on the same sparse
designs: worst mean discrepancy 0.099 posterior-sd, width ratios 0.91–1.03
over 24 comparisons.

**The canonical training recipe (the dense-tail fix).** The residual
dense-design width traced to the training DESIGN distribution and was closed
in two causal steps, measured on GBM/OU/FHN. Step 1: reweighting K (50%
log-uniform + 50% uniform over the dense half) halves the excess on all three
systems (GBM 1.27→1.14, OU 1.34→1.14, FHN's dense-K SBC failures
0.000→0.54). Step 2 resolves the remaining fork: with FIXED designs (one per
dataset), a long budget closes the tail (1.05–1.06) but overfits the design
set into overconfidence at sparse K (sr down to 0.86); FRESH designs — every
optimizer step re-draws each trajectory's observation subset from the full
grid via `fit(retokenize=...)` — make design memorization impossible and give
the flattest profile of any recipe: **sigma sr 1.06/1.06/1.05/1.07 at
K=5/9/20/100** (12000 steps, mix K). One simulation thereby serves
unboundedly many designs, which is also exactly the training mode required
for downstream sequential refinement / optimal experimental design, where
the model must answer p(m | S) coherently for every subset S of one
measured series. An H200 port (fit/sample_batch take device="auto")
reproduces CPU runs bit-comparably and was used for the recipe grid.

## Second zoo: Henon-Heiles, Heston, Merton (design-amortized)

Three further cases on the canonical recipe (fresh designs + mix K + 12000
steps), SBC over mixed random designs plus fixed-K buckets:

* **Henon-Heiles** (classical Hamiltonian with the Lubich-Oseledets-
  Vandereycken potential, lambda=0.1118 at prior centre; recover
  (omega1, omega2, lambda) from noisy q1 at random times): 3/3 mixed, every
  K-bucket passes down to K=6 — sparse-design frequency aliasing does not
  break calibration; the posterior is honestly wide there instead.
* **Heston** (hidden stochastic volatility, correlated noises, price-only
  observations; 5 parameters): both bare-point and wpair arms pass
  **all 25 cells each** — the class rule's "hidden state removes wpair's
  edge" shows as parity, not degradation.
* **Merton jump-diffusion** (5 parameters): the autopsy (near-exact
  Poisson-mixture MCMC reference + encoder forensics) split the failures
  into two measured mechanisms. wpair's sigma failure (mixed p=0.000) was a
  **+1.12 posterior-sd systematic bias**: jumps make dw² heavy-tailed, the
  running FeatureNorm scale is set by outliers (sd 3.1e4 vs a diffusion
  range of ~0.01–1) and never converges, so the flow trains against a
  drifting representation — while the pooled context encodes the robust
  statistics BETTER than the healthy arm (R²=0.967 on truncated RV): the
  information was extracted, the failure was downstream. Bare points at
  dense K failed by **width** (sr≈2.0, bias ≈0) instead. The fix follows
  the mechanism and is the session's own medicine applied to the features:
  `PairEmbedRobust` log-compresses dw and dw² before FeatureNorm (jump
  outliers become additive; near-identity on clean diffusions since
  log1p(x)≈x for small x). Result: **25/25 SBC cells pass**, including the
  dense-design frontier both original arms failed (sig at K=42: 0.002 →
  0.959; K=112: 0.000 → 0.063). The "first irreducible boundary" reduced to
  the conditioning disease in its third guise: floating scale → cardinality
  → heavy tails. A precision follow-up (exact-reference probe; SBC at these
  sizes provably misses sr≈1.3) found a real residual in the log-compressed
  arm (+0.27 sd at K=112) whose mechanism is a FLOATING CLASSIFICATION
  THRESHOLD: the jump/no-jump boundary sits at |dw| ~ few×sigma·sqrt(tau)
  while sigma spans 5× across the prior. Two cures were raced on the same
  cell: hand per-dataset median scaling with an explicit scale channel
  (+0.094 ± 0.054, sr 1.33) versus a UNIVERSAL learned set-conditioning
  module (masked-pool context → per-channel FiLM after FeatureNorm,
  zero-init: +0.127 ± 0.055, sr 1.25) — statistically indistinguishable.
  The universal module needed three correctness conditions, each found the
  hard way: the right basin (zero-init = start at the working base), the
  right order (normalize BEFORE modulating, or the learned gain re-creates
  the representation drift it cures — measured: feature sd 1093 with the
  wrong order), and honest masks (a harness bug fed padded zero-slots into
  set-level statistics; caught because SBC and the exact probe contradicted
  each other — two disagreeing instruments are worth more than either).
  Session tally of hand-vs-learned duels — log-coordinate, input
  cardinality, robust scaling: the learned counterpart matched the hand fix
  in all three, always via the same pattern: the hand fix locates the
  mechanism and the basin; the learnable module placed there is what ships.
  The residual itself was then closed by a scaling ladder on the universal
  arm: solver refinement — no effect (rk4/60 ≡ midpoint/20); capacity 2×
  (128×4) — weak (sr 1.25→1.21); training budget 3× (60k sims / 36k steps)
  — decisive: **bias +0.007 ± 0.061 (zero), width 1.12, all 25 SBC cells
  pass**. The curve was then extended to three points and it is a clean power
  law: width excess 0.246 (×1 budget) → 0.120 (×3) → **0.059 (×6: 120k sims
  / 72k steps)** — excess ∝ 1/budget, bias zero throughout, SBC sig p=0.934
  at the once-dead cell. Capacity×budget (128×4 at ×3 budget: 1.103 vs
  1.120) confirmed the lever is budget, not weights. At ×6 the hardest
  statistic of the zoo extracts 89% of the available information,
  indistinguishable from the GBM/OU dense cells. Further sharpening is a
  price list (×2 budget per halving of the excess), not an open problem.

## Third zoo: literature and real-world cases (design-amortized)

Three cases chosen from the SBI/inverse-problems canon (sbibm; PDE inverse
problems), all on the canonical recipe:

* **Hodgkin-Huxley** (sbibm's flagship; stiff spiking 4D neuron dynamics,
  recover (gNa, gK, gL, I) from noisy voltage at random times): mixed SBC
  **4/4** (gL 0.98); one dense cell (gK @K=84) fails — the familiar
  budget signature.
* **Pharmacokinetics** (oral one-compartment Bateman curve, log-normal assay
  noise — the real-world archetype of irregular designs): mixed SBC 3/3, and
  against exact-likelihood MCMC at **K=6 blood draws** the posteriors are
  unbiased (−0.01..−0.06 sd) at widths 1.09–1.33 — calibrated clinical-regime
  inference in milliseconds. Dense designs show the usual sharpest-parameter
  (ka) budget tail.
* **Fisher-KPP reaction-diffusion** — the zoo's FIRST PDE, with
  spatio-temporal designs (random time × sensor pairs, sensor id in the
  spare token slot). At base budget it genuinely failed (r p=0.000 even on
  mixed designs); the autopsy against a multi-start-validated PDE-likelihood
  MCMC found a distinctive geometry: the posterior concentrates on the
  curved ridge D·r ≈ const (front speed c=2√(Dr) sharply identified,
  position q=ln(r/D) along the ridge weakly identified), and the flow's
  error points ALONG the ridge (q bias +0.41 sd vs c −0.21 at ×1 budget).
  The "new mechanism" hypothesis was then refuted by scaling: at ×3 budget
  both components shrink (q +0.41→+0.23, c −0.21→−0.14, widths 0.96–1.05) —
  the flat direction simply converges last (weakest CFM gradient signal),
  with a gentler exponent (~budget^-0.5) than Merton's width (~1/budget).
  The curve was completed to three points at ×6 (120k/72k): q bias
  +0.41 → +0.23 → **+0.14**, c −0.21 → −0.14 → **−0.075**, widths
  0.93–1.02 — smooth ~budget^-0.5 decay, no wall. At these residuals SBC
  (n=300) sits in its half-power zone and its per-cell p-values flicker
  between retrains — the exact-reference probe, not the SBC threshold, is
  the instrument of record there. Every defect found in three zoos now lies
  on a measured budget curve or has a learnable architectural cure.

## Where all of this lives in the package

Everything measured above ships in `amortix` itself, not in experiment
scripts: `amortix.designs` (DesignProblem/DesignObserver, the canonical
fresh-design retokenizer with the mix-K law, `sbc_design`);
`amortix.problems.design_zoo` (Heston, Merton, Henon-Heiles,
Hodgkin-Huxley, pharmacokinetics, Fisher-KPP — plus the exact-likelihood
factories `merton_logpost_factory` / `pk_logpost_factory` /
`kpp_logpost_factory` for reference probes); embeddings `wpoint` /
`wfilm` (`PointEmbed`, `SetCondPairEmbed`) next to `wdiff`/`wbasis`/`wpair`
in `amortix.encoder`; and `FlowPosterior`'s `embed="auto"` / `rope="auto"`
now resolve DesignProblem cases by the verified class rule (Markov-observed
→ set-conditioned pairs, otherwise bare points; continuous time-RoPE
always). Runner: `examples/design_zoo_run.py --case pk`.

## What a passing SBC does and does not mean

Passing is necessary, not sufficient, and one reading is easy to get wrong: **a
posterior that simply returns the prior also passes SBC**, and also passes any
`cov90 in [80,97]%` check. That is why this repo never reports calibration alone —
see the prior-only and ridge controls in [`GALLERY_RESULTS.md`](GALLERY_RESULTS.md).

Conversely, a parameter can be flagged as poorly recovered by an accuracy metric
and still be perfectly calibrated: the posterior is *correctly* wide when the
parameter is weakly identified, and a point estimator merely happens to land
closer.

## How this board was earned

The path here, in order: a velocity field factorized across parameters
(correlations impossible by construction) — fixed with parameter
self-attention; a base head cannibalizing the CFM loss — fixed with a detach;
a degenerate time embedding — fixed with a scale factor; an anti-conservative
SBC chi-square test (10.7% false rejects at nominal 5%) — fixed with exact
multinomial bin probabilities; two initial-condition parameter leaks (OU, CIR)
— fixed with stationary-law sampling; and finally the floating-input-scale
mechanism above — fixed with the universal learnable embedding. Each fix was
validated against exact posteriors or MCMC references, never against the loss
curve. The numbers above are not comparable to any earlier board of this
repository, only to the truth.
