# CRITIC — audit of the problem definitions

Adversarial review of `amortix/problems/*.py` + `amortix/sde.py`, `amortix/ode.py`,
`amortix/prior.py`. Everything below is measured, not argued; the scripts are
described inline so each number can be reproduced.

**Context.** Two exact initial-condition leaks (`ou.mu`, `cir.b`, `corr(X0, ·) = 1.000`)
were just fixed. This pass looks for what else is like that. It found one more leak-class
problem (§0 — the *screen itself* is wrong), two confounded comparisons that inflate every
headline number, and three cases where a large share of the prior box carries no information
about the parameters being scored.

**Universal yardstick used throughout.** For `m ~ U(low, high)`, the estimator that ignores
the data entirely (predict the midpoint) has MAE **exactly 25.00 % of the prior range**, and
the prior sd is **28.87 % of the range**. Any reported error near 25 % means *no information
was extracted*; any posterior sd near 28.87 % means *the posterior is the prior*. Both
appear repeatedly below.

---

## Summary table

| # | finding | status | impact |
|---|---|---|---|
| 1 | `sota()` is fed 5–10× more data than the network; every gallery win/loss is confounded | CONFIRMED | critical |
| 2 | OU "exact MLE" has a **+31 % bias in θ**; it is neither exact nor near-optimal | CONFIRMED | critical |
| 3 | `sindy_sde` c2, c3 are **structurally unidentifiable over the whole prior box** | CONFIRMED | critical |
| 4 | SEIRD: ~40 % of the box has no observable epidemic; α unidentified everywhere | CONFIRMED | high |
| 5 | FHN: **62 % of the prior box is not oscillatory**, contradicting the docstring | CONFIRMED | high |
| 6 | `stoch_lv` observer sees only **40 % of the horizon** (0.43 LV periods); `sota` sees 100 % | CONFIRMED | high |
| 7 | The IC-leak screen is wrong: residual `corr(X0, μ) = 0.69`, `corr(X0, b) = 0.87` | CONFIRMED | high (process) |
| 8 | README still quotes the leaked OU numbers, provably below the information floor | CONFIRMED | high |
| 9 | The fast channel always starts at t = 0 and spans 8–12 % of the horizon | CONFIRMED | medium |
| 10 | Token features are unnormalized, spanning 4 orders of magnitude; 1 feature is dead | CONFIRMED | medium |
| 11 | NLS baselines **are** converged (clean bill) — except FHN, local minimum in 13 % | CONFIRMED | medium |
| 12 | `dt` coarse enough to shift the classical estimates by 1–2.5 pp of prior range | CONFIRMED | medium |
| 15 | linear_gaussian's "fixed, seeded" design matrix changes with the ambient default dtype | CONFIRMED | medium |
| 13 | GBM μ is weakly identifiable by construction; the per-case average hides it | CONFIRMED | low |
| 14 | Two latent observer traps: over-running channels, and `n_paths>1` starving `sota` | CONFIRMED | low (latent) |
| 16 | CIR Feller / positivity: **not** a problem (0.00 % of the box) — docstring overstates the risk | CONFIRMED | none |
| 17 | RK4 with the shipped `dt`: fine for both ODE cases | CONFIRMED | none |

Two claims I set out to test and could **not** substantiate, recorded so they are not
re-tested: the NLS baselines are properly converged (§11), and CIR never meaningfully
violates Feller or hits its positivity clamp (§16). Neither is a defect.

---

## 0. The method that found most of this: an exact information floor

For OU the shipped simulator is Euler–Maruyama, which for a linear drift is an **exact
AR(1)**:

```
X_{k+1} = mu + r (X_k - mu) + sigma sqrt(dt) Z,      r = 1 - theta*dt
X_{k+g} | X_k ~ N( mu + r^g (X_k - mu),  sigma^2 dt (1 - r^{2g}) / (1 - r^2) )
X_0 ~ N(mu, sigma^2 / (2 theta))                     (what x0_sampler draws)
```

So the likelihood of *exactly the observed token times* is available in closed form, and the
**exact Bayes posterior** can be computed by prior importance sampling (60 000 prior draws,
median ESS 1824). That is the true optimum for the data the network sees. Over 300 held-out
datasets:

| estimator | θ | μ | σ | mean |
|---|---:|---:|---:|---:|
| **EXACT Bayes posterior mean (73 observed times)** | **14.70 %** | **7.29 %** | **4.02 %** | **8.67 %** |
| `baselines.ou_mle` (full 500-point path) — the shipped "SOTA" | 22.08 % | 10.66 % | 2.06 % | 11.60 % |
| prior mean (zero information) | 25.00 % | 25.00 % | 25.00 % | 25.00 % |
| exact posterior **sd** | 18.90 % | 9.66 % | 5.17 % | — |

(MAE as % of prior range.) This one table is worth keeping as a permanent regression test:
**any reported amortized error below the exact-Bayes row is proof of a leak**, and the
"SOTA" row can now be seen for what it is. It also cross-validates the Fisher machinery used
below — the Fisher CRB for θ is 18.16 %, against the exact posterior sd 18.90 %.

---

## 1. CONFIRMED — `sota()` and the network are fed different data (critical)

**What is wrong.** Every SDE `sota()` consumes `traj`, the *full fine path*
(`n_steps + 1` points), while the network sees only the observer's tokens:

| case | fine increments given to `sota` | tokens given to the network | directly observed (`every=1`) increments |
|---|---:|---:|---:|
| ou | 500 | 74 | 50 |
| gbm | 500 | 100 | 60 |
| cir | 500 | 74 | 50 |
| double_well | 1000 | 116 | 80 |
| sindy_sde | 1000 | 125 | 80 |
| stoch_lv | 600 × 2 comp | 180 | 50 × 2 |

**Evidence.**
- OU (§0): the *optimal* estimator on the tokens gets σ = 4.02 %; the shipped baseline gets
  2.06 % from 10× more increments — so the baseline's σ advantage is a pure data-budget
  artifact. `GALLERY_RESULTS.md` already half-says this ("data-budget artifact, not a method
  limit") — but it is stated for σ only, while the *same asymmetry runs in the opposite
  direction* for the drift parameters and is not mentioned there.
- stoch_lv, direct test (§11): re-running the *same* NLS baseline on only the 83 grid points
  the observer emits moves it from **8.67 % to 12.63 %** — 46 % of its winning margin is the
  extra data, and the matched-data figure agrees with the Fisher CRB (12.40 %) and the
  summary-statistic ridge (12.6 %).
- stoch_lv, Fisher CRB (posterior sd, % of prior range, 64 draws):

  | data | α | β | δ | γ |
  |---|---:|---:|---:|---:|
  | full 600-step path (what `sota` fits) | 7.01 | 3.68 | 3.89 | 6.64 |
  | steps covered by tokens (optimistic bound for the net) | 12.97 | 13.75 | 7.92 | 14.94 |

  The baseline is handed 2–4× tighter information. `GALLERY_RESULTS.md` reports this exact
  case as "SOTA wins, 8.1 % vs 16.3 %" and attributes it to "low-noise regimes favor
  deterministic fitting". The measured cause is the data budget.

**Why it distorts the results.** Both directions of every gallery verdict are confounded.
"amortized wins" can be prior shrinkage against a data-rich but prior-free estimator;
"SOTA wins" can be the estimator simply having more data. The `win` column in
`examples/gallery.py:96` is not measuring what the table claims.

**Patch.**
1. Make `sota(tokens, traj, prob)` consume `tokens` (it already receives them), or
2. keep `traj` but add a third column to the gallery: the **information floor** (exact Bayes
   where available — OU, linear_gaussian — otherwise the Fisher CRB of §3/§4), and report
   `amort / floor` rather than `amort vs sota`. A ratio to the floor is budget-invariant.
3. In `run_case`, record and print the data budget of each estimator so the asymmetry is
   visible in the artefact.

---

## 2. CONFIRMED — the OU "exact MLE" is biased by +31 % in θ (critical)

**What is wrong.** `baselines.ou_mle` is the conditional AR(1) MLE. At `n = 500`,
`dt = 0.02`, `rho = exp(-theta dt) ≈ 0.94…0.99`, the OLS slope has the classical
small-sample (Hurwicz) bias `E[rho_hat - rho] ≈ -(1 + 3 rho)/n`, which `theta = -log(rho)/dt`
amplifies by `1/(rho dt)`. Predicted upward bias ≈ `(1+3rho)/(n dt) ≈ 0.4`.

**Evidence** (600 datasets, `E[hat - true]`):

| paths | θ bias | σ bias | MAE θ | MAE μ | MAE σ | mean |
|---|---:|---:|---:|---:|---:|---:|
| Euler paths (as shipped) | **+0.5068 (+31.1 %)** | +0.0170 (+2.0 %) | 24.06 % | 9.21 % | 2.05 % | 11.77 % |
| exact-transition paths | +0.4736 (+29.1 %) | +0.0026 (+0.3 %) | 23.41 % | 9.21 % | 1.69 % | 11.44 % |

Two separate defects are visible:
- **the θ bias is an estimator defect** (+0.47 survives on exact-transition data), and it
  pushes the baseline's θ error (22–24 %) to within a whisker of the 25 % *zero-information*
  level;
- **the σ bias is a model-mismatch defect**: `ou_mle` assumes the exact OU transition
  (`rho = exp(-θ dt)`) while the simulator produces Euler paths (`r = 1 - θ dt`). On exact
  paths the σ bias drops from +0.0170 to +0.0026. The algebraic prediction for the mismatch,
  `σ√(2 θ̂ / (θ (2 - θ dt))) - σ`, is +0.0144 — it accounts for essentially all of it. So the
  "exact MLE" is *not exact for the data amortix generates*.

**Why it distorts the results.** `README.md` reports "θ: amortized 16.6 % vs exact MLE
24.5 %" as evidence that the amortized posterior is "competitive with the exact MLE (better
overall)". The exact Bayes posterior mean gets **14.70 %** on strictly less data. The
baseline is not near-optimal; beating it is not evidence of anything.

**Patch.** In `amortix/baselines.py:67`:
- bias-correct θ̂ (Hurwicz/Kendall correction, or a parametric bootstrap — 200 resimulations
  per dataset is cheap), **and**
- either simulate OU with its exact transition (available in closed form, ~5 lines in
  `ou.simulate_paths`) or use the Euler pseudo-MLE (`rho = 1 - θ dt`) so the estimator matches
  the generator. Simulating exactly is the better fix: it also removes §12 for this case.
- Rename `SOTA_NAME` from `"exact MLE"` to `"conditional AR(1) MLE"` until it is corrected,
  and add the exact-Bayes estimator of §0 as the real reference.

---

## 3. CONFIRMED — `sindy_sde` c2 and c3 cannot be recovered at all (critical)

**What is wrong.** The prior pins `c1 ∈ [-2, 0]` and `c3 ∈ [-1, -0.1]`, so the path is
strongly confined near X = 0 and the monomials `{1, X, X², X³}` are nearly collinear over the
visited range.

**Evidence.**
- Expected Fisher information from the **full 1000-step fine path** (i.e. more data than any
  consumer gets), 64 prior draws, posterior sd in % of prior range (prior sd = 28.87 %):

  | | c0 | c1 | c2 | c3 | σ |
  |---|---:|---:|---:|---:|---:|
  | posterior sd | 17.13 | 19.94 | **27.88** | **28.40** | 2.68 |
  | info gain (prior sd / post sd) | 1.69× | 1.45× | **1.04×** | **1.02×** | 10.8× |

  Fisher condition number 2.9 × 10⁴; the softest eigendirection is `(c0 0.04, c1 0.11,
  c2 0.44, c3 0.82, σ 0.00)`.
- Not a regime effect — conditioning on how far the path travels does not rescue them:

  | regime | c0 | c1 | c2 | c3 | σ |
  |---|---:|---:|---:|---:|---:|
  | wide, max\|X\| > 1.5 (n=24) | 19.31 | 18.42 | 25.73 | 23.54 | 2.80 |
  | medium, 0.8 < max\|X\| ≤ 1.5 (n=166) | 17.84 | 19.78 | 27.63 | 28.24 | 2.65 |
  | narrow, max\|X\| ≤ 0.8 (n=66) | 12.55 | 20.78 | 28.52 | 28.79 | 1.74 |

  48.5 % of paths never reach |X| > 1; the median ratio `|c3 X³| / |c1 X|` at the path
  extreme is 0.51.
- Independent confirmation from summary statistics: a degree-2 ridge on 22 hand-made path
  statistics gets out-of-sample R² = **0.023** (c2) and **0.033** (c3), MAE 24.2 % / 23.9 % —
  the 25 % zero-information level.

**Why it distorts the results.** `GALLERY_RESULTS.md` and `HANDOFF.md` both lead with
"SINDy-SDE: 18.0 % vs 32.4 %, ~1.8× better — most strikingly nonparametric drift discovery".
Two of the five averaged parameters are ones where **neither** estimator can do better than
the prior, and the classical baseline's 32.4 % is *worse than the 25 % you get by ignoring
the data entirely* (it is only that good because `sota` clips into the prior box at
`sindy_sde.py:116`). The flagship claim of the package is an average over an unidentifiable
subspace against a worse-than-nothing competitor.

**Patch (pick one, all are small):**
- **Orthogonalize the library.** Replace `{1, X, X², X³}` with polynomials orthonormal under
  the (approximate) stationary measure. The coefficients then have a well-conditioned Fisher
  matrix and the case becomes a real test of drift discovery.
- **Or open the box:** raise `sigma` to `[0.6, 1.5]` and/or set `c1 ∈ [-0.5, 0]` so paths
  actually explore |X| ≳ 2 where X³ separates from X.
- Either way, report per-coefficient errors together with the CRB, and drop the "1.8× better"
  headline until the comparison is against a baseline that beats the prior mean.

---

## 4. CONFIRMED — SEIRD: ~40 % of the box has no observable epidemic, α is unidentified (high)

**What is wrong.** `noise_std = 0.004` is *additive* and applied identically to I(t) and
D(t), but D(t) is an order of magnitude smaller than I(t); and the observation grid is
uniform over 60 days while the epidemic's informative phase is exponential.

**Evidence** (3000 prior draws unless stated).
- **38.2 %** of draws have peak I below 3 × noise_std; **47.6 %** have final D below
  3 × noise_std. Median observed-channel SNR (sd of clean signal / noise sd): I = 1.63,
  D = 1.17; the D channel has SNR < 1 in **46.7 %** of draws.
- Of the 40 observation tokens, **11 have an across-dataset clean sd below `noise_std`** —
  4 of the 20 I-observations and 7 of the 20 D-observations. 27.5 % of the observation budget
  is pure noise, by construction, in every dataset.
- Fisher CRB (posterior sd, % of range; prior 28.87 %):

  | regime | β1 | β2 | α | γ_r | γ_d |
  |---|---:|---:|---:|---:|---:|
  | big epidemic, peak I > 20·noise (n=46) | 10.36 | 8.44 | **25.86** | 3.75 | 1.20 |
  | moderate, 3–20·noise (n=111) | 14.57 | 15.76 | **26.93** | 21.15 | 6.08 |
  | no epidemic, peak I < 3·noise (n=99) | 17.62 | 20.17 | **28.36** | 23.14 | 21.61 |

  Overall α: 27.33 %, info gain **1.06×** — unidentified *even in the largest epidemics*.
  Fisher condition number 5.9 × 10⁵; the softest direction is consistently
  `+0.48 α + 0.46 γ_r + 0.26 β2 + 0.14 β1` (‖mean direction‖ = 0.72 across draws) — the
  classic latent-period / removal-rate trade-off.
- Trivial-stats ridge: α R² = 0.095, MAE 23.2 %.

**Why it distorts the results.** `GALLERY_RESULTS.md`: "SEIRD, amort 21.7 % vs NLS 24.6 %,
amort wins". Both numbers sit at the 25 % no-information level. That row records two
estimators returning the prior, and is currently cited in `HANDOFF.md` as evidence that
"the method wins clearly where the target is drift / structure".

**Patch.**
- Make the noise proportional (`noise_std × (value + floor)`) or give I and D separate noise
  levels (D ≈ 4 × 10⁻⁴), so the D channel carries its ~10× smaller signal at comparable SNR.
- Place `obs_steps` on a log/geometric grid over days 1–60 so the exponential phase is
  resolved instead of spending 11 of 40 tokens on flat pre-epidemic values.
- Condition the prior on `R0 = beta1/(gamma_r+gamma_d) > 1.5`, or raise `I0` from 1e-3, so
  the simulated epidemic is observable across the box.
- Drop α from the recovered set, or accept it and report it against its CRB.

---

## 5. CONFIRMED — FitzHugh–Nagumo: 62 % of the prior box is not oscillatory (high)

**What is wrong.** The docstring states "For parameters in the chosen prior box the system
sits in its relaxation / oscillatory regime, producing a train of spikes". With `I ∈ [0, 0.5]`
most of the box is *excitable*, not oscillatory: after the initial transient the trajectory
sits at a fixed point and the only information left is the scalar v*, which cannot separate
(a, b, I).

**Evidence** (3000 draws).
- **61.6 %** of draws produce **zero spikes** after t = 10 (v never crosses 1 upward).
  22.5 % have a post-transient v range below 4 × noise_std.
- Fisher CRB by regime (posterior sd, % of range; prior 28.87 %):

  | regime | a | b | eps | I |
  |---|---:|---:|---:|---:|
  | oscillatory, ≥ 2 spikes after t=10 (n=38) | 6.56 | 4.61 | 1.00 | 7.06 |
  | 1 spike (n=53) | 12.07 | 8.06 | 0.72 | 12.32 |
  | **excitable, no spike (n=165)** | **18.58** | **23.94** | 3.95 | **16.72** |

  `b` in the dominant regime has an info gain of 1.21× — essentially the prior.
- On a 5× longer solve (T = 200), **59.5 %** of draws have no spike at all in t ∈ [100, 200]
  — a genuine stable fixed point, not a slow transient. Among the true oscillators the median
  period is **25 time units**, so the shipped horizon T = 40 shows ~1.6 cycles, not "a train
  of spikes". Even the most excitable corner (a = 0.7, b = 0.8, eps = 0.15, **I = 0.5**, the
  top of the prior) produces exactly **1** spike after t = 10.
- `obs_steps[0] = 0` and `x0` is the constant `[-1, 1]`, so the first observation has an
  across-dataset clean sd of **0.000**: 1 of only 25 observations is a constant plus noise.
- **The 25-point grid aliases the fast variable badly.** The largest jump between consecutive
  *observed* samples is a median **79 %** of the full v range (p90 = 100 %), and exceeds half
  the range in **80.8 %** of datasets — against 6.3 % per simulation step. The sampled spike
  height is a function of phase, not of the parameters.

**Why it distorts the results.** The reported FHN row (amort 13.4 % vs NLS 19.7 %, "amort
wins") averages over a box that is two qualitatively different problems, and `HANDOFF.md`
lists "FHN I" among weakly-identified parameters without noting that this is a property of
*the box*, not of the method.

**Patch.** Oscillation is not controlled by `I` alone (oscillatory draws span I = 0.006…0.5),
so pick the regime by *measurement*, not by intuition: extend the horizon to
`n_steps = 4000` (T = 200, ≈ 8 periods) and/or filter the prior with a cheap spike test, then
report the oscillatory and excitable boxes as separate cases. Independently: drop the t = 0
observation (`torch.linspace(1, n_steps, 25)`), and raise the observation count to ≥ 100 so
consecutive samples differ by well under the v range.

---

## 6. CONFIRMED — `stoch_lv`: the observer sees 40 % of the horizon (high)

**What is wrong.** `Channel(every=6, count=40)` reaches grid index 240 of 600. Tokens cover
`t ∈ [0, 2.40]` of a horizon of 6.0 — **40 %** — while `sota` fits `traj[0::6]` over all 600
steps. 60 % of every simulation is computed and thrown away.

**Evidence.**
- Horizon coverage across the gallery (last observed step / `n_steps`): ou 96 %, cir 96 %,
  double_well 90 %, sindy_sde 90 %, gbm 80 %, **stoch_lv 40 %**.
- Median linearized LV period `2π/√(αγ)` = 5.57, so the tokens cover **0.43 periods** while
  `sota` fits 1.08 periods. The docstring claims "horizon 6 → a few predator-prey
  oscillations"; the *observer* never sees one.
- Fisher CRB full-path vs token-covered: α 7.01 → 12.97, β 3.68 → 13.75, δ 3.89 → 7.92,
  γ 6.64 → 14.94 (% of range).
- The diffusion (`s1 = s2 = 0.05`) is **known and not recovered**, so the fast channel's
  stated purpose (quadratic variation → diffusion) does not apply here. Channel-ablation
  ridge R² — slow-only vs both: α 0.629 / 0.624, β 0.682 / 0.684, δ 0.810 / 0.829,
  γ 0.556 / 0.599. The 100 fast-channel tokens (56 % of the token budget) add ~0.02 R².

**Patch.** `Channel(every=15, count=40)` covers all 600 steps at the same token cost; drop
the fast channel (or cut it to `count=10`) and spend the budget on the slow one. Then re-run
the gallery — this is the case the table currently reports as the clearest classical win.

---

## 7. CONFIRMED — the initial-condition leak screen is wrong (high, process)

**What is wrong.** The stated check — "corr(x0, parameter) across all cases, only ou/cir
exceeded 0.3, now handled" — does not hold after the fixes.

**Evidence** (4000 draws, `max_j |corr(X0_j, m_i)|`):

```
          ou: theta=0.006   mu=0.687   sigma=0.008
         cir: a=0.001       b=0.866    sigma=0.004
         gbm: 0.000 (deterministic S0)      double_well: 0.000
   sindy_sde: 0.000                            stoch_lv: 0.000
```

`corr(X0, μ) = 0.687` and `corr(X0, b) = 0.866` — well above the 0.3 threshold that was
believed to be satisfied.

**Is it a defect?** No — and that is the point. Under a *stationary* start `E[X0] = μ` (OU)
and `E[X0] = b` (CIR), so a high correlation is the statistically correct behaviour, and both
`sota` implementations use the path level too. **The threshold is unachievable for any
stationary-start SDE**, so the screen as stated can never be passed and gives false comfort
when it appears to be.

**Patch.** Replace the `corr(x0, param) < 0.3` screen with the **information-floor check** of
§0: compute (exactly where the likelihood is tractable, otherwise the Fisher CRB) the best
achievable error, and assert that no estimator reports below it. That is the criterion that
actually detects a leak, and it caught the pre-fix μ = 1.4 % immediately (see §8).

---

## 8. CONFIRMED — README still publishes the leaked OU numbers (high)

`README.md` (the "Result on Ornstein–Uhlenbeck" table) reports μ = **1.4 %** and an overall
7.8 % for the amortized posterior, and concludes "competitive with the exact MLE (better
overall)". The exact Bayes posterior mean on the same token set achieves μ = **7.29 %** (§0).
1.4 % is **5.2× below the information-theoretic optimum** — arithmetically impossible without
the leak that was just fixed. The table has not been regenerated.

**Patch.** Regenerate the README table and the OU rows of `GALLERY_RESULTS.md` post-fix, and
add the exact-Bayes column from §0 so the impossibility is visible next time.

---

## 9. CONFIRMED — the fast channel always starts at t = 0 and spans 8–12 % of the horizon (medium)

**What is wrong.** `PathObserver.tokens_from_traj` builds indices as
`arange(0, count+1) * every` — *every* channel starts at grid index 0. The fast channel is
therefore always the same short prefix of the trajectory.

**Evidence.**

| case | fast window | as % of horizon | grid points duplicated between the two channels |
|---|---|---:|---:|
| ou | t ≤ 1.00 of 10.0 | 10 % | 3 |
| gbm | t ≤ 0.60 of 5.0 | 12 % | 7 |
| cir | t ≤ 1.00 of 10.0 | 10 % | 3 |
| double_well | t ≤ 0.80 of 10.0 | 8 % | 4 |
| sindy_sde | t ≤ 0.80 of 10.0 | 8 % | 5 |
| stoch_lv | t ≤ 0.50 of 6.0 | 8 % | 9 |

For `double_well` and `sindy_sde`, `x0_sampler` returns exactly 0, so the fast channel always
observes the same escape from the barrier over a narrow range of X — precisely the range where
X² and X³ are indistinguishable (cf. §3). Channel-ablation ridge R², fast-only:
double_well θ2 = 0.063, sindy c1 = 0.045, c2 = −0.010, ou θ = 0.062, cir a = 0.046.

**Patch.** Draw the fast window's start index uniformly per dataset (and per replicate path):
`start = randint(0, n_steps - count*every)`. Cost: 3 lines. Benefit: the diffusion cue is
sampled at a random state (essential for the state-dependent diffusions of CIR/GBM/LV), the
drift gets a second look at a different X, and the channel duplication disappears.

---

## 10. CONFIRMED — token features are unnormalized and one is dead (medium)

**What is wrong.** `PathObserver` emits `[t/T, x, dx, dx², log10 dt_obs, cid]` and
`SetTransformer.embed = nn.Linear(n_features, dim)` ingests them raw — no standardization
anywhere in `sde.py`, `encoder.py` or `flow.py`.

**Evidence** — per-channel feature sd (400 datasets):

| case | fast `dx²` | slow `x` | ratio |
|---|---:|---:|---:|
| ou | 0.0337 | 0.842 | 25× |
| cir | 0.00391 | 0.397 | 100× |
| gbm | 0.00358 | 0.955 | 267× |
| sindy_sde | 0.00528 | 0.510 | 97× |
| stoch_lv (dim 1) | 5.08e-05 | 0.889 | **17 500×** |

The quadratic-variation cue — the *entire reason* the fast channel exists — enters the shared
linear embedding at the smallest magnitude of any feature. Additionally:
- `cid` is identically 0 in the 5 cases with `obs_dims=(0,)` — a dead input dimension;
- `dx²` is algebraically `dx**2`, so 1 of the 6 features is redundant to a nonlinear encoder;
- `dx` and `dx²` differ between the two channels purely by `dt_obs` (×√20 and ×20 for OU), so
  the network must learn to undo the `dt` scaling from the `res` feature.

**Patch.** Emit scale-free increments: replace `dx` with `dx / sqrt(dt_obs)` and `dx²` with
`log(dx²/dt_obs + eps)`. Both are O(σ) and O(log σ²) on every channel, and the `res` feature
still identifies the resolution. Alternatively, standardize the token matrix using statistics
accumulated during `simulate(n_train)`. Drop `cid` when `len(obs_dims) == 1`.

---

## 12. CONFIRMED — `dt` is coarse enough to move the classical estimates by 1–2.5 pp (medium)

Paired comparison with **coupled Brownian increments** (the fine path's increments are summed
in blocks of 8 to drive the coarse path), 64 draws. Mean shift of `sota(dt) − sota(dt/8)`, in
% of prior range:

| case | shifts |
|---|---|
| double_well | θ1 **+2.45**, θ2 **+2.40**, σ +1.07 |
| cir | a +0.85, b +0.05, σ +0.99 |
| ou | θ +0.90, μ +0.01, σ +0.94 |
| sindy_sde | c0 +0.02, c1 +0.08, c2 −0.37, c3 −0.41, σ +0.70 |
| stoch_lv | α −0.51, β −0.55, δ +0.04, γ +0.61 |
| gbm | μ −0.01, σ +0.14 |

Strong error `|X(dt) − X(dt/8)| / sd(X)`: median 0.004–0.015, p95 up to 0.083 (gbm).

**Why it matters.** The double-well shift (2.4 pp) is 40 % of the gap the gallery uses to
declare a winner there (14.6 % vs 20.7 %), and it means the benchmark scores recovery of the
*Euler-discretized* parameters, not of the SDE's. **Patch:** halve `dt` for `double_well`
(0.005) and `ou`/`cir` (0.01) — or, better, state explicitly that the target is the Euler
model and keep the estimators consistent with it (see §2).

---

## 11. CONFIRMED (mostly a clean bill) — the NLS baselines are converged (medium)

I suspected the shipped `least_squares(..., max_nfev=200)` calls were stopping early, which
would make every "amortized wins" against them meaningless. **They are not.** A fit must be
able to reach a cost at least as low as the cost at the true parameters (the truth is a
feasible point), so `cost(estimate) > cost(truth)` is a convergence failure:

| case | stopped above the cost at the truth | median cost ratio ship/true | `max_nfev=3000, xtol=ftol=1e-12` improves |
|---|---:|---:|---:|
| seir | 0/30 | 0.942 | 0/30 (identical estimates) |
| fhn | **4/30** | 0.906 | 0/30 (identical estimates) |
| stoch_lv | 0/30 | 0.250 | 0/30 (identical estimates) |

So the NLS baselines are fair, with one caveat: **FHN falls into a local minimum in 13 % of
datasets**, which inflates its reported error somewhat. Raising `max_nfev` does nothing;
only multi-start would. **Patch:** add 3 random restarts to `fhn.sota` and keep the best cost.

The same run supplies the cleanest single proof of §1 and §6:

```
stoch_lv, 30 datasets, MAE % of prior range
  shipped sota (fits all 600 steps, t <= 6.0)      alpha 8.64  beta 7.23  delta 6.62  gamma 12.20 | 8.67
  same NLS on the 83 grid points the observer emits (t <= 2.4)
                                                   alpha 12.63 beta 12.35 delta 9.65  gamma 15.91 | 12.63
```

Three independent routes agree on ~12.4–12.6 % as the floor for the token budget: the
matched-data NLS (12.63 %), the Fisher CRB (12.40 %), and the summary-statistic ridge
(12.6 %). The shipped baseline's 8.67 % is bought with data the network is not given, so
**46 % of the "deterministic NLS wins" gap is data budget, not method.**

Also worth recording: SEIRD's NLS baseline is *converged* and still returns
**α = 49.32 %** MAE and a case mean of **25.43 %** — i.e. converged nonlinear least squares
on this problem is worse than predicting the prior mean, and on α it is twice as bad as
guessing. That is a property of the case (§4), not of the optimizer.

---

## 13. CONFIRMED — GBM μ is weakly identifiable by construction (low)

Fisher CRB for μ: 20.79 % (full path) / 21.86 % (tokens) against a prior sd of 28.87 % — an
info gain of only 1.32–1.39×. By volatility: μ CRB 10.02 % (σ < 0.2), 17.78 % (0.2–0.4),
22.98 % (σ ≥ 0.4), matching the textbook `sd(μ̂) = σ/√T ∈ [7.5 %, 44.7 %]` of the μ range for
T = 5. The docstring says this; the *table* does not — it reports one averaged number
(11.7 %) that is `(μ ≈ 21 + σ ≈ 2.5)/2`. **Patch:** report per-parameter with the CRB
alongside, or shorten the σ range / lengthen T if μ is meant to be recoverable.

---

## 14. CONFIRMED — two silent traps in the observer machinery (low, latent)

**(a) over-running channels fabricate zero increments.** `sde.py:109-110`:
`idx = (arange(0, count+1) * every).clamp(max=self.n_steps)`. Any channel with
`count * every > n_steps` produces duplicated end indices, hence tokens with `dx = 0` and
`dx² = 0` — which the encoder reads as "the diffusion is zero here". No shipped case triggers
it (max coverage 96 %), but `ADDING_CASES.md` invites new cases and this fails silently.
**Patch:** `assert c.count * c.every <= n_steps` in `PathObserver.__init__`.

**(b) `n_paths > 1` silently starves the baseline.** `SDEProblem._tokens_for` (`sde.py:159`)
concatenates the tokens of all `n_paths` replicate trajectories but returns only `last` as
`traj`, so `observe()` hands `sota()` **one** path while the network gets all of them. Every
shipped case uses `n_paths=1`, so this is latent — but `ou.py`/`cir.py`/`gbm.py`/`double_well.py`/
`sindy_sde.py` all expose `n_paths` as a constructor argument, and `GALLERY_RESULTS.md`
explicitly proposes "more paths" as the way to close the σ gap. Doing that would silently turn
finding §1 from a 10× data asymmetry into an `n_paths × 10 ×` one. **Patch:** return the full
stack of trajectories from `_tokens_for`/`observe`, and give `sota()` the same replicates.

---

## 15. CONFIRMED — the linear-Gaussian testbed's "fixed" design matrix is not fixed (medium)

**What is wrong.** `linear_gaussian.py:30-33` builds the design matrix at import time:

```python
_g = torch.Generator().manual_seed(20240630)
A = (torch.randn(N_OBS, D_PARAM, generator=_g) * 0.6
     + torch.linspace(0.4, 1.0, D_PARAM)[None, :])
```

`torch.randn` and `torch.linspace` use **`torch.get_default_dtype()`**, and a seeded generator
produces *different values* in float32 and float64. So the comment "fixed, correlated design
matrix (seeded once, not learned)" is false: any caller that sets
`torch.set_default_dtype(torch.float64)` before importing `amortix` silently gets a different
problem — which is exactly how I found it.

**Evidence** (same seed, same code, only the ambient default dtype differs):

| default dtype | A row 0 | posterior sd, % of prior range |
|---|---|---|
| float32 (as shipped) | `[0.162, -0.741, -0.104, 1.311]` | 6.27 / 7.17 / 4.40 / 4.63 |
| float64 | `[0.306, 0.871, 1.318, 1.310]` | 13.95 / 9.99 / 12.82 / 14.04 |

Under float64 the ground-truth posterior is **~2.3× wider** — a materially harder problem
carrying the same name and the same seed. For the one case in the package whose entire purpose
is to be an unambiguous ground truth, that is a reproducibility defect.

**Patch.** Pin the dtype (`torch.randn(..., dtype=torch.float32)`, likewise `linspace`), or
hard-code the 24 numbers of `A` as a literal. Sweep the package for the same pattern —
`BoxUniform.sample` (`prior.py:41`) and every `torch.randn` in `sde.py` inherit the ambient
dtype too, so a "seeded, reproducible" run is only reproducible at a fixed default dtype.

**For the record**, with the correct (float32) `A` the testbed is well conditioned and is a
fine instrument: posterior sd 6.27 / 7.17 / 4.40 / 4.63 % of range against a prior sd of
28.87 % (info gain 4.0–6.6×), Fisher condition number 25, posterior correlations up to
−0.73 as advertised, and the exact posterior mean achieves MAE 4.52 / 4.96 / 3.31 / 3.58 %
over 300 datasets.

---

## 16 & 17. Checked and clean

- **CIR Feller / positivity is not a problem.** Prior-box probability of `2ab < σ²` is
  **0.00 %** (0 of 20 000 draws); only the extreme corner (a = b = 0.3, σ = 0.5) violates it.
  Across 208 sampled points (8 corners + 200 random) exactly 1 path touched the 0-clamp, and
  0.007 % of all states. The docstring's hedging ("mostly holds", "the simulator clamps X …
  for safety") overstates the risk; the clamp branch is effectively dead code and untested.
- **RK4 is accurate enough.** `|obs(dt) − obs(dt/16)|`: SEIRD median 1.6e-6, max 9.0e-4
  (0.23 × noise_std); FHN median 4.4e-9, max 8.3e-7 (1.7e-5 × noise_std). No action.
- **No blow-ups or non-finite values anywhere.** Corners + random draws, all cases: 0
  non-finite paths; max |X| ≤ 35.7 (GBM at μ = 0.4, σ = 0.6 over 5 years), 13.5 (stoch_lv),
  ≤ 6 elsewhere. GBM never touches its positivity clamp.

---

## Appendix A — how hard are these problems, really?

A degree-2 ridge regression on ~22 hand-written path statistics (first/last/mean/sd/min/max
of x, mean dx, mean dx², mean |dx|, lag-1 autocorrelation, slope of dx on x — per channel),
trained on 2800 draws and scored on 1200 held out:

| case | ridge MAE (mean, % of range) | `GALLERY_RESULTS.md` amortized | zero-information |
|---|---:|---:|---:|
| ou | 9.5 | 12.2 | 25.0 |
| seir | 17.0 | 21.7 | 25.0 |
| gbm | 11.1 | 11.7 | 25.0 |
| cir | 8.8 | 13.9 | 25.0 |
| double_well | 13.6 | 14.6 | 25.0 |
| stoch_lv | 12.6 | 16.3 | 25.0 |
| fhn | 13.7 | 13.4 | 25.0 |
| sindy_sde | 17.4 | 18.0 | 25.0 |

The gallery column is the (flagged-stale) old-engine one, so Appendix B repeats the
comparison against a rerun of the *current* engine at a matched budget.

---

## Appendix B — matched-budget rerun of the current engine

`uv run python examples/gallery.py --quick` (n_train = 3000, epochs = 12, 30 test datasets,
400 posterior draws), run today on the post-fix code. This is half the standard budget
(6000/20), so the amortized column understates the shipped configuration — but it is the
budget my ridge control was given (2800 training draws), so the two are directly comparable.

| case | amortized (3000/12) | ridge on summary stats (2800) | shipped `sota` | zero-info | floor: mean posterior **sd** |
|---|---:|---:|---:|---:|---:|
| linear_gaussian | 20.4 % | **4.3 %** | 3.5 % | 25.0 | 5.62 % (exact) |
| ou | 16.3 % | **9.5 %** | 10.5 % | 25.0 | **11.24 % (exact Bayes; its MAE = 8.67 %)** |
| seir | 23.6 % | **17.0 %** | 25.4 % | 25.0 | 17.42 % (CRB) |
| gbm | 19.3 % | **11.1 %** | 11.1 % | 25.0 | 12.35 % (CRB) |
| cir | 17.6 % | **8.8 %** | 8.6 % | 25.0 | 8.67 % (CRB) |
| double_well | 20.7 % | **13.6 %** | 15.1 % | 25.0 | 12.41 % (CRB) |
| stoch_lv | 24.0 % | **12.6 %** | 8.7 % | 25.0 | 12.40 % (CRB, token budget) |
| fhn | 21.3 % | **13.7 %** | 17.5 % | 25.0 | 13.11 % (CRB) |
| sindy_sde | 21.6 % | **17.4 %** | 31.2 % | 25.0 | 19.50 % (CRB) |

(Floors are the mean over parameters of the posterior **sd** from §3/§4/§13/§15, evaluated at
the token budget where that is meaningful; for a near-Gaussian posterior the corresponding MAE
floor is ≈ 0.8 × that, so e.g. OU's exact Bayes sd 11.24 % ↔ MAE 8.67 %. `sota` may sit below
a token-budget floor because it is given the full fine path — that is finding §1.)

Three things this rerun establishes:

1. **The degree-2 ridge on ~22 summary statistics beats the amortized posterior in 9/9 cases
   at a matched budget**, usually by a factor of ~1.5–2, and beats the (higher-budget, old
   engine) `GALLERY_RESULTS.md` numbers in 7/8. A ridge-on-summary-statistics row belongs in
   the gallery permanently as the "is this problem hard at all" control — the same role the
   25 % prior-mean row plays. Whatever the gallery currently measures, it is not the value of
   the encoder or of flow matching.
2. **The posterior barely contracts at this budget.** Reported `post.std` is 24.1–28.4 % of
   the prior range against a prior sd of **28.87 %** — ratios of 0.83 to 0.98. For OU, where
   the truth is computable, the exact posterior sd averages **11.24 %** while the flow reports
   **24.1 %**: the posterior is **2.1× too wide**, and its error (16.3 %) is 1.9× the exact
   Bayes error (8.67 %).
3. **`cov90 ∈ [80, 97] %` is therefore near-vacuous as a calibration criterion** — the run
   scores "well-calibrated 9/9" (cov90 89–93 %) precisely *because* the posterior is close to
   the prior, and a posterior that returns the prior always has nominal coverage.
   `CALIBRATION.md` already notes that strict SBC only passes 17/29; this adds the reason the
   coverage number looked fine anyway. The gallery's `calib` counter
   (`examples/gallery.py:102`) should be replaced by an SBC pass count and/or a
   `post.std / prior.sd` contraction column.


---

## Appendix C — suggested order of work

Ordered by (impact on published claims) / (effort). None of these require touching the
inference engine.

1. **Add the two control rows to `examples/gallery.py`** — prior-mean (25.00 %) and
   ridge-on-summary-statistics — plus a `post.std / 28.87 %` contraction column, and replace
   the `cov90 ∈ [80,97]` counter with an SBC pass count. ~40 lines. Until this exists, no
   number in the gallery can be interpreted (§App A, §App B).
2. **Give `sota()` the same data as the network**, or add the information-floor column
   (§0, §1, §6, §11). This decides 7 of the 9 current verdicts.
3. **Regenerate `README.md` and `GALLERY_RESULTS.md`** post-leak-fix; the OU table is
   provably impossible as printed (§8).
4. **Fix `ou_mle`** (bias correction + Euler/exact consistency) or relabel it (§2).
5. **`stoch_lv`: `Channel(every=15, count=40)`** — one number, recovers 60 % of the horizon
   the simulator already pays for (§6).
6. **Mark or repair the unidentifiable parameters** — sindy `c2`/`c3`, SEIRD `α`, FHN in the
   excitable regime (§3, §4, §5). Reporting an "error" for these is reporting the prior.
7. **Pin dtypes in `linear_gaussian.A`** (§15), assert channel coverage in `PathObserver`
   (§14), randomize the fast-window start (§9), emit scale-free increments (§10).

## Appendix D — how to reproduce

Each experiment is a standalone script against the installed package; all were run with
`uv run python …` from the repo root. Summarised:

- **exact OU Bayes floor (§0, §8)** — enumerate the observed grid indices
  `U = ∪_c {0, every, …, count·every}`, evaluate the exact Euler-AR(1) log-likelihood of
  `X[U]` on 60 000 prior draws, self-normalize. 300 test datasets.
- **`ou_mle` bias (§2)** — 600 datasets simulated twice from the same parameters, once with
  `simulate_paths` (Euler) and once with the exact OU transition; run `baselines.ou_mle` on
  both and report `E[hat − true]`.
- **Fisher / identifiability (§3, §4, §5, §13, §15)** — for SDEs the exact Euler transition
  gives `F = Σ_k [(∂f/∂m)(∂f/∂m)ᵀ dt/g² + 2(∂log g/∂m)(∂log g/∂m)ᵀ]` (PSD by construction,
  derivatives by central differences); for ODEs `F = JᵀJ / noise_std²`. Rescale by the prior
  range, add the prior precision `I/(1/√12)²`, report `sqrt(diag(inverse))`. 64 draws.
  Summing over all steps vs only token-covered steps gives the two rows of §1.
- **regime splits (§4, §5, §3)** — same Fisher, grouped by peak I / spike count / max |X|.
- **prior-box validity (§12, §16, §17)** — 2^d corners plus 200 random draws; Euler error by
  running `dt` and `dt/8` on *coupled* Brownian increments (the fine increments summed in
  blocks of 8); RK4 error by re-solving at `dt/16`.
- **leak / difficulty probes (§7, §9, §App A)** — per-channel summary statistics → degree-2
  ridge, 70/30 split; per-token single-feature correlation scan; `corr(x0_sampler(m), m)`.
- **matched-budget rerun (§App B)** — `uv run python examples/gallery.py --quick`.
