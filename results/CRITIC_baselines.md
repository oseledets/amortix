# CRITIC — audit of the classical baselines (`sota`) in the gallery

**Date:** 2026-08-06 · **Scope:** every `sota(tokens, traj, prob)` in `amortix/problems/*.py`
plus `amortix/baselines.py`. **No source file was modified** — this is a report.

**Reproduce.** Every test set below is built exactly as `examples/gallery.py`
builds its own, so the numbers are directly comparable to `GALLERY_RESULTS.md`
(verified: at K=40 the shipped baselines reproduce that table to <1 pp — see
F11). Metric throughout is the gallery's: **mean absolute error as % of the prior
range**, over `K` datasets. The common harness for every table is:

```python
import importlib, numpy as np, torch
def make_case(name, K, seed=123):
    mod  = importlib.import_module(f"amortix.problems.{name}")
    prob = mod.make()
    gen  = torch.Generator().manual_seed(seed)          # same seed as gallery.py
    m    = prob.prior.sample(K, generator=gen)
    tokens, traj = prob.observe(m, generator=gen)
    return mod, prob, m.numpy().astype(np.float64), tokens, traj.numpy().astype(np.float64)

def err_pct(est, truth, prob):                          # the gallery metric
    rng = (prob.prior.high - prob.prior.low).numpy().astype(np.float64)
    return np.abs(est - truth) / rng * 100.0

def observed_union(prob):        # == amortix.mcmc.observed_indices(prob)
    n = prob.observer.n_steps
    return np.unique(np.concatenate([
        np.clip(np.arange(c.count + 1) * c.every, None, n) for c in prob.observer.channels]))
```
Each finding states the estimator variant it measured; all are a few lines on top
of this harness. Known-truth checks (F4, F8, F10) simulate from the stated
transition law directly with `np.random.default_rng(seed)` and compare the
estimator's Monte-Carlo mean/sd against the analytic Cramér–Rao width.

---

## Verdict in one paragraph

The suspicion is **CONFIRMED and is the largest single defect**: 6 of 9 `sota`
functions consume the raw fine path (500–1000 increments) while the network sees
73–122 points. But the audit found a **second, opposite-direction** unfairness of
comparable size: the baselines are **not clipped to the prior box** that the
amortized posterior is confined to by construction, which costs them 15–25%.
The two errors partly cancel in the aggregate column of `GALLERY_RESULTS.md`,
which is why the table looked plausible — but they do **not** cancel per
parameter, and the per-parameter claims ("classical wins on σ", "amortix is 1.8×
better on SINDy") are artifacts. Separately, three estimators do not deliver what
their docstring advertises: `ou_mle` is algebraically the right conditional MLE
but is biased +39% in θ and 1.40× wider than Cramér–Rao, so "near-optimal" is
false; the CIR "Euler pseudo-MLE" is plain OLS and divides by a quantity it does
not guard; the SINDy "library fit" is a degenerate OLS rescued by a clip.
**In 8 of 16 parameters across the five
closed-form SDE cases, the shipped baseline is worse than predicting the prior
mean.**

---

## Table 1 — Information parity (Task 1)

What each `sota` consumes vs what `prob.observer` gives the network. "net pts" is
the union of the channel index sets (`amortix.mcmc.observed_indices` computes
exactly this); it is the whole of the network's information, because every token
feature (`t, x, dx, dx², log dt, component`) is a function of those path values.

| case | observer | `n_tokens` | `obs_dims` | **network sees** | **`sota` consumes** | data ratio | net horizon | parity |
|---|---|---:|---|---|---|---:|---:|---|
| `linear_gaussian` | `_VectorObserver` | 6 | – | 6 scalars = raw `y` | 6 scalars (`traj` ≡ `y`) | 1.0× | – | **OK** |
| `ou` | `PathObserver` | 74 | (0,) | **73 pts / 72 incr** (fast `every=1,count=50`, t≤1.0; slow `every=20,count=24`, t≤9.6) | **full path, 501 pts / 500 incr** | **6.9×** | 96% | **UNFAIR** |
| `seir` | `TimeSeriesObserver` | 40 | (2,4)=I,D | 20 noisy times ×2 series (σ=0.004) | the **tokens** (same 40 noisy values) | 1.0× | 100% | **OK** |
| `gbm` | `PathObserver` | 100 | (0,) | **95 pts / 94 incr** (fast `1×60`, t≤0.6; slow `10×40`, t≤4.0) | **full path, 501 / 500** | **5.3×** | 80% | **UNFAIR** |
| `cir` | `PathObserver` | 74 | (0,) | **73 pts / 72 incr** (fast `1×50`, t≤1.0; slow `20×24`, t≤9.6) | **full path, 501 / 500** | **6.9×** | 96% | **UNFAIR** |
| `double_well` | `PathObserver` | 116 | (0,) | **114 pts / 113 incr** (fast `1×80`, t≤0.8; slow `25×36`, t≤9.0) | **full path, 1001 / 1000** | **8.8×** | 90% | **UNFAIR** |
| `stoch_lv` | `PathObserver` | 180 | (0,1) | **83 pts / 90 incr**, **t ≤ 2.4** (fast `1×50`, t≤0.5; slow `6×40`, t≤2.4) | **101 pts spanning t ∈ [0, 6.0]** | 1.2× pts but **2.5× horizon** | **40%** | **UNFAIR (horizon)** |
| `fhn` | `TimeSeriesObserver` | 25 | (0,)=v | 25 noisy v (σ=0.05) | the **tokens** (same 25 noisy values) | 1.0× | 100% | **OK** |
| `sindy_sde` | `PathObserver` | 125 | (0,) | **122 pts / 121 incr** (fast `1×80`, t≤0.8; slow `20×45`, t≤9.0) | **full path, 1001 / 1000** | **8.2×** | 90% | **UNFAIR** |

Parity is fine exactly where `sota` is written against `tokens`
(`linear_gaussian`, `seir`, `fhn`) and broken exactly where it is written against
`traj`. `amortix/mcmc.py` already solved this problem for the MCMC reference
(`data="observed"` is its default, and it documents the 2.73× / 2.54× σ factor);
the `sota` functions were never updated to match.

---

# Ranked findings

## F1 · CONFIRMED · The σ column is pure information, not method

**What is wrong.** `ou`, `gbm`, `cir`, `double_well`, `sindy_sde` estimate σ from
the quadratic variation of the *full* fine path. σ is the one parameter whose
precision is set almost entirely by the *number of increments*, so this is where
the extra data buys the most.

**Evidence.** Re-running each baseline restricted to exactly the observed points
(union of channel indices; correct per-gap `dt`), σ MAE (% of prior range):

| case | full path (shipped) | restricted to observed | degradation | Cramér–Rao floor for the network's data |
|---|---:|---:|---:|---:|
| `ou` | 1.87% | **4.71%** | **2.5×** | 5.22% |
| `gbm` | 1.73% | **3.78%** | **2.2×** | 3.95% |
| `cir` | 1.67% | **6.51%** | **3.9×** | 5.98% |
| `double_well` | 1.83% | **6.54%** | **3.6×** | 6.31% |
| `sindy_sde` | 1.88% | **6.98%** | **3.7×** | 6.94% |

(The CIR row uses the *same* Euler estimator, restricted — the honest "how much
was extra data" answer. Swapping in the exact non-central-χ² transition, which
can also mine the 24 slow-channel increments for σ, recovers 6.51% → **4.98%**;
still 3.0× the full-path number. See F8.)

The last column is decisive. For a diffusion observed at `n` increments, any
unbiased σ estimator has relative sd `1/sqrt(2n)`, i.e. MAE
`sqrt(2/π)·σ/sqrt(2n)`. **The restricted baselines land on that floor in all five
cases** (within 8%). So the full-vs-restricted gap is not estimator quality — it
is exactly `sqrt(n_full/n_obs)` ≈ 2.2–3.5× of raw information. The degradation
factors reproduce the independent workstream's 2.73× (OU) and 2.54× (GBM) MCMC
findings from a completely different direction.

**Headline impact.** `README.md`'s OU table reads "σ: amortized 5.5%, exact MLE
2.1%" — a 2.6× classical win. Against the *same data the network sees* the MLE
gets **4.7%**, so it is a **1.17× tie**. `GALLERY_RESULTS.md` already says this
is "a data-budget artifact, not a method limit" but still prints the artifact in
the headline table and counts CIR / stoch-LV as SOTA wins. Every sentence of the
form "classical wins on σ" must go.

**Patch.**
```python
# amortix/problems/gbm.py (same shape for ou, cir, double_well, sindy_sde)
def sota(tokens, traj, prob):
    from ..mcmc import observed_indices          # already exists, already tested
    idx = observed_indices(prob)                 # e.g. 95 of 501 for GBM
    s = np.clip(np.asarray(traj, np.float64).reshape(-1)[idx], CLAMP, None)
    gaps = np.diff(idx) * prob.observer.dt_sim   # NON-UNIFORM -- do not use dt_sim
    r = np.diff(np.log(s))
    a = r.sum() / gaps.sum()                     # = (log S_end - log S_start)/T_obs
    sigma2 = np.mean((r - a * gaps) ** 2 / gaps)
    return np.array([a + sigma2 / 2.0, np.sqrt(max(sigma2, 0.0))])
```
The per-gap `dt` is the part that is easy to get wrong: reusing `dt_sim` on a
subsampled path inflates θ/σ by the subsampling factor. If you prefer to keep a
full-path number, print **both** columns and label them
`SOTA(observed)` / `SOTA(full path, upper bound)`.

---

## F2 · CONFIRMED · Baselines are not clipped to the prior box; the amortized posterior always is

**What is wrong.** The amortized posterior cannot leave `[low, high]` (probit
normalization, `prior.py`). The baselines can and do. Only `sindy_sde` clips.
This is a real, unearned handicap: a classical statistician handed the same
uniform prior would report the *constrained* MLE.

**Evidence.** Fraction of shipped estimates falling outside the prior box, and
the effect of clipping (MAE % of prior range, mean over parameters):

| case | out-of-box | shipped | + clip to box | gain |
|---|---|---:|---:|---:|
| `ou` | θ 21%, μ 11%, σ 3% | 11.83% | **9.59%** | −19% |
| `gbm` | μ 19%, σ 2% | 11.46% | **9.65%** | −16% |
| `cir` | a 21%, b 5%, σ 2% | 9.9% (23.4% at K=200, see F8) | **8.06%** | −19% |
| `double_well` | θ₁ 23%, θ₂ 21%, σ 3% | 15.94% | **12.57%** | −21% |

**Headline impact.** This is the finding that most changes the *ranking*.
`GALLERY_RESULTS.md` claims double-well as an amortized win, "14.6% vs 20.7%,
+6 pp over Kramers–Moyal". The clipped full-path Kramers–Moyal scores **12.57%**,
which would **flip that case to a SOTA win**. It survives only once F1 is also
applied (fair + clipped = 19.62%, see F5). Right now two large errors are
silently cancelling, and the reported margin is not evidence of anything.

**Patch.** Clip every `sota` return into the prior box, consistently:
```python
lo, hi = prob.prior.low.numpy(), prob.prior.high.numpy()
return np.clip(out, lo, hi)          # sindy_sde already does exactly this
```
Better still for the regression-based ones, see F5 (bounded least squares beats
clip-after-the-fact).

---

## F3 · CONFIRMED · 8 of 16 parameters: the baseline is worse than a constant

**What is wrong.** For `m ~ U[lo, hi]` the constant predictor `(lo+hi)/2` has MAE
= range/4 = **25.0%** of the prior range, by construction, for every parameter.
Any "SOTA" above 25% is losing to a number typed in by hand. Nobody checked.

**Evidence** (`K = 60`, shipped `sota`, measured constant on the same test set):

| case | parameter | shipped SOTA | constant | |
|---|---|---:|---:|---|
| `ou` | θ | 25.40% | 22.35% | **worse than constant** |
| `gbm` | μ | 28.15% | 24.83% | **worse than constant** |
| `cir` | a | 25.75% | 22.35% | **worse than constant** |
| `double_well` | θ₁ | 23.75% | 22.35% | **worse than constant** |
| `sindy_sde` | c₁ | 26.69% | 22.55% | **worse than constant** |
| `sindy_sde` | c₂ | 42.86% | 23.26% | **worse than constant** |
| `sindy_sde` | c₃ | 51.62% | 28.30% | **worse than constant** |
| `sindy_sde` | **mean** | **28.49%** | **24.14%** | **worse than constant** |

**Headline impact.** "SINDy-SDE: 18.0% vs 32.4%, ~1.8× better" is a comparison
against an estimator that loses to a constant. The honest statement is "amortix
18.0%, constant predictor 25%, classical SINDy fit 32.4% (degenerate)". Most of
this is cured by F2 + F5; what remains (`ou` θ, `gbm` μ) is genuine weak
identifiability and should be *said* rather than dressed up as a baseline.

**Patch.** Add a `constant (prior mean)` row to `examples/gallery.py` — one line,
and it permanently prevents this class of claim:
```python
mid = (0.5 * (prob.prior.low + prob.prior.high)).numpy()
triv = np.abs(mid[None, :] - mt) / rng * 100      # ~25% by construction
```

---

## F4 · CONFIRMED · `ou_mle` is not "near-optimal": +39% bias in θ, 1.40× the Cramér–Rao width

**What is wrong.** `README.md` calls this "a known-optimal classical method" and
"near-optimal". The algebra in `amortix/baselines.py` is correct — it *is* the
exact conditional MLE of the AR(1) representation, and I verified μ and σ are
unbiased and efficient — but the θ = −log(ρ̂)/dt map inherits the classical
small-sample downward bias of ρ̂ and blows it up by 1/dt.

**Evidence.** Known ground truth, exact OU transitions (no Euler error),
θ=1.0, μ=0.0, σ=0.8, n=500, dt=0.02, X0 stationary, R=3000 replicates:

| param | estimator mean | truth | bias | sd | CRB sd | **sd/CRB** |
|---|---:|---:|---:|---:|---:|---:|
| θ | **1.4429** | 1.00 | **+0.443 (+44%)** | 0.624 | 0.447 | **1.395** |
| μ | 0.0008 | 0.00 | +0.001 | 0.2564 | 0.2530 | 1.014 |
| σ | 0.8002 | 0.80 | +0.000 | 0.0256 | 0.0253 | 1.012 |

On the actual gallery test set the shipped estimator's θ bias is **+39.0% ± 3.5%**.
(Only +1.7% of that is the Euler-vs-exact-OU discretization mismatch,
θ·dt/2 — the rest is small-sample AR(1) bias.) μ and σ are clean.

**Fix works.** The Kendall / Marriott–Pope correction
`ρ̂ ← ρ̂ + (1+3ρ̂)/n` reduces the bias to **+6.5%** and cuts known-truth RMSE
0.765 → 0.623 (**−19%**); on the gallery set θ MAE improves 24.81% → 21.21%
(unclipped) and 19.51% → 17.99% (clipped). Note the correction **must** be
combined with box-clipping: pushing ρ̂ toward 1 makes `μ = a/(1−ρ)` explode
(unclipped μ MAE 2.8·10⁶%).

**Headline impact.** OU is the package's *validation* case — the one whose whole
job is to prove the amortized posterior is competitive with a known-optimal
estimator. It is being compared against an estimator that is 40% wider than the
information bound and biased by 0.7 sd. `README.md`'s "amortix beats fine-
resolution MLE on drift θ" is partly beating a bias, not a method.

**Patch** (`amortix/baselines.py`, inside `ou_mle`, after computing `b`):
```python
if bias_correct:                      # Kendall (1954) / Marriott-Pope AR(1)
    b = min(b + (1.0 + 3.0 * b) / n, 1 - 1e-9)
    a = sy - b * sx
```
and clip the returned dict into the prior box at the call site.

---

## F5 · CONFIRMED · The SINDy / Kramers–Moyal drift fits are degenerate; the clip is doing the work

**What is wrong.** The *time convention is correct* — both `double_well` and
`sindy_sde` regress `dX/dt` on the basis evaluated at the **left** endpoint
`x[:-1]` with `dt = dt_sim`, which is the right Itô / Euler–Maruyama convention
for this simulator. The problem is elsewhere: the polynomial library is
near-degenerate over the range the path actually explores, so raw OLS is
garbage and `np.clip(out, lo, hi)` is the estimator.

**Evidence** (`sindy_sde`, K=40): median `cond(design)` = 37 (max 197), median
`max|X|` explored = 0.95 — the `X²`/`X³` columns are barely distinguishable from
`X`. Raw, pre-clip OLS:

| | c₀ | c₁ | c₂ | c₃ |
|---|---:|---:|---:|---:|
| fraction outside prior box | 22% | 50% | **90%** | **92%** |
| raw MAE (% of prior range) | 28% | 88% | **500%** | **751%** |
| after `np.clip` (= shipped) | 18.6% | 35.4% | 44.1% | 46.8% |

**A properly regularized classical fit is much better.** Bounded least squares
(`scipy.optimize.lsq_linear` with the prior box as bounds — the same prior the
network gets, applied *inside* the fit instead of after it), K=300:

| case / variant | mean MAE |
|---|---:|
| `sindy_sde` shipped (full path, clip-after-OLS) | 29.34% |
| `sindy_sde` **full path, bounded LS** | **23.00%** (−22%) |
| `sindy_sde` **observed tokens, bounded LS** | **24.92%** |
| `sindy_sde` constant (prior mean) | 25.61% |
| `double_well` shipped (full path) | 15.94% → 12.57% clipped |
| `double_well` **full path, bounded LS** | **10.96%** (−13% vs clipped) |
| `double_well` **observed tokens, bounded LS** | **19.62%** |

(Ridge was tried too and is worse than bounded LS: 24.19% / 13.04%.)

**Headline impact.** Two claims change:
- "**SINDy-SDE: 18.0% vs 32.4%, ~1.8× better**" → the fair-and-strong baseline is
  **24.92%**, so the margin is **1.4×**, against a baseline that only just beats
  a constant (25.6%). This is still a win, but it is a win over a degenerate
  problem, not over "the classical library fit".
- "**Double-well: 14.6% vs 20.7%, +6 pp**" → fair-and-strong baseline **19.62%**;
  the win survives (F1 and F2 nearly cancel here) but for the wrong reasons, and
  the drift margin is what carries it, not σ.

**Patch.**
```python
from scipy.optimize import lsq_linear
idx  = observed_indices(prob)                       # F1
i0, i1 = idx[:-1], idx[1:]
dts  = (i1 - i0) * prob.observer.dt_sim             # per-gap dt, NOT dt_sim
X, dX = x[i0], x[i1] - x[i0]
A, y = design(X), dX / dts
w = np.sqrt(dts)                                    # equalize Var(dX/dt)=sigma^2/dt
coef = lsq_linear(A * w[:, None], y * w, bounds=(lo[:k], hi[:k])).x
eps  = dX - (A @ coef) * dts                        # residual QV, see F8
sigma = np.sqrt(max(np.mean(eps ** 2 / dts), 1e-12))
```

---

## F6 · CONFIRMED · `stoch_lv`: the baseline is fitted over 2.5× the network's time horizon

**What is wrong.** `sota` hard-codes `obs_idx = np.arange(0, n_steps + 1, 6)` —
101 points spanning the **whole** horizon t ∈ [0, 6]. The observer's slow channel
is `every=6, count=40`, so its last index is 6·40 = **240 of 600**: the network
never sees anything after **t = 2.4**. For an oscillatory system, the number of
observed periods is what identifies the rates, so this is the most severe parity
break in the gallery even though the *point counts* are similar (101 vs 83).

This is also arguably a **problem-definition bug, not only a baseline bug**: a
"slow channel over the full horizon" that stops at 40% of it looks like
`count` was chosen without checking `every * count == n_steps`. Compare the other
cases, which do reach 80–96%.

**Evidence** (K=20, same test set, only the fitted index set changed):

| estimator | α | β | δ | γ | mean |
|---|---:|---:|---:|---:|---:|
| **shipped** (101 pts, t ∈ [0, 6]) | 8.29% | 5.96% | 5.04% | 9.19% | **7.12%** |
| **restricted to the observed times** (t ≤ 2.4) | 10.77% | 13.27% | 11.29% | 18.23% | **13.39%** |
| shipped, 5 random starts | 8.29% | 5.96% | 5.04% | 9.19% | 7.12% |
| restricted, 5 random starts | 10.77% | 13.27% | 11.29% | 18.23% | 13.39% |

**+88% error** once the baseline is fitted to the network's actual data — the
largest parity effect in the gallery, and it is concentrated in γ (9.2% → 18.2%),
the predator death rate, which is exactly what the unseen later oscillations
would pin down.

**Headline impact.** `GALLERY_RESULTS.md`: "Stochastic Lotka–Volterra, amort
16.3% vs deterministic NLS 8.1% — SOTA wins", explained away as "low-noise
regimes favor deterministic fitting". The fair number is **13.39%**: SOTA still
wins, but by **1.2×, not 2×**, and the stated explanation is wrong — most of the
gap was 2.5× more trajectory, not the noise level.

**Patch.** Either fix the observer (`Channel(every=6, count=100)` so the slow
channel spans all 600 steps, which is clearly what "coarse over the full horizon"
in the docstring intended) **or** fit the baseline on `observed_indices(prob)`.
Fixing the observer is the better call — the network is currently being denied
60% of the trajectory for no stated reason:
```python
Channel(every=1, count=50, label="fast"),
Channel(every=6, count=100, label="slow"),   # was count=40 -> stopped at t=2.4
```

**Not a defect:** `_lv_rk4` faithfully reproduces `StochasticLotkaVolterra.drift`
(same clamp, same x0), and the fit is well-behaved — median `nfev` = 10, all 20
runs terminate on `ftol` (never `max_nfev`), and **5-start multistart changes
nothing (0/20 datasets improved, results bit-identical)**. This baseline is not
crippled by its single start. One honest caveat to state in the docs: fitting a
*deterministic* ODE to a *stochastic* path treats integrated process noise as iid
measurement error, so the NLS residuals are correlated and it is not the MLE —
defensible at s = 0.05, but it should be named as an approximation.

---

## F7 · CONFIRMED · Task 5 — the new stationary X0: baselines stay *correct*, become *inefficient*

`ou.py` and `cir.py` now draw X0 from the stationary law instead of X0 = μ / X0 = b.

**Are the baselines still correct?** **Yes.** Both are *conditional* estimators —
`ou_mle` regresses X_{k+1} on X_k, and the CIR `sota` regresses dX on [1, X].
Neither reads X0's marginal law, so consistency cannot depend on which law X0 is
drawn from. Confirmed empirically (OU, K=300, same parameter draws, only the IC
switched):

| X0 | θ | μ | σ | mean |
|---|---:|---:|---:|---:|
| old (X0 = μ exactly) | 23.06% | 8.61% | 2.06% | 11.24% |
| new (stationary) | 24.16% | 8.40% | 1.95% | 11.50% |

The baseline is unmoved. **Any headline change after the IC fix is a
network-side change** — the network lost a free read of μ, the baseline lost
nothing.

**Should they now include the stationary X0 term?** **Yes — it is no longer
negligible, and it is no longer a leak.** For stationary OU on [0, T]:

    I_path(μ) = θ²T/σ²        I_X0(μ) = 2θ/σ²        gain = I_X0/I_path = 2/(θT)

| θ | info gain from X0 | sd reduction on μ |
|---:|---:|---:|
| 0.3 (prior low) | **+66.7%** | −22.5% |
| 1.0 | +20.0% | −8.7% |
| 3.0 (prior high) | +6.7% | −3.2% |

At the bottom of the OU prior box, **the single initial point carries two thirds
as much information about μ as the entire path**. Measured: adding
`log N(X0; μ, σ²/2θ)` to the likelihood improves μ MAE **8.72% → 7.48%** on the
full path and **9.48% → 7.75%** restricted to the observed points (−18%), and
also helps θ (30.26% → 28.97%). CIR obeys the same `2/(aT)` law
(X0 ~ Gamma(2ab/σ², σ²/2a), I_X0(b) = 2a/(σ²b)); measured b MAE 4.93% → 4.26%.

Under the **old** IC this term would have been infinitely informative (μ = X0),
which is precisely why ignoring it used to be right. Under the new IC, ignoring
it is just leaving information on the table.

**Patch** — add an opt-in stationary term:
```python
def ou_mle(path, dt, stationary_x0=True):
    ...                                        # existing conditional MLE = starting point
    if stationary_x0:                          # then refine numerically
        def nll(p):
            th, mu, ls = p; sg = np.exp(ls); rho = np.exp(-th * dt)
            v = sg**2 / (2*th) * (1 - rho**2); m = mu + (x[:-1] - mu) * rho
            out = 0.5 * np.sum(np.log(2*np.pi*v) + (x[1:] - m)**2 / v)
            v0 = sg**2 / (2*th)                # <-- the new term
            return out + 0.5 * (np.log(2*np.pi*v0) + (x[0] - mu)**2 / v0)
        ...
```

**Also stale:** `amortix/mcmc.py`'s module docstring still says *"the simulator
starts every path at X_0 = mu, so the raw data also pins mu exactly"*. That
sentence has been false since commit `8222173` and will mislead the next reader
into thinking the OU μ result is still leaky. `cir.py`'s own class docstring
header also still reads `X0 = b`.

---

## F8 · CONFIRMED · CIR: an unguarded division by `â` produces 8000%-error outliers

**What is wrong.** The docstring says "Euler pseudo-MLE". The Euler transition is
Gaussian with **heteroscedastic** variance σ²·X·dt, so its actual pseudo-MLE is
**GLS** with weights 1/(X·dt). The code runs plain OLS.

**How much does it matter? Less than expected — but the failure mode is nasty.**
Known-truth Monte Carlo (a=1.5, b=1.0, σ=0.3, dt=0.02, n=500, R=3000):

| | OLS RMSE | GLS RMSE | ratio |
|---|---:|---:|---:|
| a | 0.8082 | 0.7970 | 1.014 |
| b | 0.0637 | 0.0637 | 1.000 |

So in the well-behaved regime GLS buys ~1%: **mislabeled, but not materially
inefficient.** Rename it "Euler OLS" or switch to GLS — either is fine.

**The real CIR defect is a broken guard, and it is severe.** `b̂ = c₀/(â·dt)` is
unbounded as `â → 0`, and the guard only catches `abs(a_hat) < 1e-8`:

```python
a_hat = -c1 / dt
if not np.isfinite(a_hat) or abs(a_hat) < 1e-8:   # threshold 5 orders too small,
    a_hat = 1e-3                                  # and it ignores the SIGN
b_hat = c0 / (a_hat * dt)
```

A fit that returns **â = −0.0009** (anti-mean-reverting, so `b` is meaningless)
sails straight through and yields **b̂ = −96.8**, an error of **8104% of the prior
range**. One such dataset is enough to destroy the reported mean:

| K | mean MAE | median MAE | worst b̂ | â there |
|---:|---:|---:|---:|---:|
| 40 | 9.64% | 7.84% | 0.713 | 1.906 |
| 100 | 9.72% | 7.25% | 0.922 | 1.506 |
| **200** | **23.37%** | 7.42% | **−96.8** | **−0.0009** |
| 300 | 9.91% | 7.59% | −0.403 | −0.073 |

The K=200 mean is 3× the K=40 mean because of a **single** dataset. Any CIR
number quoted without clipping or a median is not measuring the estimator.

**Patch:** reject non-mean-reverting fits instead of dividing by them —
```python
if not np.isfinite(a_hat) or a_hat <= 1e-3:   # sign matters: a<=0 => b undefined
    a_hat, b_hat = 1e-3, float(np.mean(X))    # fall back to the sample mean
else:
    b_hat = c0 / (a_hat * dt)
return np.clip([a_hat, b_hat, sigma_hat], lo, hi)
```

Three further CIR notes:
- **`a` carries the same +29% small-sample bias as OU's θ** (known-truth mean
  1.928 vs true 1.500), for the same reason. Same correction applies.
- σ is unbiased and efficient: GLS sd 0.00940 vs CRB 0.00949 (ratio 0.991).
- **The exact non-central-χ² transition is the right restricted baseline, and it
  helps σ.** My first pass reported it as *worse*; that was an optimizer artifact
  (single Nelder–Mead pass). With two starts and a tightened tolerance (K=100,
  all clipped):

  | restricted estimator | a | b | σ | mean |
  |---|---:|---:|---:|---:|
  | Euler OLS on observed 73 | 20.22% | 5.46% | 6.43% | 10.70% |
  | Euler GLS on observed 73 | 18.17% | 5.52% | 6.53% | 10.08% |
  | **exact ncx2 on observed 73** | 20.43% | 5.32% | **4.98%** | 10.25% |

  The σ gain is real and explicable: over the slow channel's Δ = 0.4 gaps the
  Euler approximation mis-attributes drift to diffusion, while the exact
  transition extracts those 24 increments correctly — which is why 4.98% beats
  the *fast-channel-only* CRB of 5.98%. On the **full** path Euler and exact
  agree (a: 18.51% vs 18.70% clipped), confirming the Euler approximation is
  fine at dt = 0.02 and only breaks on the subsampled grid.
- Not a defect, checked and cleared: the `clamp_min(0)` positivity guard **never
  fires** on this prior box (min 2ab/σ² = 3.47, so Feller holds for every draw;
  0 of 300 paths touched 0). So the exact CIR law is *not* misspecified here, and
  a `sota` rewritten around it is legitimate.

---

## F9 · CONFIRMED (minor) · σ from raw quadratic variation carries a small `+E[drift²]·dt` bias

`double_well` and `sindy_sde` use `σ² = mean(dX²)/dt`, which estimates
`σ² + E[drift²]·dt`. Using the drift **residual** instead
(`eps = dX − drift̂·dt`; `σ² = mean(eps²/dt)`, exactly as `cir.py` already does)
removes it:

| case | raw QV bias | residual QV bias | statistical se |
|---|---:|---:|---:|
| `double_well` | +0.96% | **+0.14%** | 2.24% |
| `sindy_sde` | +0.52% | **−0.00%** | 2.24% |

Below the noise floor at the current budget, so it changes no headline — but it
is a free, one-line fix and it will matter if the fast channel is widened.

---

## F10 · CONFIRMED (no action) · `gbm`'s MLE is correct, unbiased and efficient

Good news, recorded so it is not re-litigated. Known-truth exact GBM
(μ=0.10, σ=0.30, n=500, dt=0.01, R=4000):

| param | mean | truth | sd | CRB sd | sd/CRB |
|---|---:|---:|---:|---:|---:|
| μ | 0.1026 | 0.10 | 0.1345 | 0.1342 | **1.002** |
| σ | 0.2995 | 0.30 | 0.00952 | 0.00949 | **1.003** |

The derivation in the docstring is right (`σ̂² = var(r)/dt`,
`μ̂ = mean(r)/dt + σ̂²/2`) and `ddof=1` is a marginal improvement on the strict MLE.
Euler-discretization bias on the package's own paths is **+0.37% ± 0.16%** on σ
— real but one eighth of the 3.16% statistical se. `linear_gaussian` is likewise
clean: `sota` is the exact posterior mean, `tokens ≡ y` (verified
`np.allclose`), and the n=800 rejection sampler contributes 0.13% MC jitter
against a 4.11% MAE. **No action for either.**

`gbm`'s remaining weakness is F2 and F3 only: μ is genuinely unidentifiable here
(sd(μ̂) = σ/√T is 7.5–44.7% of the prior range across the σ prior), 19% of
estimates land outside the box, and the shipped μ number (28.15%) loses to a
constant.

---

## F11 · CONFIRMED · The published tables are stale and must not be quoted

- `GALLERY_RESULTS.md` and `README.md` were measured **before** commit `8222173`,
  i.e. **with the leaky IC** (OU X0 = μ exactly, CIR X0 = b exactly). The
  README's flagship "μ: amortized 1.4% vs MLE 7.8%" was measured when μ was
  literally the first data point. That 5.6× is not a result.
- `GALLERY_RESULTS.md` already carries a staleness banner about the *engine*
  change; it needs a second one about the *IC* change, which invalidates the
  numbers in a different and more serious way.
- **K=40 is too small for these heavy-tailed baselines.** Reproducing the gallery
  test sets exactly, the unclipped shipped baseline's mean MAE swings with K:

  | case | K=40 | K=100 | K=300 | K=300 clipped | K=300 median |
  |---|---:|---:|---:|---:|---:|
  | `ou` | 12.91% | 12.17% | 11.83% | 9.59% | 8.36% |
  | `gbm` | 11.69% | 14.18% | 11.68% | 9.62% | 8.48% |
  | `cir` | 9.64% | 9.72% | 9.91% | 8.06% | 7.59% |
  | `double_well` | **20.70%** | **15.20%** | **15.94%** | 12.57% | 12.30% |
  | `sindy_sde` | 32.38% | 28.46% | 29.34% | 29.34% | 26.91% |

  The K=40 column reproduces `GALLERY_RESULTS.md` (OU 13.3, GBM 11.7, DW 20.7,
  SINDy 32.4) — confirming my test sets are the gallery's. But **double-well's
  20.70% is a 36% overestimate** driven by outliers that vanish at K=100+. That
  single number is the entire "+6 pp over Kramers–Moyal" headline. Raise K, clip
  (F2), and report medians alongside means.

---

## Appendix — corrected comparison

Shipped baseline vs a baseline that is both **fair** (F1: restricted to the
observed points, correct per-gap dt) and **strong** (F2/F4/F5/F7: box-constrained,
bias-corrected, stationary-X0 term). `amort` is the stale `GALLERY_RESULTS.md`
column, shown only to indicate which way each case moves — **it must be
re-measured** (F11).

| case | shipped SOTA | fair + strong SOTA | constant | stale amort | what changes |
|---|---:|---:|---:|---:|---|
| `ou` | 11.83% | **10.75%** (θ 20.6 / μ 7.0 / σ 4.7) | 25.15% | 12.2% | σ win 2.6× → **1.17× tie**; μ term added |
| `gbm` | 11.46% | **11.84%** (μ 19.9 / σ 3.8) | 24.79% | 11.7% | σ 1.7 → 3.8; aggregate unchanged by cancellation |
| `cir` | 23.37% (8.20 clipped) | **10.1%** (ncx2 10.25 / GLS 10.08) | 24.33% | 13.9% | σ 1.7 → 5.0–6.5; b outliers fixed |
| `double_well` | 15.94% (12.57 clipped) | **19.62%** | 25.15% | 14.6% | amort win survives, margin re-derived |
| `sindy_sde` | 29.34% | **24.92%** | 25.61% | 18.0% | "1.8× better" → **1.4×** vs a near-constant |
| `stoch_lv` | 7.12% (K=20) | **13.39%** | – | 16.3% | SOTA win 2× → **1.2×**; γ 9.2 → 18.2 |
| `seir`, `fhn`, `linear_gaussian` | parity already OK | see F12 | – | – | multistart only |

### Recommended order of work
1. **F1** — route every `sota` through `amortix.mcmc.observed_indices`. Biggest
   effect, smallest diff, and the helper already exists and is tested.
2. **F2** — clip all baselines to the prior box (consistency with `sindy_sde`).
   Do 1 and 2 **together**: individually they move the table in opposite
   directions and either one alone makes things *less* honest, not more.
3. **F11** — re-measure `GALLERY_RESULTS.md` and the `README.md` OU table from
   scratch, and add the "constant predictor" column (F3) permanently.
4. **F4/F5/F7/F8** — strengthen the estimators themselves.
5. **F6** — decide whether `stoch_lv`'s observer is under-specified (`every *
   count = 240 ≠ 600`) or the baseline is over-fed, and fix the one that is wrong.
