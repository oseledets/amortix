# Baselines: what to compare against, and in what order

A posterior claim means nothing in isolation. Every result in this repository
is read against a three-tier ladder of baselines, from "sanity floor" to
"information-theoretic ceiling". When adding a case or evaluating a change,
run the tiers in this order and report all of them.

## Tier 0 — controls (any claim must clear these)

* **Prior-only**: predict the prior mean, ignore the data. Exactly 25% mean
  absolute error (as % of the prior range) for a uniform prior. Any method
  that does not beat this has learned nothing. `amortix.controls.prior_mean_error`.
* **Ridge on summaries**: a degree-2 ridge regression on ~20 hand-written
  summary statistics — the cheapest serious estimator, closed-form and convex.
  Neural machinery that does not beat this earns nothing.
  `amortix.controls.RidgeSummary`, `ridge_control_error`.

These two also disambiguate calibration: a posterior that merely returns the
prior passes SBC — only the controls reveal it (see CALIBRATION.md,
"What a passing SBC does and does not mean").

## Tier 1 — classical per-case estimators

Every gallery problem ships its own `sota(tokens, traj, prob)` — the standard
classical estimator a domain practitioner would reach for:

| case | classical baseline |
|---|---|
| ou, gbm | exact MLE (closed form) |
| cir | Euler pseudo-MLE |
| double_well, sindy_sde | Kramers–Moyal / SINDy regression |
| stoch_lv, fhn, seir | (multi-start) nonlinear least squares |
| linear_gaussian | exact posterior mean (the Bayes optimum) |

Caveat recorded in GALLERY_RESULTS.md: several classical baselines read the
full fine simulation path and thus enjoy a 5–9× data advantage over the
network's observed tokens; treat accuracy comparisons accordingly.

## Tier 2 — reference posteriors (the instrument of record)

Point accuracy is half the story; the other half is the full distribution.
Wherever a tractable likelihood exists, compare **distributions**, not points:
signed centre bias and width ratio of the amortized posterior against a
reference posterior on the *same observed points*.

* **Closed form / conjugate**: GBM (Poisson-free log-return conjugacy),
  linear_gaussian (exact).
* **Exact-likelihood MCMC** (`amortix.mcmc.metropolis`, adaptive, box prior)
  with the likelihood factories shipped in the package:
  `merton_logpost_factory` (Poisson-mixture, near-exact),
  `pk_logpost_factory` (log-normal residuals on the Bateman curve),
  `kpp_logpost_factory` (Gaussian noise around the deterministic PDE solve),
  `log_likelihood_ou` / `log_likelihood_gbm` (per-gap exact transitions).
  **Validate the chains** (multi-start agreement) before trusting them —
  a reference is only a reference once checked; see CALIBRATION.md for the
  validation protocol and for a case where this mattered.

Measured rule of thumb: SBC at 300–500 simulations misses width ratios up to
~1.3 and its per-cell p-values flicker when residual biases sit at 0.1–0.2
posterior-sd. SBC is the *screen*; Tier 2 is the *record*.

## Tier 3 — external packages (measured, not just cited)

`sbi` implements NPE — the directly comparable amortized method — and is the
reference implementation behind the `sbibm` community benchmark (Lueckmann
et al., 2021), with which our Hodgkin–Huxley, Lotka–Volterra and SIR-family
cases overlap by design. The head-to-head is measured, not asserted:
`examples/baseline_npe.py` trains both on the SAME simulation budget with
each package's defaults and scores both against the exact GBM posterior
(200 test datasets; bias in exact-posterior sd / width ratio; sbi 0.27,
20k sims, one laptop CPU):

| arm | train | inference/dataset | mu bias/width | sigma bias/width |
|---|---|---|---|---|
| amortix CFM, raw prices | 488 s | 426 ms | +0.00 / 1.00 | −0.00 / 1.05 |
| sbi NPE (MAF), raw prices | 29 s | 13 ms | −0.02 / 1.06 | **+0.45 / 2.64** |
| sbi NPE, hand log-prices | 34 s | 12 ms | −0.04 / 1.01 | +0.32 / 2.29 |
| sbi NPE, hand log-returns | 10 s | 13 ms | +0.07 / 1.07 | +0.13 / 1.44 |
| sbi FMPE, raw prices | 27 s | 370 ms | −0.03 / 1.03 | +0.20 / 2.46 |
| sbi FMPE, hand log-returns | 17 s | 333 ms | +0.01 / 1.03 | −0.02 / 1.09 |
| BayesFlow 2 coupling, raw prices | 384 s | 4 ms | −0.04 / 0.95 | −0.13 / 1.85 |
| BayesFlow 2 coupling, hand log-returns | 459 s | 6 ms | −0.01 / **0.42** | −0.15 / **0.54** |
| BayesFlow 2 flow matching, raw prices | 220 s | 2220 ms | +0.06 / 1.00 | −0.36 / 1.99 |
| BayesFlow 2 flow matching, hand log-returns | 296 s | 2311 ms | −0.04 / **0.46** | −0.30 / 0.85 |
| Simformer (port*), raw prices | 591 s | 1057 ms | −0.04 / 1.08 | −0.30 / 3.16 |
| Simformer (port*), hand log-returns | 548 s | ~1000 ms | −0.08 / 1.10 | **−0.48** / 1.36 |
| exact-likelihood MCMC | — | 42 ms | (exact) | (exact) |

\* Simformer (Gloeckler et al., ICML 2024) is research code: we ported its
authors' minimal example (joint transformer-diffusion, their architecture
and settings, full 75k-step training) and ran it on an H200 GPU — timing
rows not device-comparable, accuracy rows are.

The additive control (OU, fixed 74-point design, same budget, H200; reference
= exact-likelihood MCMC with two validated chains per dataset — worst
discrepancy 0.29 sd over 100 datasets):

| arm | train | inference/dataset | theta bias/width | sigma bias/width |
|---|---|---|---|---|
| amortix CFM, raw values | 183 s | 32 ms | −0.01 / 1.03 | +0.01 / 1.05 |
| sbi NPE, raw values | 128 s | 10 ms | +0.07 / 1.15 | +0.19 / 1.88 |
| sbi FMPE, raw values | 87 s | 147 ms | +0.11 / 1.01 | +0.08 / 1.60 |

On additive data the external BIAS vanishes (the floating-scale mechanism is
gone by construction) and width inflation halves but persists — the control
localizes the dominant GBM failure to input conditioning and leaves a smaller
representation gap even in the benign case. Reproduce:
`python examples/baseline_ou.py --device cuda`.

The expensive-reference case (Fisher–KPP, fixed K=40 spatio-temporal design;
one likelihood eval = one PDE solve, so the MCMC reference costs 271 s per
dataset — two validated chains, worst discrepancy 0.22 sd over 32 datasets):

| arm | train | inference/dataset | D bias/width | r bias/width |
|---|---|---|---|---|
| amortix CFM | 379 s | 31 ms | −0.05 / 1.02 | +0.13 / 1.04 |
| sbi NPE | 218 s | 7 ms | +0.19 / 1.28 | −0.06 / 1.26 |
| sbi FMPE | 115 s | 405 ms | **+7.3 / 12.5** | −0.25 / 1.19 |
| PDE-likelihood MCMC | — | 271 s | (reference) | (reference) |

Here amortization pays for itself after fewer than THREE datasets (training
379 s vs 271 s/dataset for the reference), and FMPE at defaults fails
catastrophically on the diffusivity while staying reasonable on the reaction
rate — defaults behave discontinuously across parameters, visible only
because the expensive reference was computed. Reproduce:
`python examples/baseline_kpp.py --device cuda`.

Readings: (1) at defaults amortix matches the exact posterior; FOUR external
engines on raw prices (MAF / two flow matchings / joint diffusion) all fail
sigma by the floating-scale mechanism (sbi's own z-scoring warning fires) —
the failure CALIBRATION.md's conditioning analysis predicts. (2) Hand-feeding
the exact likelihood coordinates (log-returns) makes the outcomes SCATTER,
not converge: NPE stays 44% wide, FMPE is fine (1.09 — flow matching beats
MAF at fixed input), BayesFlow flips to overconfidence (0.42–0.85, the
dangerous direction), Simformer's port lands −0.48 sd biased. None of these
announces itself; each is visible only against the exact reference. The
representation problem is the method gap, and amortix solves it *learnably*,
from raw data. (3) Cost honesty: the small externals train and sample
faster at these defaults, and on tractable-likelihood single-dataset
problems plain MCMC (42 ms, exact) beats everything amortized; amortix earns
its bill when no likelihood exists, when thousands of datasets/designs are
processed in batch, and when input representation must not be a per-problem
decision. On an H200 GPU amortix trains 2.1× faster with bit-identical
posteriors (CPU-RNG design) and 72 ms/dataset batched inference; per-arm
JSON for both devices is in results/. Reproduce with:

```bash
uv run --with sbi python examples/baseline_npe.py --arms amortix,npe,npe_log,npe_ret,fmpe,fmpe_ret,mcmc
KERAS_BACKEND=torch uv run --with bayesflow python examples/baseline_bayesflow.py --input raw
KERAS_BACKEND=torch uv run --with bayesflow python examples/baseline_bayesflow.py --input returns
```
