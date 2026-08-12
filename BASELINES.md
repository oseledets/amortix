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

## Tier 3 — external methods (context, not gating)

For positioning against the wider SBI field: BayesFlow and the `sbi` package
implement neighbouring amortized methods (NPE and friends); `sbibm`
(Lueckmann et al., 2021) is the community benchmark suite — our
Hodgkin–Huxley, Lotka–Volterra and SIR-family cases overlap with it by
design. This repository's differentiators are the SDE-first zoo with real
numerical solvers, variable-design amortization (p(m | any K points)), and
the exact-reference evaluation harness above.
