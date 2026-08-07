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

| case | dim | calib-err | SBC pass | cov50 | cov90 |
|---|---|---|---|---|---|
| linear_gaussian | 4 | 0.8pp | 3/4 | 50% | 91% |
| ou | 2 | 1.5pp | **2/2** | 52% | 92% |
| seir | 5 | 1.3pp | **5/5** | 49% | 90% |
| gbm | 2 | 1.5pp | 1/2 | 50% | 90% |
| cir | 3 | 0.9pp | **3/3** | 50% | 90% |
| double_well | 3 | 1.6pp | **3/3** | 50% | 89% |
| stoch_lv | 4 | 1.4pp | 3/4 | 51% | 89% |
| fhn | 4 | 1.5pp | **4/4** | 50% | 90% |
| sindy_sde | 5 | 1.3pp | **5/5** | 48% | 90% |

**29/32 parameters pass**, mean calibration error **1.3pp**, and every case lands
at 48–52% / 89–92% against nominal 50 / 90.

Per-parameter p-values (p > 0.05 passes):

```
linear_gaussian: m1 0.44   m2 0.03   m3 0.57   m4 0.06
             ou: theta 0.87  sigma 0.32
           seir: beta1 0.51  beta2 0.33  alpha 0.07  gamma_r 0.70  gamma_d 0.20
            gbm: mu 0.92   sigma 0.00
            cir: a 0.34    b 0.42    sigma 0.65
    double_well: theta1 0.50  theta2 0.74  sigma 0.67
       stoch_lv: alpha 0.00  beta 0.21  delta 0.52  gamma 0.19
            fhn: a 0.76    b 0.89    eps 0.66  I 0.89
      sindy_sde: c0 0.16   c1 0.29   c2 0.14   c3 0.16   sigma 0.85
```

## The three failures share one cause

None of them is a location or a width error. All three are about **dependence**:

* **`stoch_lv` alpha (p=0.00)** — alpha and beta enter the dynamics as a product,
  the strongest coupling in the gallery.
* **`linear_gaussian` m2 (0.03) and m4 (0.06)** — both marginal, while m1 and m3
  pass comfortably. The corner plot shows why: our correlations are right in sign
  and structure but attenuated 10–40% (−0.640 vs an exact −0.710, 0.349 vs 0.415,
  0.128 vs 0.216).
* **`gbm` sigma (0.00)** — the one hard failure, and the interesting one: we are
  *more accurate* on sigma than the ridge control (3.79% vs 5.31%) while getting
  the shape of its uncertainty wrong. Diffusion was the systematic SBC offender
  across every 1-D SDE before this campaign; it is now the last one standing.

So the residual is a single defect — dependence structure recovered but
under-shrunk — visible three ways: in SBC, in the corner plot, and in the 1.6×
gap to the Monte-Carlo floor on the exact-posterior testbed.

## What a passing SBC does and does not mean

Passing is necessary, not sufficient, and one reading is easy to get wrong: **a
posterior that simply returns the prior also passes SBC**, and also passes any
`cov90 in [80,97]%` check. That is why this repo never reports calibration alone —
see the prior-only and ridge controls in [`GALLERY_RESULTS.md`](GALLERY_RESULTS.md).

Conversely, a parameter can be flagged as poorly recovered by an accuracy metric
and still be perfectly calibrated. `sindy_sde`'s c3 is exactly that: an accuracy
heuristic marked it "posterior too wide", but SBC passes it at 0.16 — the
posterior is *correctly* wide, the parameter is weakly identified, and a point
estimator merely happens to land closer.

## How this board was earned

The previous canon read 17/29 and was worthless in both directions. It was
measured on a velocity field factorized across parameters (correlations were
impossible by construction), a degenerate time embedding, 88–460 optimizer steps,
problems that leaked their parameters through the initial condition (OU handed
over mu, CIR handed over b), and a chi-square test that rejected 10.7% of
perfectly calibrated parameters at a nominal 5% (47% at n_post=100). All of that
is fixed; the numbers above are not comparable to it, only to the truth.
