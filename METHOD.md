# amortix — the method, as implemented

Amortized Bayesian parameter recovery: given a **prior** `π(m)` and a **simulator**
`d = ℱ(m) + noise`, learn a network that, for *any* observed dataset `d`, returns
samples from the posterior `p(m | d)` in one forward+ODE solve (milliseconds),
with calibrated uncertainty. Training is the method of Sherki–Oseledets–Muravleva
(arXiv:2503.01375) — conditional flow matching with a transformer set-encoder —
plus the calibration-oriented refinements below.

The pipeline, end to end:

```
m ~ π        simulate        tokenize         encode            flow (CFM)
prior  ──►  d = ℱ(m)  ──►  token set T(d) ──► context c ──► posterior p(m|d)
                                              (transformer)   (data-base + ODE)
```

---

## 1. The `Problem` contract

A problem bundles three things (`amortix/sde.py`, `amortix/ode.py`):

- **prior** `π(m)` — `BoxUniform(low, high, names)`.
- **simulator** — a numerical solver producing a trajectory/solution from `m`:
  - `SDEProblem`: `drift(x,m)`, `diffusion(x,m)`, `x0_sampler` → vectorized
    Euler–Maruyama (`euler_maruyama`, vector state, optional `corr_chol` for
    correlated Brownian noise).
  - `ODEProblem`: `rhs(x,m,t)`, `x0` → batched RK4 (`rk4`).
- **observer** — turns a trajectory into a permutation-invariant **token set**.

`simulate(n)` draws `n` parameters and returns `(m, tokens)`; `observe(m)` also
returns the raw trajectory (for classical baselines). Everything is batched over
the `n` parameter draws, which is what makes generating training data cheap.

## 2. Observation → tokens (the physics-aware part)

Each observation becomes a *set* of tokens the transformer ingests.

**SDE** (`PathObserver`), token = `[t/horizon, x, Δx, Δx², log₁₀(dt_obs), comp_id]`:
- `Δx²` is the per-step **quadratic-variation cue** → identifies the diffusion.
- multiple **channels** at different resolutions: a *fast* channel (`every=1`,
  short window) exposes diffusion (high-frequency); a *slow* channel (`every≫1`,
  long horizon) exposes drift / mean-reversion (low-frequency). `log dt_obs` tells
  the encoder which timescale a token is from.
- `comp_id` distinguishes observed state components (multi-D SDEs, `obs_dims`);
  `n_paths` replicate trajectories are concatenated into one set (set-of-paths).

**ODE** (`TimeSeriesObserver`), token = `[t/horizon, value(+noise), channel]`:
selected state components observed (with measurement noise) at chosen times.

## 3. Encoder: transformer set → context vector (`amortix/encoder.py`)

`SetTransformer`: linear embed → `n_layer` pre-norm blocks → norm → pool.
- **attention** with **rotary position embeddings (RoPE)**, **RMSNorm**, **ReLU²**
  feed-forward (the arXiv:2503.01375 choices).
- pooling: **attention pooling (PMA)** — a learned query attends over the tokens
  (`pool="attn"`, default). This replaced mean-pooling, which bottlenecked the
  summary and hurt per-parameter calibration.

Output: a single context vector `c` summarizing the dataset (any token count).

## 4. Parameter space: probit normalization (`amortix/prior.py`)

The flow trains in a normalized space. For a uniform prior we use the **probit map**

    z = Φ⁻¹( (m − low)/(high − low) )          # normalize
    m = low + (high − low)·Φ(z)                # denormalize

which sends the prior **exactly** to the standard normal `N(0,I)` (the flow's base
marginal), and keeps every posterior sample inside the prior box. (An earlier
affine standardization left a Gaussian-base / bounded-uniform-target mismatch that
piled mass at the boundaries.)

## 5. Conditional Flow Matching (`amortix/flow.py`)

Learn a velocity field `v_θ(z, t, c)` transporting the base to the posterior along
the linear (optimal-transport) path. With target `z₁ = normalize(m)` and base
sample `z₀`:

    z_t = (1−t) z₀ + t z₁,     t ~ U(0,1)
    L_CFM = E ‖ v_θ(z_t, t, c) − (z₁ − z₀) ‖²

Inference integrates `dz/dt = v_θ(...)` from `t=0→1`, then denormalizes.

**Solver cost.** The linear-path CFM trajectory is nearly straight, so high-order
integration is wasted work. Measured against an RK4/200-step reference (OU, 20
datasets × 400 draws), the deviation in posterior mean/std as % of prior range:

| solver | steps | net evals | error | time |
|---|---|---|---|---|
| RK4 | 60 | 240 | 0.000% | 21.7 s |
| **midpoint** | **20** | **40** | **0.002%** | **3.3 s** |
| midpoint | 10 | 20 | 0.009% | 1.8 s |
| Euler | 20 | 20 | 0.143% | 1.5 s |

Midpoint/20 is the default: **6.6× cheaper than RK4/60 for a 0.002% difference**
— negligible next to calibration errors of a few percentage points. Euler is
visibly worse and not worth the extra 2×. `sample_batch` also solves many
datasets jointly (`chunk`), which matters for SBC studies over hundreds of them.

**How `v_θ` is conditioned on the data (`conditioning=`):**
- `"xattn"` (**default**): dense, DiT-style. The encoder's *full per-observation
  token memory* `M = [T, dim]` is kept (no pooling bottleneck). One token per
  parameter (carrying `z_i`) cross-attends to `M` at every block, with `adaLN(t)`
  modulation (`CrossCondVelocity`). So the σ-token can look straight at the
  high-frequency increment tokens, the drift-token at the long-horizon ones. The
  memory's K,V depend only on the dataset → computed once and reused across the
  whole ODE solve (`encode_memory`), keeping sampling cheap.
- `"concat"`: lightweight. Pool `M → c` (one vector), `VelocityNet` = MLP over
  `[z, sinusoidal(t), c]`. Faster but the single-vector summary bottlenecks
  per-parameter information — this is what limited calibration (see §9).

## 6. Data-dependent base — spread alignment (`BaseHead`)

Flow matching can map *any* source to *any* target, but if the base spread differs
greatly from the **conditional** posterior spread (narrow for well-identified
parameters), the deterministic ODE must perform a large stiff contraction that a
finite network underfits → a **mis-calibrated** posterior. So we align the source
to the target *per dataset*: a head predicts a Gaussian base

    (μ̂, ŝ) = BaseHead(c),    z₀ = μ̂ + ŝ ⊙ ε,   ε ~ N(0,I)

trained to match the posterior's first two moments via a Gaussian NLL

    L_base = E[ ½ ( (z₁ − μ̂)² / ŝ² + 2 ln ŝ ) ]
    L = L_CFM + λ · L_base

The flow then only refines the *shape*. The NLL keeps `ŝ ≈ posterior std` — an ODE
cannot create spread from a point, so the base must seed it. (`base="standard"`
recovers a plain `N(0,I)` source; `base="data"` is the default. The data-base has
extra parameters, so it needs an adequate training budget to pay off.)

**Inference:** `c = Encoder(T(d))` → `z₀ = μ̂ + ŝ ε` → RK4 ODE → `m = denormalize(z₁)`.
One ODE solve per dataset; amortized over all possible datasets.

## 7. Diagnostics & baselines

- `amortix/diagnostics.py`: **Simulation-Based Calibration** (rank uniformity),
  coverage curve (derived from ranks), `diagnose()` report + SBC/coverage plot.
  A single coverage number can look right while SBC fails — SBC is the real test.
- each problem ships a **classical SOTA baseline** (`sota(tokens, traj, prob)`):
  exact MLE (OU, GBM), pseudo-MLE (CIR), Kramers–Moyal / SINDy least-squares
  (double-well, polynomial-drift), nonlinear least squares (SEIRD, FHN, stoch-LV).
  `examples/gallery.py` benchmarks amortized vs SOTA; `examples/calib_gallery.py`
  runs SBC across the gallery.

## 8. Summary of design choices that matter

| choice | why |
|---|---|
| multi-resolution channels + `Δx²` token | drift (low-freq) vs diffusion (high-freq) identified jointly |
| transformer + RoPE + attention encoder | permutation-invariant encoding of variable-size data |
| **cross-attention conditioning (default)** | velocity attends to all observation tokens — no single-vector bottleneck; per-parameter info preserved |
| probit normalization | base = prior marginal exactly; samples stay in-box |
| linear-path CFM | simple, OT-straight transport; one ODE solve to sample |
| data-dependent Gaussian base | aligns source/target spread per dataset → calibration |
| SBC harness | strict calibration check beyond a single coverage number |

## 9. Calibration: what's solved, what isn't (honest)

**Solved on simple posteriors.** GBM (2 params, tractable) is the clean case: a
controlled same-budget A/B (12k/40) showed dense **cross-attention conditioning**
beats `concat` — mu SBC-p 0.002→0.356, sigma 0.004→0.038, calib-err 4.3→1.7pp;
and at 50k/60/dim96 GBM passes SBC fully (mu 0.165, **sigma 0.571**, 2/2). So
collapsing the data to one vector *was* a real bottleneck, and for simple
posteriors fidelity scales to ~perfect with conditioning + budget. The widened
fast-channel hypothesis for σ was *refuted* (60→240 tokens didn't move it);
dense conditioning + scale did.

**Not solved across the gallery.** The canonical run (xattn, 50k/60, SBC 500×200)
scores **17/29 SBC-pass, 3.7pp** — essentially the same aggregate as the concat
12k/40 canon (16/29, 3.8pp). Per case it is mixed: gbm 2/2, fhn 3/4, ou 2/3 up;
but cir 1/3, double_well 1/3, stoch_lv 1/4 with systematic (p≈0) per-parameter
biases. The failures cluster on **strongly-correlated** parameters (Lotka–Volterra
α/β, SEIR β₂/γ_d enter as products), **multimodal** ones (bistable double-well θ₂),
and **weakly-identified** ones (CIR a/b, FHN I). Coverage stays usable everywhere
(cov50 45–63%, cov90 86–92%); it is the strict rank-uniformity that fails.

**Leading suspect:** the data-dependent base is a *diagonal* Gaussian — it cannot
seed posterior **correlations**, which are strong in these coupled dynamics; the
velocity must then build all correlation/multimodality from a factorized start,
which it underfits. Candidate directions: a correlated/low-rank or mixture base,
more ODE steps for complex posteriors, and a controlled same-budget conditioning
A/B on a coupled case (the gallery comparison above is budget-confounded). Note
this is a *fidelity* gap; posterior *sharpness* remains correctly
information-limited (OU's θ stays broad — proper Bayesian behavior).
