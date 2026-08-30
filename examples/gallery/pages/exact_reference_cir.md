# A frozen evaluation set carries validated exact-likelihood references and its own resolution floor

The other gallery pages train a model and check it against a single reference instance; this one runs the package's full evaluation instrument. `build_eval_set` freezes four observation instances of the Cox--Ingersoll--Ross process at $K = 20$ points each, draws **two independent** exact-likelihood MCMC chains per instance, refuses the set if the chains disagree, and records the set's own resolution floor; `evaluate` then scores a trained posterior against it in seconds. CIR is imported from the package rather than defined in the script: its simulator draws the exact noncentral chi-square transition, and that simulator is what makes an exact-likelihood reference possible at all — defining your own system is [`01_quickstart_gbm.py`](../01_quickstart_gbm.py) and [`04_custom_problem.py`](../04_custom_problem.py). A `tiny` model trained at a small budget (8,000 trajectories, 3,000 optimizer steps) scores a median FID of 0.3369 against a floor of 0.0095: a well-resolved measurement of a posterior that the budget leaves visibly over-dispersed.

<img src="../../../docs/media/cir_reference.png" width="100%">

*The four frozen evaluation sets in the $(a, b)$ plane: 2,000 draws from the amortized posterior (blue dots) over the highest-density regions of the exact-likelihood MCMC reference (orange contours); the black cross marks the generating $(a, b)$. Each panel title carries that set's FID, computed on all three parameters with the same draw configuration as the printed median; the suptitle quotes the floor at the 2,000-draw configuration.*

## The problem

$$dX = a(b - X)\,dt + \sigma\sqrt{X}\,dW,$$

with a uniform box prior $a \in [0.3,\ 3.0]$ (mean-reversion rate), $b \in [0.3,\ 1.5]$ (long-run level), $\sigma \in [0.10,\ 0.50]$. The simulator (`CIRDesign` in [`amortix/problems/design_basic.py`](../../../amortix/problems/design_basic.py)) runs 500 fine steps of size $dt = 0.02$ (horizon $T = 10$), starting from the stationary $\mathrm{Gamma}(2ab/\sigma^2,\ \sigma^2/2a)$ distribution. An observation set (a *design*) is $K$ grid times drawn uniformly at random and time-sorted, with $K \in [2, 128]$ admissible and no observation noise; the evaluation sets here use $K = 20$ (duplicate times are merged, so a set carries up to 20 distinct points).

**The simulation is exact.** Each fine step draws the noncentral chi-square transition through its Poisson--Gamma representation — with $e = e^{-a\,dt}$, $c = \sigma^2(1 - e)/4a$, $\mathrm{df} = 4ab/\sigma^2$ and $\lambda = X e / c$, the next state is $c$ times a noncentral $\chi^2(\mathrm{df}, \lambda)$ draw:

```python
lam = x * edt / c
k = rng.poisson(lam / 2.0)
x = c * rng.gamma(shape=df / 2.0 + k, scale=2.0)       # ncx2 exact
```

That choice is what makes the exact reference possible: the reference likelihood describes the same chain the simulator generates, gap by gap, with no discretization gap between the data-generating process and the posterior it is scored against. The class docstring notes the contrast with OU, whose reference composes its Euler chain in closed form; CIR's Euler chain has no such form, so the simulator is lifted to the exact transition instead.

## The code

Training is the same `fit` call as the other gallery examples, at the `tiny` size (width 32, two transformer blocks) and a small budget:

```python
post = model_of_size(prob, "tiny")
post.fit(n_train=8000, steps=3000, batch=256,
         retokenize=prob.make_retokenizer(), verbose=True)
```

The central exhibit is what follows — the two calls that everything the technical report measures also goes through, from [`amortix/evaluation.py`](../../../amortix/evaluation.py):

```python
es = build_eval_set(prob, "cir", K=20, n_sets=4, n_chain=20000,
                    seed=11, workers=4)
print(f"\nevaluation set: {es!r}")
r = evaluate(post, es, n_draw=2000)
print(f"median FID {r['fid_median']:.4f} against a floor of "
      f"{r['null_median']:.4f}")
```

`build_eval_set` draws 4 parameter vectors from the prior, one trajectory and one $K = 20$ design each — all fixed by `seed=11` — and tokenizes them exactly as training does. For each instance it then draws two **independent** references at different seeds (`100 + i` and `77000 + i`): each is a 20,000-draw adaptive-Metropolis chain over the exact log-posterior, thinned to 4,000 stored draws, run in parallel worker processes (`workers=4`). The log-posterior (`cir_logpost_factory`, same file as the simulator) is the noncentral chi-square gap transitions plus the stationary Gamma density of the start — the same $c$, $\mathrm{df}$, $\lambda$ algebra the simulator uses:

```python
ll = float(np.sum(ncx2.logpdf(s[1:] / c, df, lam) - np.log(c)))
ll += float(sp_gamma.logpdf(s[0], a=df / 2.0,
                            scale=sg * sg / (2.0 * a)))
```

The second chain validates the first: build_eval_set refuses the set when the chains disagree — from the `build_eval_set` docstring: *"Raises if the two independent reference draws disagree by more than `max_discrepancy` posterior standard deviations on any instance: such an evaluation set measures its own sampler rather than the model, and saving it would let that number reach a table."* The code that measures the disagreement:

```python
@property
def discrepancy(self) -> np.ndarray:
    """Per-set, per-parameter |mean_a - mean_b| in posterior-sd units."""
    sd = 0.5 * (self.chain_a.std(1) + self.chain_b.std(1))
    return np.abs(self.chain_a.mean(1) - self.chain_b.mean(1)) / np.maximum(sd, 1e-12)
```

with the default limit `max_discrepancy=0.25`. The second chain also prices the instrument's resolution:

```python
@property
def floor(self) -> float:
    """Median FID between the two independent reference draws: the
    smallest difference this evaluation set can resolve."""
    return float(np.median([fid(self.chain_b[i], self.chain_a[i])
                            for i in range(len(self.chain_a))]))
```

The result is an `EvalSet`, a frozen artifact: tokens, mask, both chains, the true parameters, and a `meta` record of how it was made round-trip through a single `.npz` (this script keeps the set in memory; passing `path=` writes it, and everything downstream loads the file instead of rebuilding).

`evaluate` draws 2,000 samples from the model per instance and computes the FID of each against the first chain. The floor it reports is recomputed for the measurement actually made, per the comment in the source:

```python
# The floor must be measured in the SAME estimator configuration as the
# value: n draws on the left, the full reference on the right. Comparing an
# n-vs-n floor against an n-vs-N measurement understates the offset,
# because the estimator's noise has two terms (one per sample set) and only
# the model's shrinks with n.
```

The `--png` path draws with the same configuration (`n=2000, seed=0`), which is why the per-panel FIDs in the figure are consistent with the printed median.

## Output

The training trace of the recorded run (`cfm` is the flow-matching training loss; times in parentheses are cumulative seconds on one laptop CPU, indicative only):

```
[fit] simulating 8000 training trajectories (31 batches/epoch)...
  epoch   0  step     31  cfm 2.1138  (38.1s)
  epoch   9  step    310  cfm 0.8374  (396.2s)
  epoch  18  step    589  cfm 0.7510  (734.8s)
  epoch  27  step    868  cfm 0.7380  (1127.6s)
  epoch  36  step   1147  cfm 0.7375  (1464.1s)
  epoch  45  step   1426  cfm 0.7238  (1690.9s)
  epoch  54  step   1705  cfm 0.7240  (1876.3s)
  epoch  63  step   1984  cfm 0.7061  (2075.1s)
  epoch  72  step   2263  cfm 0.7162  (2216.6s)
  epoch  81  step   2542  cfm 0.7045  (2277.4s)
  epoch  90  step   2821  cfm 0.7073  (2343.2s)
  epoch  96  step   3007  cfm 0.7015  (2403.0s)
```

and the final printout:

```
evaluation set: EvalSet(cir, K=20, 4 sets, chain=20000, floor=0.0087, max inter-chain 0.100 sd)
median FID 0.3369 against a floor of 0.0095
```

The repr is the set's own certificate: the worst per-parameter disagreement between the two reference chains is 0.100 posterior standard deviations, against the admission limit of 0.25, and the floor at the stored chain length is 0.0087. The score line quotes 0.0095 instead — the same floor recomputed at the 2,000-draw configuration of the measurement, as the source comment above requires.

The per-set FIDs in the figure are 0.4423, 0.5341, 0.2316 and 0.0611 — the printed 0.3369 is their median. The spread of nearly an order of magnitude across four instances is why the default `n_sets` is 32; this script builds 4 to keep the reference chains to minutes. The figure shows what the numbers mean: in sets 0--2 the blue draws cover the orange reference regions but spill well beyond them — toward larger $a$ and across $b$ — so the amortized posterior at this budget is too wide rather than centered in the wrong place; set 3 sits nearly on top of its reference at 0.0611.

The over-dispersion reflects the training budget. This run uses the `tiny` model with 8,000 simulations and 3,000 optimizer steps; the technical report's main comparison trains the same architecture under the full recipe — 90,000 optimizer steps, simulation budgets from 5,000 to 480,000 per system — and its Cox--Ingersoll--Ross row reads $0.0733 \pm 0.0110$ at the same `tiny` size (against a floor of 0.0028 on the report's longer-chain sets).

## Verification

* **Two independent chains, and a gate.** Every instance carries two reference draws from different seeds, and `build_eval_set` raises rather than returning a set whose chains disagree by more than 0.25 posterior standard deviations on any parameter of any instance. The gate runs on every execution of this script; the recorded run passed it at 0.100 sd.
* **The floor travels with the score.** 0.3369 is printed next to 0.0095 in the same line, and the floor is itself a measurement: the FID between the two reference chains at the measurement's own draw count. The ratio here is about 35; `evaluate` flags a comparison as unresolved when it falls below roughly 2.
* **The reference describes the simulator, exactly.** The likelihood the chains sample is the noncentral chi-square transition over each observed gap plus the stationary Gamma start — the same distributions the simulator draws from, so no discretization error separates the data-generating chain from the posterior it is scored against.

Unlike the GBM and oscillator pages, this example has no shrunk counterpart pinned in [`tests/test_examples.py`](../../../tests/test_examples.py); the checks above are the instrument's own, and the admission gate re-runs whenever the set is rebuilt.

## Running the example

```bash
python examples/gallery/03_exact_reference_cir.py                 # train + build + score, ~30 min
python examples/gallery/03_exact_reference_cir.py --png           # also render docs/media/cir_reference.png
python examples/gallery/03_exact_reference_cir.py --ckpt cir.pt   # load the checkpoint if it exists, else train and save it
```

The ~30 minutes end to end on CPU is the script's own estimate; in the recorded run the training trace ends at 2403.0 s, with the reference chains built afterwards in four worker processes. The script is [`03_exact_reference_cir.py`](../03_exact_reference_cir.py).

## References

* [arXiv:2503.01375](https://arxiv.org/abs/2503.01375) — the amortized-posterior method implemented by the package.
* [`report/techreport.pdf`](../../../report/techreport.pdf) — the evaluation methodology: reference construction and validation, the FID normalizations and floors, and the results across the full problem zoo, including the Cox--Ingersoll--Ross row quoted above.
* J. C. Cox, J. E. Ingersoll, S. A. Ross. A theory of the term structure of interest rates. *Econometrica* 53(2), 1985 — the CIR process.
