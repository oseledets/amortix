# A prior and a simulator suffice for amortized inference in a user-defined system

A damped harmonic oscillator is added as a new inference problem by subclassing `DesignProblem` with a prior box and a `trajectories()` simulator — about 25 lines in total, all shown below. Random-design training, tokenization with observation noise, and inference at any number of observation points are inherited from the base class unchanged. Trained at the same budget as the GBM quickstart, the posterior concentrates at the generating $(\omega, \gamma)$ from 12 noisy observations of one trajectory.

<img src="../../../docs/media/custom_oscillator.png" width="100%">

*Left: one simulated trajectory (grey) and its $K = 12$ observed points (blue), each carrying additive Gaussian noise of standard deviation 0.05. Right: 2,000 draws from the amortized posterior in the $(\omega, \gamma)$ plane; the black cross marks the generating parameters. The thin band of draws stretching toward large $\omega$ consists of high-frequency aliases, discussed below.*

## The problem

$$\ddot x = -\omega^2 x - 2\gamma\,\dot x, \qquad x(0) = 1,\quad \dot x(0) = 0,$$

with a uniform box prior $\omega \in [0.5,\ 3.0]$ (frequency), $\gamma \in [0.05,\ 0.5]$ (damping). The simulator integrates 400 steps of size $dt = 0.05$ (horizon $T = 20$) with the semi-implicit Euler update — the velocity is advanced first, then the position with the updated velocity. The object of inference is the parameter pair of this discrete simulator exactly as written.

An observation set (a *design*) is $K$ grid times drawn uniformly at random and time-sorted — $K = 12$ in this run, with the class declaring the admissible range $4 \le K \le 64$ (`k_min`, `k_max`). Each observed value is $x(t)$ plus additive Gaussian noise of standard deviation `obs_noise = 0.05`, applied by the base class at tokenization time.

## The code, walked through

The entire problem definition:

```python
class DampedOscillator(DesignProblem):
    obs_noise = 0.05

    def __init__(self):
        self.prior = BoxUniform(low=[0.5, 0.05], high=[3.0, 0.5],
                                names=["omega", "gamma"])
        self.observer = DesignObserver(dt_sim=0.05, n_steps=400, k_max=64)
        self.k_min = 4

    def trajectories(self, m, generator=None):
        om, ga = m[:, 0], m[:, 1]
        x = torch.ones(m.shape[0]); v = torch.zeros(m.shape[0])
        out = torch.zeros(m.shape[0], 401, 1); out[:, 0, 0] = x
        dt = self.observer.dt_sim
        for i in range(400):
            a = -(om ** 2) * x - 2.0 * ga * v
            v = v + dt * a
            x = x + dt * v
            out[:, i + 1, 0] = x
        return out
```

`BoxUniform` is the prior; `DesignObserver` records the grid metadata (step size, number of steps, largest design size); setting `obs_noise` as a class attribute is all it takes for the base class to corrupt observations with measurement noise whenever they are tokenized. `trajectories` is vectorized over the batch dimension and threads the `generator` argument — these dynamics are deterministic, so it goes unused here, but a stochastic simulator draws from it (the GBM quickstart does).

Training is the same call as in the other gallery examples:

```python
torch.manual_seed(0)
post = model_of_size(prob, "tiny")
post.fit(n_train=3000, steps=1200, batch=256,
         retokenize=prob.make_retokenizer(), verbose=True)
```

`make_retokenizer()` comes from the base class: the 3,000 trajectories are simulated once, and at every optimizer step each one is re-observed through a fresh random design — a new size $K$ from the package's mixed size law on $[4, 64]$, new uniformly random times, and a new noise draw. One simulation serves unboundedly many designs, and the noise realization is never repeated.

Inference on one seed-fixed instance:

```python
gen = torch.Generator().manual_seed(3)
m_true = prob.prior.sample(1, gen)
raw = prob.trajectories(m_true, gen)
tidx, cidx = prob.sample_design(gen, 12)
tokens = prob.tokens_for(raw[0], tidx, cidx, gen)
d = post.sample(tokens, n=2000)
```

`tokens_for` applies the observation noise and packs the 12 points into the same six-feature token layout used during training; `post.sample` returns 2,000 posterior draws conditioned on them. The design size is chosen at query time — the same trained network answers any $K$ in $[4, 64]$.

## What comes out

The final printout of the recorded run:

```
true (omega, gamma): [0.5106588006019592, 0.09750621020793915]
posterior mean     : [0.5780892372131348, 0.08795817196369171]  sd [0.32611966133117676, 0.023325694724917412]
```

The damping is recovered tightly: the posterior standard deviation of $\gamma$ is 0.023, about 5% of the prior range 0.45. For $\omega$, the bulk of the draws sits at the truth (0.51, near the lower prior edge; see the right panel), while the standard deviation of 0.33 is inflated by the thin tail of draws reaching toward $\omega = 3$, and the posterior mean of 0.58 is pulled above the bulk by the same tail.

That tail is a property of the inference problem at this design size. Twelve noisy samples of a slow, damped oscillation are also consistent with much faster oscillations that pass near the same points — the aliasing familiar from sparse sampling — so a posterior that excluded them would claim more than the data contain. Adding observation points removes the tail: with a denser design the aliases no longer fit through the observed values, and the same network, queried at larger $K$, drops them.

The run log for this example records only the printout above; the script's own estimate of the training time is a few minutes (one laptop CPU, indicative only), with the fit budget identical to the quickstart (`n_train=3000, steps=1200, batch=256`). The instance is fixed by `manual_seed(3)` and the initialization by `torch.manual_seed(0)`; the printed mean and standard deviation still vary slightly between platforms because training is stochastic, which is why the test below asserts margins rather than exact values.

## Why believe it

* [`tests/test_examples.py`](../../../tests/test_examples.py)`::test_oscillator_recovers_truth` pins a shrunk version of this script — the `pico` model, `n_train=1500`, 400 optimizer steps, the same seed-3 instance with $K = 12$ — and asserts recovery of the generating parameters: the posterior mean must lie within 0.35 of the prior range of the truth in each parameter, and the posterior standard deviation must stay below half the prior range in each parameter.
* The test imports `DampedOscillator` from this example file itself (`_example_module("04_custom_problem")`), so the class checked in CI is the class shown above.
* A user-defined system has no external reference — nothing outside the simulator knows its posterior, so the check available here is recovery of the generating parameters on a seed-fixed instance. When a tractable likelihood for the system can be written down, `amortix.evaluation.build_eval_set` constructs per-design reference posteriors (two independent reference draws per instance, cross-checked against each other before the set is saved), and the comparison becomes direct, as in the GBM quickstart with its closed-form reference.

## Run it

```bash
python examples/gallery/04_custom_problem.py                 # train + infer; a few minutes on CPU
python examples/gallery/04_custom_problem.py --png           # also render docs/media/custom_oscillator.png
python examples/gallery/04_custom_problem.py --ckpt osc.pt   # save the model; later runs load it and skip training
```

The recorded run log carries no timing lines; the estimate above is the script's own (one laptop CPU, indicative only). The script is [`04_custom_problem.py`](../04_custom_problem.py).

## References

* [arXiv:2503.01375](https://arxiv.org/abs/2503.01375) — the amortized-posterior method implemented by the package.
* [`report/techreport.pdf`](../../../report/techreport.pdf) — the evaluation methodology: reference construction and validation, and how posteriors are scored across the problem zoo.
