# amortix

Amortized Bayesian parameter recovery for dynamical systems, in pure Python on top of PyTorch.

Given a prior over the parameters of an ODE/SDE/PDE model and a simulator, `amortix` trains a
posterior approximator once, by conditional flow matching: a transformer set-encoder reads the
observed points and conditions a velocity field that transports Gaussian noise to the posterior.
After training, inference for any new observation set is a single batched ODE solve — about 70 ms
for 4,000 posterior draws on one GPU, under a second on a laptop CPU. The method is that of
[Bayesian Inverse Problems Meet Flow Matching](https://arxiv.org/abs/2503.01375) (Sherki,
Oseledets, Muravleva); the same recipe applied to bioprocess calibration is
[arXiv:2604.22496](https://arxiv.org/abs/2604.22496).

## Installation

From a clone, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra plot
uv run amortix cases          # smoke test: lists the fixed-design cases
```

or as a package in your own environment:

```bash
pip install git+https://github.com/oseledets/amortix.git
```

## Quick example

Recover the two parameters of geometric Brownian motion and compare the sampled posterior with
the exact one on the same observed points (a few minutes on a laptop CPU):

```python
import torch
from amortix.evaluation import fid, model_of_size
from amortix.problems.design_basic import GBMDesign, gbm_exact_from_points

prob = GBMDesign()                        # dX = mu X dt + sigma X dW, uniform prior
post = model_of_size(prob, "tiny")
post.fit(n_train=3000, steps=1200, batch=256,
         retokenize=prob.make_retokenizer())

gen = torch.Generator().manual_seed(1)
m_true = prob.prior.sample(1, gen)        # one synthetic observation set
raw = prob.trajectories(m_true, gen)
tidx, cidx = prob.sample_design(gen, 20)  # 20 observation points at random times
tokens = prob.tokens_for(raw[0], tidx, cidx, gen)

draws = post.sample(tokens, n=2000)       # posterior draws, milliseconds
exact = gbm_exact_from_points(prob, raw[0, :, 0], tidx, n_samples=2000)
print(fid(draws.numpy(), exact))          # distance to the exact posterior
```

The trained network is *design-amortized*: it accepts any number of observation points at
arbitrary times (and sensors), so the 20 points above can be 5 or 100 without retraining. The
production configuration used for the numbers in the technical report is one line away:

```python
from amortix import FlowPosterior, DESIGN_ZOO

prob = DESIGN_ZOO["heston"]()             # or any DesignProblem subclass
post = FlowPosterior(prob)
post.fit(n_train=120_000, steps=72_000,   # about 1 h on one GPU
         retokenize=prob.make_retokenizer())
```

Four worked examples, each self-contained and commented, are in
[`examples/gallery/`](examples/gallery/):

| script | shows |
|---|---|
| [`01_quickstart_gbm.py`](examples/gallery/01_quickstart_gbm.py) | train + sample + check against an exact posterior |
| [`02_any_design_pk.py`](examples/gallery/02_any_design_pk.py) | one network, any number of observation points |
| [`03_exact_reference_cir.py`](examples/gallery/03_exact_reference_cir.py) | frozen evaluation sets and validated MCMC references |
| [`04_custom_problem.py`](examples/gallery/04_custom_problem.py) | add your own system in ~25 lines |

## What is inside

- **Problem contract.** A `Problem` is a prior, a simulator, and an observation spec.
  `SDEProblem` builds the simulator from a drift and a diffusion (Euler–Maruyama, vector state,
  correlated noise); `ODEProblem` from a right-hand side (batched RK4). Adding a system means
  writing these functions and a prior box; see `04_custom_problem.py`.
- **Posterior model.** `FlowPosterior` combines a SetTransformer encoder (RoPE attention over
  observation tokens, so irregular sampling is native) with a conditional-flow-matching velocity
  field and an ODE sampler.
- **Benchmark suite.** Fourteen systems with validated reference posteriors: linear-Gaussian,
  geometric Brownian motion, Ornstein–Uhlenbeck, Cox–Ingersoll–Ross, double well,
  polynomial-drift SDE, stochastic Lotka–Volterra, Heston, Merton jump-diffusion, SEIR,
  FitzHugh–Nagumo, Hodgkin–Huxley, pharmacokinetics, and a Fisher–KPP reaction–diffusion PDE.
  References are exact where a closed form exists and are otherwise computed by adaptive MCMC,
  with particle-filter likelihoods where the transition density is intractable; the validation
  of every reference, including nested-sampling and tempered-SMC cross-checks, is described in
  the technical report.
- **Evaluation.** Accuracy is the Fréchet distance between the sampled and reference posteriors
  on frozen evaluation sets, complemented by simulation-based calibration; the same machinery is
  exposed for user-defined problems (`amortix.evaluation`).

## Technical report

Architecture, training recipe, evaluation methodology, the benchmark suite with its reference
posteriors, and measured training/inference costs are documented in the technical report:
[`report/techreport.pdf`](report/techreport.pdf).

## References

1. C. Sherki, I. Oseledets, E. Muravleva. *Bayesian Inverse Problems Meet Flow Matching:
   Efficient and Flexible Inverse Problem Solving*. [arXiv:2503.01375](https://arxiv.org/abs/2503.01375).
2. D. Fokina, M. Baldan, C. Romankiewicz, W. Laudensack, R. Ulber, M. Bortz. *Deep Learning for
   Model Calibration in Simulation of Itaconic Acid Production*.
   [arXiv:2604.22496](https://arxiv.org/abs/2604.22496).

## License

MIT.
