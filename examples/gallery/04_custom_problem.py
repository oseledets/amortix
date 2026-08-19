"""Add your own system in ~25 lines: a damped oscillator, observed anywhere.

Subclass DesignProblem, provide a prior and a trajectories() simulator, and
everything else -- fresh-design training, tokenization, any-K inference --
comes from the base class. ~2 minutes on CPU.

Run:  python examples/gallery/04_custom_problem.py
"""
import torch

from amortix.designs import DesignObserver, DesignProblem
from amortix.evaluation import model_of_size
from amortix.prior import BoxUniform


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


prob = DampedOscillator()
post = model_of_size(prob, "pico")
post.fit(n_train=3000, steps=1200, batch=256,
         retokenize=prob.make_retokenizer(), verbose=True)

gen = torch.Generator().manual_seed(3)
m_true = prob.prior.sample(1, gen)
raw = prob.trajectories(m_true, gen)
tidx, cidx = prob.sample_design(gen, 12)
d = post.sample(prob.tokens_for(raw[0], tidx, cidx, gen), n=2000)
print(f"\ntrue (omega, gamma): {m_true[0].tolist()}")
print(f"posterior mean     : {d.mean(0).tolist()}  sd {d.std(0).tolist()}")
