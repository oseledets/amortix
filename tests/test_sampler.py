"""The sampler's settings are explicit, and the time grid resolves narrow posteriors.

Checked here:
  * sample / sample_batch refuse a call without n, n_steps, solver, t_grid;
  * flow.time_grid covers [0, 1]; flow.integrate_flow with t_grid="uniform"
    returns bit-identical samples to the sampler of commit 12b20f4 (the last
    one with the uniform grid built in), for every solver, the data base and
    the variable-length path;
  * on the EXACT flow-matching field of a correlated Gaussian target of width
    s << 1 (prior width 1) the uniform grid of 20 midpoint steps returns the
    wrong joint structure, and the late grid resolves it at the same cost;
  * DesignProblem.value_coord is required, and a log-normal-noise problem
    must declare "log".

    uv run pytest tests/test_sampler.py -q
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys

import pytest
import torch

from amortix import FlowPosterior, OrnsteinUhlenbeck
from amortix.designs import DesignObserver, DesignProblem, tokens_from_data
from amortix.flow import integrate_flow, time_grid
from amortix.prior import BoxUniform
from amortix.problems.linear_gaussian import LinearGaussian
from amortix.problems.design_basic import GBMDesign
from amortix.problems.design_zoo import PharmacoKineticsDesign

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE_GRID_COMMIT = "12b20f4"


def test_time_grid_covers_the_unit_interval():
    for kind in ("uniform", "late"):
        g = time_grid(20, kind)
        assert len(g) == 20 and g[0][0] == 0.0
        assert all(abs(g[i][0] + g[i][1] - g[i + 1][0]) < 1e-12 for i in range(19))
        assert abs(g[-1][0] + g[-1][1] - 1.0) < 1e-12
        assert all(h > 0 for _, h in g)
    assert time_grid(20, "uniform")[-1][1] == pytest.approx(1 / 20)
    assert time_grid(20, "late")[-1][1] == pytest.approx(1 / 8000)
    with pytest.raises(ValueError, match="t_grid"):
        time_grid(5, "geometric")


def _tiny(flow_module, prob, **kw):
    torch.manual_seed(0)
    post = flow_module.FlowPosterior(prob, dim_model=16, depth=2, **kw)
    post.fit(n_train=64, steps=3, batch=32, seed=1, verbose=False, device="cpu")
    return post.eval()


def test_sampler_settings_are_required():
    prob = LinearGaussian()
    import amortix.flow as flow
    post = _tiny(flow, prob)
    tok = prob.simulate(2, torch.Generator().manual_seed(5))[1]
    full = dict(n=8, n_steps=4, solver="midpoint", t_grid="late")
    assert post.sample_batch(tok, **full).shape == (2, 8, 4)
    assert post.sample(tok[0], **full).shape == (8, 4)
    for missing in full:
        kw = {k: v for k, v in full.items() if k != missing}
        with pytest.raises(TypeError):
            post.sample_batch(tok, **kw)
        with pytest.raises(TypeError):
            post.sample(tok[0], **kw)
    with pytest.raises(ValueError, match="t_grid"):
        post.sample_batch(tok, n=4, n_steps=3, solver="midpoint", t_grid="geometric")
    with pytest.raises(ValueError, match="solver"):
        post.sample_batch(tok, n=4, n_steps=3, solver="heun", t_grid="late")


def _shadow_flow(ref):
    """flow.py at the git ref, imported as a shadow module inside the package."""
    try:
        src = subprocess.check_output(["git", "show", f"{ref}:amortix/flow.py"], cwd=REPO,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    tag = re.sub(r"\W", "_", ref)
    path = os.path.join(REPO, "tests", f"_flow_shadow_{tag}.py")
    with open(path, "wb") as f:
        f.write(src)
    try:
        spec = importlib.util.spec_from_file_location(f"amortix._flow_shadow_{tag}", path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "amortix"
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.remove(path)
    return mod


def test_uniform_grid_is_bit_identical_to_the_sampler_before_the_grid():
    old = _shadow_flow(PRE_GRID_COMMIT)
    if old is None:
        pytest.skip(f"flow.py of {PRE_GRID_COMMIT} not available through git")
    import amortix.flow as flow
    prob = LinearGaussian()
    tok = prob.simulate(3, torch.Generator().manual_seed(5))[1]
    for ctor in ({}, dict(base="data"), dict(conditioning="concat")):
        for solver in ("euler", "midpoint", "rk4"):
            a = _tiny(old, prob, **ctor).sample_batch(tok, n=64, n_steps=7, seed=2, solver=solver)
            b = _tiny(flow, prob, **ctor).sample_batch(tok, n=64, n_steps=7, seed=2, solver=solver,
                                                      t_grid="uniform")
            assert torch.equal(a, b), (ctor, solver)
    ou = OrnsteinUhlenbeck()
    tt = ou.simulate(2, torch.Generator().manual_seed(1))[1]
    a = _tiny(old, ou).sample_batch([tt[0], tt[1][:40]], n=16, n_steps=6, seed=4)
    b = _tiny(flow, ou).sample_batch([tt[0], tt[1][:40]], n=16, n_steps=6, seed=4, solver="midpoint",
                                     t_grid="uniform")
    assert torch.equal(a, b)


def _gauss_field(mu, S):
    """Exact CFM field of N(mu, S) from N(0, I) on the linear path, batched signature."""
    d = mu.numel()
    eye = torch.eye(d, dtype=mu.dtype)

    def f(z, t):
        tt = float(t[0])
        Vt = tt * tt * S + (1 - tt) ** 2 * eye
        B = torch.linalg.solve(Vt.T, (tt * S - (1 - tt) * eye).T).T
        return mu + (z - tt * mu) @ B.T
    return f


def _target(width):
    g = torch.Generator().manual_seed(3)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=torch.float64))
    lam = torch.tensor([1.0, 0.3, 0.06, 0.01], dtype=torch.float64)
    R = (Q * lam[None]) @ Q.T
    dd = R.diag().sqrt()
    R = R / dd[:, None] / dd[None, :]
    D = torch.diag(width * torch.tensor([0.6, 1.4, 0.9, 1.1], dtype=torch.float64))
    return torch.tensor([0.8, -0.5, 0.3, -1.0], dtype=torch.float64), D @ R @ D


def _errors(z, mu, S):
    a, Q = torch.linalg.eigh(S)
    W = (Q * a.rsqrt()[None]) @ Q.T
    M = W @ torch.cov(z.T) @ W
    shift = ((z.mean(0) - mu) / S.diag().sqrt()).abs().max().item()
    return shift, torch.linalg.matrix_norm(M - torch.eye(4, dtype=M.dtype), ord=2).item()


@pytest.mark.parametrize("width", [0.03, 0.01])
def test_late_grid_resolves_a_narrow_target_where_the_uniform_grid_does_not(width):
    mu, S = _target(width)
    f = _gauss_field(mu, S)
    z0 = torch.randn(1, 20000, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    shift_u, cov_u = _errors(integrate_flow(f, z0, 20, "midpoint", "uniform")[0], mu, S)
    shift_l, cov_l = _errors(integrate_flow(f, z0, 20, "midpoint", "late")[0], mu, S)
    # Monte-Carlo floor of the covariance estimate at 20000 draws: ~0.03
    assert cov_l < 0.1 and shift_l < 0.1
    assert cov_u > 5 * cov_l


def test_value_coord_is_required_and_consistent_with_the_noise():
    class NoCoord(DesignProblem):
        def __init__(self):
            self.prior = BoxUniform(low=[0.0], high=[1.0])
            self.observer = DesignObserver(dt_sim=0.1, n_steps=10, k_max=4)

        def trajectories(self, m, generator=None):
            return torch.ones(m.shape[0], 11, 1)

    class RawWithLogNormal(NoCoord):
        value_coord = "raw"
        LOGSD = 0.1

    gen = torch.Generator().manual_seed(0)
    for cls, msg in ((NoCoord, "value_coord"), (RawWithLogNormal, "LOGSD")):
        prob = cls()
        raw = prob.trajectories(torch.zeros(1, 1))
        with pytest.raises(ValueError, match=msg):
            prob.tokens_for(raw[0], torch.tensor([1, 2]), torch.tensor([0, 0]), gen)
        with pytest.raises(ValueError, match=msg):
            prob.make_retokenizer()(raw, gen)
        with pytest.raises(ValueError, match=msg):
            tokens_from_data(prob, [0.1, 0.2], [1.0, 1.0])
    # the declared coordinate reaches slot 1
    pk = PharmacoKineticsDesign()
    tok = tokens_from_data(pk, [1.0, 2.0], [10.0, 100.0])
    assert torch.allclose(tok[:, 1], torch.log(torch.tensor([10.0, 100.0])))
    gbm = GBMDesign()
    tok = tokens_from_data(gbm, [1.0, 2.0], [10.0, 100.0])
    assert torch.allclose(tok[:, 1], torch.tensor([10.0, 100.0]))
