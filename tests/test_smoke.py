"""Smoke tests: every gallery case must build, simulate, train, sample, and have
a working classical baseline. Fast by design (tiny budgets) -- this verifies the
install and the Problem contract, not statistical quality.

    uv run pytest -q
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from amortix import FlowPosterior
from amortix.problems import GALLERY


@pytest.fixture(scope="module")
def gen():
    return torch.Generator().manual_seed(0)


def test_gallery_is_populated():
    assert len(GALLERY) == 9


@pytest.mark.parametrize("name", GALLERY)
def test_case_contract(name, gen):
    """make() / simulate / observe / sota must satisfy the documented contract."""
    mod = importlib.import_module(f"amortix.problems.{name}")
    prob = mod.make()
    d = prob.prior.dim

    assert isinstance(mod.SOTA_NAME, str) and mod.SOTA_NAME
    assert len(prob.prior.names) == d

    m, tokens = prob.simulate(8, generator=gen)
    assert m.shape == (8, d)
    assert tokens.shape[0] == 8 and tokens.shape[2] == prob.observer.N_FEATURES
    assert torch.isfinite(tokens).all(), "simulator produced non-finite tokens"

    tk, traj = prob.observe(prob.prior.sample(4, generator=gen), generator=gen)
    assert torch.isfinite(traj).all(), "simulator produced non-finite trajectory"

    est = np.asarray(mod.sota(tk[0].numpy(), traj[0].numpy(), prob))
    assert est.shape == (d,), f"sota must return one value per parameter"
    assert np.all(np.isfinite(est)), "sota returned non-finite values"


@pytest.mark.parametrize("name", GALLERY)
def test_train_and_sample(name):
    """A tiny fit must run and produce in-prior posterior samples."""
    mod = importlib.import_module(f"amortix.problems.{name}")
    prob = mod.make()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    _, tokens = prob.simulate(2)
    s = post.sample(tokens[0], n=32, n_steps=10)
    assert s.shape == (32, prob.prior.dim)
    assert torch.isfinite(s).all()
    # probit denormalization must keep every sample inside the prior box
    assert (s >= prob.prior.low - 1e-4).all() and (s <= prob.prior.high + 1e-4).all()


@pytest.mark.parametrize("conditioning", ["concat", "xattn"])
def test_both_conditionings_run(conditioning):
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob, conditioning=conditioning)
    assert post.conditioning == conditioning
    post.fit(n_train=128, epochs=1, verbose=False)
    s = post.sample(prob.simulate(2)[1][0], n=16, n_steps=10)
    assert s.shape == (16, 3)


def test_xattn_is_default():
    """The calibration study settled on dense cross-attention conditioning."""
    from amortix import OrnsteinUhlenbeck
    assert FlowPosterior(OrnsteinUhlenbeck()).conditioning == "xattn"


def test_prior_probit_roundtrip():
    """probit normalize/denormalize must round-trip and map the prior to N(0,1)."""
    from amortix import OrnsteinUhlenbeck
    prior = OrnsteinUhlenbeck().prior
    m = prior.sample(20000, generator=torch.Generator().manual_seed(0))
    z = prior.normalize(m)
    assert torch.allclose(prior.denormalize(z), m, atol=1e-4)
    assert z.mean().abs() < 0.05, "normalized prior should be zero-mean"
    assert abs(float(z.std()) - 1.0) < 0.05, "normalized prior should be unit-variance"


def test_diagnostics_runs():
    """SBC harness must produce ranks, coverage and uniformity p-values."""
    from amortix import OrnsteinUhlenbeck
    from amortix.diagnostics import run_sbc, coverage_from_ranks, sbc_uniformity
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    res = run_sbc(post, prob, n_sims=12, n_post=20)
    assert res["ranks"].shape == (12, 3)
    assert (res["ranks"] >= 0).all() and (res["ranks"] <= 20).all()
    cov = coverage_from_ranks(res["ranks"], 20, (0.5, 0.9))
    assert set(cov) == {0.5, 0.9} and cov[0.9].shape == (3,)
    p = sbc_uniformity(res["ranks"], 20)
    assert p.shape == (3,) and np.all((p >= 0) & (p <= 1))


def test_sde_vector_state_and_correlated_noise():
    """Vector-state Euler-Maruyama with a correlation Cholesky must run."""
    from amortix import euler_maruyama
    chol = torch.linalg.cholesky(torch.tensor([[1.0, -0.5], [-0.5, 1.0]]))
    m = torch.ones(4, 1)
    traj = euler_maruyama(lambda x, mm: -x, lambda x, mm: torch.ones_like(x),
                          torch.zeros(4, 2), m, dt=0.01, n_steps=20, corr_chol=chol)
    assert traj.shape == (4, 21, 2) and torch.isfinite(traj).all()


def test_sample_batch_matches_single():
    """Batched sampling must agree with the single-dataset path."""
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    tk, _ = prob.observe(prob.prior.sample(3, generator=torch.Generator().manual_seed(1)))
    batch = post.sample_batch(tk, n=32, seed=5, n_steps=10)
    single = post.sample_batch(tk[:1], n=32, seed=5, n_steps=10)
    assert batch.shape == (3, 32, prob.prior.dim)
    assert torch.allclose(batch[0], single[0], atol=1e-5)


@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
def test_solvers_agree(solver):
    """All ODE solvers must produce finite, in-prior samples."""
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    s = post.sample(prob.simulate(2)[1][0], n=32, n_steps=10, solver=solver)
    assert torch.isfinite(s).all()
    assert (s >= prob.prior.low - 1e-4).all() and (s <= prob.prior.high + 1e-4).all()


def test_cli_cases_runs(capsys):
    """The console-script entry point must work."""
    from amortix.cli import main
    main(["cases"])
    out = capsys.readouterr().out
    for name in GALLERY:
        assert name in out


def test_base_head_gets_no_gradient_from_cfm():
    """The base must be trained by its own NLL only.

    If z0 were not detached, the CFM loss could be reduced by dragging the base
    onto the target -- paying the base to absorb the flow's job. With
    base_weight=0 the base head must therefore receive exactly zero gradient.
    """
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob, base="data")
    post.fit(n_train=256, epochs=1, base_weight=0.0, verbose=False)
    grads = [p.grad for p in post.base_head.parameters() if p.grad is not None]
    assert grads, "base head should still take part in the graph"
    assert all(float(g.abs().max()) == 0.0 for g in grads), \
        "CFM loss leaked into the base head (z0 not detached)"


def test_velocity_couples_parameters():
    """The velocity field must NOT be factorized across parameters.

    An ODE whose velocity for parameter i depends only on z_i maps independent
    coordinates to independent coordinates, so it can never turn an independent
    base into a correlated posterior -- no matter the budget. This regression
    guards the parameter self-attention that makes the field coupled.
    """
    from amortix.problems.linear_gaussian import make
    prob = make()
    d = prob.prior.dim
    post = FlowPosterior(prob, conditioning="xattn")
    post.fit(n_train=128, epochs=1, verbose=False)
    memory = post.encoder.encode(prob.simulate(1)[1])
    cache = post.velocity.encode_memory(memory)
    z = torch.zeros(1, d, requires_grad=True)
    v = post.velocity(z, torch.full((1,), 0.5), cache)
    J = torch.stack([torch.autograd.grad(v[0, i], z, retain_graph=True)[0][0]
                     for i in range(d)])
    off = (J - torch.diag(torch.diag(J))).abs().max()
    assert float(off) > 1e-8, "velocity is factorized across parameters"


def test_posterior_samples_are_independent():
    """Parameter self-attention must act within a sample, never across samples."""
    from amortix.problems.linear_gaussian import make
    prob = make()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    tk, _ = prob.observe(prob.prior.sample(1, generator=torch.Generator().manual_seed(0)))
    a = post.sample_batch(tk, n=8, seed=3, n_steps=10)
    b = post.sample_batch(tk, n=64, seed=3, n_steps=10)
    # the first 8 draws must not change when 56 more are drawn alongside them
    assert torch.allclose(a[0], b[0, :8], atol=1e-5), "samples interact with each other"
