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
    assert len(GALLERY) == 8


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
