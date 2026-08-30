"""Shrunk, seed-fixed versions of the gallery examples.

Each test reproduces one script's inference path at a fraction of its budget
(pico model, n_train=1500, 400 optimizer steps) and asserts a statistical
property -- a floor or a ratio, never an exact number, because training is
stochastic across platforms.

    uv run pytest tests/test_examples.py -q
"""
from __future__ import annotations

import importlib.util
import os

import torch

from amortix.evaluation import fid, model_of_size
from amortix.problems.design_basic import gbm_exact_from_points

_GALLERY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "examples", "gallery")


def _example_module(name: str):
    """Import a gallery script as a module (their names start with digits)."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_GALLERY, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit_pico(prob, seed: int = 0):
    post = model_of_size(prob, "pico")
    post.fit(n_train=1500, steps=400, batch=256, seed=seed,
             retokenize=prob.make_retokenizer(), verbose=False)
    return post


def test_gbm_beats_prior_fid():
    """Example 01, shrunk: the trained posterior must sit at least 2x closer
    to the exact posterior (in FID) than prior samples do."""
    prob = _example_module("01_quickstart_gbm").GBM()
    post = _fit_pico(prob)
    gen = torch.Generator().manual_seed(1)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)
    tidx, cidx = prob.sample_design(gen, 20)
    tokens = prob.tokens_for(raw[0], tidx, cidx, gen)
    draws = post.sample(tokens, n=2000)
    exact = gbm_exact_from_points(prob, raw[0, :, 0], tidx, n_samples=2000)
    f_model = fid(draws.numpy(), exact)
    prior_draws = prob.prior.sample(2000, torch.Generator().manual_seed(2))
    f_prior = fid(prior_draws.numpy(), exact)
    assert f_model <= 0.5 * f_prior, (f_model, f_prior)


def test_oscillator_recovers_truth():
    """Example 04, shrunk: posterior mean near the truth (within 0.35 of the
    prior range per parameter) and posterior sd below half the prior range."""
    mod = _example_module("04_custom_problem")
    prob = mod.DampedOscillator()
    post = _fit_pico(prob)
    gen = torch.Generator().manual_seed(3)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)
    tidx, cidx = prob.sample_design(gen, 12)
    d = post.sample(prob.tokens_for(raw[0], tidx, cidx, gen), n=2000)
    prior_range = prob.prior.high - prob.prior.low
    err = (d.mean(0) - m_true[0]).abs()
    assert (err <= 0.35 * prior_range).all(), (err / prior_range).tolist()
    assert (d.std(0) < 0.5 * prior_range).all(), \
        (d.std(0) / prior_range).tolist()


def test_pk_design_size_monotonicity():
    """Example 02, shrunk: densifying the design from K=6 to K=50 must
    tighten the posterior for at least one parameter."""
    prob = _example_module("02_any_design_pk").PK()
    post = _fit_pico(prob)
    gen = torch.Generator().manual_seed(7)
    m_true = prob.prior.sample(1, gen)
    raw = prob.trajectories(m_true, gen)
    sd = {}
    for K in (6, 50):
        tidx, cidx = prob.sample_design(gen, K)
        tokens = prob.tokens_for(raw[0], tidx, cidx, gen)
        sd[K] = post.sample(tokens, n=2000).std(0)
    assert (sd[50] < sd[6]).any(), (sd[6].tolist(), sd[50].tolist())
