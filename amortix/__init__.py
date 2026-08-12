"""amortix -- amortized parameter recovery for dynamical systems via flow matching.

Core idea (arXiv:2503.01375): give a prior + simulator, get a fast amortized
posterior over parameters. A transformer set-encoder conditions a conditional
flow-matching velocity field; inference is a single ODE solve.

This is the SDE-recovery seed of the package.
"""
from __future__ import annotations

from .prior import BoxUniform
from .sde import SDEProblem, PathObserver, Channel, euler_maruyama
from .ode import ODEProblem, TimeSeriesObserver, rk4
from .encoder import SetTransformer
from .flow import FlowPosterior
from .diagnostics import diagnose, run_sbc, coverage_from_ranks
from .problems.ou import OrnsteinUhlenbeck
from .problems.seir import SEIRD
from .problems.gbm import GeometricBrownianMotion
from .problems.cir import CIR
from .problems.double_well import DoubleWell
from .problems.stoch_lv import StochasticLotkaVolterra
from .problems.fhn import FitzHughNagumo
from .problems.sindy_sde import PolynomialDriftSDE

from .designs import DesignProblem, DesignObserver, sbc_design
from .problems.design_zoo import DESIGN_ZOO

__all__ = [
    "BoxUniform",
    "SDEProblem",
    "PathObserver",
    "Channel",
    "euler_maruyama",
    "ODEProblem",
    "TimeSeriesObserver",
    "rk4",
    "SetTransformer",
    "FlowPosterior",
    "diagnose",
    "run_sbc",
    "coverage_from_ranks",
    "OrnsteinUhlenbeck",
    "SEIRD",
    "GeometricBrownianMotion",
    "CIR",
    "DoubleWell",
    "StochasticLotkaVolterra",
    "FitzHughNagumo",
    "PolynomialDriftSDE",
    "DesignProblem",
    "DesignObserver",
    "sbc_design",
    "DESIGN_ZOO",
]
