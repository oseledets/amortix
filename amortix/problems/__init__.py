from __future__ import annotations

from .ou import OrnsteinUhlenbeck
from .seir import SEIRD
from .gbm import GeometricBrownianMotion
from .cir import CIR
from .double_well import DoubleWell
from .stoch_lv import StochasticLotkaVolterra
from .fhn import FitzHughNagumo
from .sindy_sde import PolynomialDriftSDE
from .linear_gaussian import LinearGaussian

# gallery registry: name -> module path (each exposes make/sota/SOTA_NAME)
GALLERY = [
    "linear_gaussian",
    "ou", "seir", "gbm", "cir", "double_well", "stoch_lv", "fhn", "sindy_sde",
]

__all__ = [
    "OrnsteinUhlenbeck", "SEIRD", "GeometricBrownianMotion", "CIR",
    "DoubleWell", "StochasticLotkaVolterra", "FitzHughNagumo",
    "PolynomialDriftSDE", "LinearGaussian", "GALLERY",
]
