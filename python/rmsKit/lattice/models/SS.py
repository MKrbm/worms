"""Shastry-Sutherland model.

Normaly this model is next nearest neighbor interaction on a square lattice.
"""
import numpy as np
from ..core.paulis.spin import *
from .. import utils
import logging
from typing import List, Any, Tuple, Dict
from numpy._typing import NDArray


def local(params: Dict[str, Any], D: int = 1) -> Tuple[List[NDArray[Any]], int]:
    """Generate the local Hamiltonian of the Shastry-Sutherland model."""
    J0 = params["J0"]
    J1 = params["J1"]
    # TODO: J2 = params["J2"]
    hx = params["hx"]
    lt = params["lt"]  # lattice type
    if lt != 1:
        raise ValueError("Shastry-Sutherland only accept lattice type 1")
    h_bond = SzSz + SxSx + SySy
    h_single = hx * Sx

    # n: for bond type 1.

    bond = [[]]



