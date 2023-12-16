import numpy as np
from ..core.paulis.spin1 import SzSz, SxSx, SySy, Sz, Sx, I3
import logging
from ..core import utils
from typing import List, Any, Tuple, Dict
from numpy._typing import NDArray
unitary_algorithms = ["original"]


def local(params: Dict[str, Any], D: int = 1) -> Tuple[List[NDArray[Any]], int]:
    J0 = params["J0"]
    J1 = params["J1"]
    hz = params["hz"]
    hx = params["hx"]
    lt = params["lt"]
    h_heis = (SzSz + SxSx + SySy)
    h_single = hz * Sz + hx * Sx

    if D != 1:
        raise ValueError("BLBQ is 1D model")

    h = J1 * h_heis @ h_heis + J0 * h_heis + \
        (np.kron(h_single, I3) + np.kron(I3, h_single)) / 2

    e, v = np.linalg.eigh(h)

    if lt > 0:
        sps = 3
        if lt > 1:
            L = lt
            _h = utils.sum_ham(h / 2, [[i, i+1] for i in range(2*L-1)], 2 * L, sps)
            _h += utils.sum_ham(h / 2, [[L-1, L]], 2 * L, sps)
            h = _h
            sps = sps**L
        return [h], sps

    raise NotImplementedError("Negative lattice type not implemented")


def system(_L: list[int], params: dict) -> Tuple[NDArray[Any], int]:

    if len(_L) != 1:
        raise ValueError("BLBQ is 1D model")
    L = _L[0]
    P = {
        "J0": params["J0"],
        "J1": params["J1"],
        "hz": params["hz"],
        "hx": params["hx"],
        "lt": params["lt"],
    }
    logging.info(f"params : {P}")
    logging.info(f"L      : {L}")

    H_list, sps = local(P, D=1)

    bonds = [[i, (i + 1) % L] for i in range(L)]

    if len(H_list) != 1:
        raise RuntimeError("Local Hamilonian bust contain only one term")

    _H = utils.sum_ham(H_list[0], bonds, L, sps)
    return _H, sps
