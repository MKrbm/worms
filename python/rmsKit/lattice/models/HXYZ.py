import numpy as np
from ..core.constants import *
from ..core.utils import *
import logging
import utils
from typing import List, Any, Tuple, Dict
from numpy._typing import NDArray
unitary_algorithms = ["original", "2site", "3siteDiag"]


def local(params: Dict[str, Any], D: int = 1) -> Tuple[List[NDArray[Any]], int]:
    Jz = params["Jz"]
    Jx = params["Jx"]
    Jy = params["Jy"]
    hz = params["hz"]
    hx = params["hx"]
    lt = params["lt"]  # lattice type
    h_bond = Jz * SzSz + Jx * SxSx + Jy * SySy
    h_single = hz * Sz + hx * Sx

    if lt > 0:
        h = h_bond + (np.kron(h_single, I2) + np.kron(I2, h_single)) / \
            (2 * D)  # n* there is 4 bond per site
        sps = 2
        if lt > 1:
            L = lt
            _h = utils.sum_ham(h / 2, [[i, i+1] for i in range(2*L-1)], 2 * L, sps)
            _h += utils.sum_ham(h / 2, [[L-1, L]], 2 * L, sps)
            h = _h
            sps = sps**L
        return [h], sps

    raise NotImplementedError("Negative lattice type not implemented") # TODO: implement different lattice types such as dimer and triangular


def system(_L: list[int], params: dict) -> Tuple[NDArray[Any], int]:

    if len(_L) == 1:
        L = _L[0]
        logging.info(f"L      : {L}")
        logging.info(f"params : {params}")
        H_list, sps = local(params, D=1)

        bonds = [[i, (i + 1) % L] for i in range(L)]
        if len(H_list) != 1:
            raise RuntimeError("something wrong")
        _H = utils.sum_ham(H_list[0], bonds, L, sps)
        return _H, sps

        # else: # 2D
        #     raise ValueError(f"ua = {ua} not supported")
        # L = _L[0] * _L[1]
        # logging.info(f"L      : {L} (3 site per unit cell)")
        # logging.info(f"params : {params}")
        # H_list, sps = local(ua, params, 2)  # 2 dimensional
        #
        # if ua == "original":
        #     s = np.arange(L)
        #     x = s % _L[0]
        #     y = s // _L[0]
        #     T_x = (x + 1) % _L[0] + y * _L[0]
        #     T_y = x + ((y + 1) % _L[1]) * _L[0]
        #     bonds = [[i, T_x[i]] for i in range(L)] + [[i, T_y[i]] for i in range(L)]
        #     print(bonds)
        #     if len(H_list) != 1:
        #         raise RuntimeError("something wrong")
        #     _H = rms.sum_ham(H_list[0], bonds, L, 2)
        #     return _H
        # else:
        #     raise ValueError(f"ua = {ua} not supported")
    raise NotImplementedError("2D not implemented")
