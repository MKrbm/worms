import numpy as np
from ..core.paulis.spin import *
from .. import utils
import logging
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

    # TODO: implement different lattice types such as dimer and triangular
    raise NotImplementedError("Negative lattice type not implemented")


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
    
    elif len(_L) == 2:
        Lx, Ly = _L
        logging.info(f"Lx      : {Lx}")
        logging.info(f"Ly      : {Ly}")
        logging.info(f"params : {params}")
        H_list, sps = local(params, D=2)

        bonds = []
        for i in range(Lx):
            for j in range(Ly):
                bonds.append([[i, j], [(i + 1) % Lx, j]])
                bonds.append([[i, j], [i, (j + 1) % Ly]])
        if len(H_list) != 1:
            raise RuntimeError("something wrong")
        _H = utils.sum_ham(H_list[0], bonds, Lx * Ly, sps)
        return _H, sps

    raise NotImplementedError("2D not implemented")
