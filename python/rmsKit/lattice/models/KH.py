"""kagome heisenberg model."""
import numpy as np
from ..core.paulis.spin import Sx, Sz, I2, SxSx, SySy, SzSz
from .. import utils
from typing import List, Any, Tuple, Dict
from numpy._typing import NDArray
import logging


# def ind2state(ind):
#     state = []
#     for i in range(4):
#         state.append(ind % 8)
#         ind //= 8
#     return state[::-1]
#
#
# def state2ind(state):
#     num = 0
#     for s in state:
#         num *= 8
#         num += s
#     return num
#
#
# P = np.zeros((4096, 4096))
# for ind in range(len(P)):
#     state_ = ind2state(ind)
#     for i in range(4):
#         state = state_.copy()
#         for x in range(8):
#             state[i] = x
#             ind_ = state2ind(state)
#             P[ind, ind_] = 1
#     P[ind, ind] = 0


def local(params: Dict[str, Any], D: int = 2) -> Tuple[List[NDArray[Any]], int]:
    """Generate the local Hamiltonian of the Kagome Heisenberg model.

    params
    ------
    lt : str
    lattice type
    params : dict
    """
    Jz = params["Jz"]
    Jx = params["Jx"]
    Jy = params["Jy"]
    hz = params["hz"]
    hx = params["hx"]
    lt = params["lt"]  # lattice type
    h_bond = Jz * SzSz + Jx * SxSx + Jy * SySy
    h_single = hz * Sz + hx * Sx
    if D != 2:
        raise ValueError("Kagome Heisenberg model is 2D model")

    if lt == 1:
        h = h_bond + (np.kron(h_single, I2) + np.kron(I2, h_single)) / \
            4.0  # n* there is 4 bond per site
        sps = 2
        return [h], sps

    elif lt == 3:
        _h = utils.sum_ham(h_bond / 6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], 6, 2)
        _h += utils.sum_ham(h_single / 6, [[i] for i in range(6)],
                            6, 2)  # n* there is 6 bond per site

        # h = [
        #     _h + utils.sum_ham(h_bond, [[0, 5]], 6, 2),
        #     _h + utils.sum_ham(h_bond, [[0, 4]], 6, 2),
        #     _h + utils.sum_ham(h_bond, [[2, 4]], 6, 2),
        # ]
        h = [
            _h + utils.sum_ham(h_bond, [[2, 3]], 6, 2),
            _h + utils.sum_ham(h_bond, [[1, 3]], 6, 2),
            _h + utils.sum_ham(h_bond, [[2, 4]], 6, 2),
        ]
        sps = 8
        return [_h for _h in h], sps
        # rewrite this with forloop
    else:
        error_message = f"Other lattice type mode not implemented yet lattice type = {lt}"
        raise NotImplementedError(error_message)
    return None, None


def system(_L: list[int], params: dict) -> Tuple[NDArray[Any], int]:

    L = np.array(_L)
    [L1, L2] = _L
    N = L1 * L2 * 3
    if L1 != 2 or L2 != 2:
        raise ValueError("Kagome Heisenberg model only accept L1 = L2 = 2")
    logging.info(f"L      : {L} (3 site per unit cell)")
    logging.info(f"params : {params}")
    logging.info(f"Kagome Heisenberg model ({N} sites)")
    if len(L) != 2:
        raise ValueError("Only 2D model is supported")
    if N > 12:
        raise ValueError("Only N <= 12 is supported")
    H_list, sps = local(params, 2)
    lt = params["lt"]
    if lt == 3:
        _bonds_prime = [[1, 1, 0], [0, 2, 0], [2, 2, 1], [0, 3, 1], [1, 3, 2], [2, 3, 0]]
        bonds: List[List[List[int]]] = [[], [], []]
        for bond in _bonds_prime:
            bonds[bond[0]].append(bond[1:])
            bonds[bond[0]].append(bond[1:][::-1])

        if len(H_list) != 3:
            raise RuntimeError("something wrong")
        _H = utils.sum_ham(H_list[0], bonds[0], 4, 8)
        _H += utils.sum_ham(H_list[1], bonds[1], 4, 8)
        _H += utils.sum_ham(H_list[2], bonds[2], 4, 8)
        return _H, sps
    else:
        raise ValueError(f"lattice type = {lt} not supported")
