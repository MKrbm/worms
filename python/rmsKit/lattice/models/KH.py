# kagome heisenberg model

import sys, os
import numpy as np
from ..core.constants import *
from ..core.utils import *
import utils
import logging

unitary_algorithms = ["original", "3site", "3siteDiag"]


def ind2state(ind):
    state = []
    for i in range(4):
        state.append(ind % 8)
        ind //= 8
    return state[::-1]


def state2ind(state):
    num = 0
    for s in state:
        num *= 8
        num += s
    return num


P = np.zeros((4096, 4096))
for ind in range(len(P)):
    state_ = ind2state(ind)
    for i in range(4):
        state = state_.copy()
        for x in range(8):
            state[i] = x
            ind_ = state2ind(state)
            P[ind, ind_] = 1
    P[ind, ind] = 0


def local(lt: str, params: dict):
    """
    params
    ------
    lt : str 
        lattice type 
    params : dict
    """
    if lt not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")
    Jz = params["Jz"]
    Jx = params["Jx"]
    Jy = params["Jy"]
    hz = params["hz"]
    hx = params["hx"]
    h_bond = Jz * SzSz + Jx * SxSx + Jy * SySy
    h_single = hz * Sz + hx * Sx

    if lt == "original":
        h = h_bond + (np.kron(h_single, I2) + np.kron(I2, h_single)) / 4.0  # n* there is 4 bond per site
        sps = 2
        return [h], sps

    if lt == "3site":
        _h = utils.sum_ham(h_bond / 6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], 6, 2)
        _h += utils.sum_ham(h_single / 6, [[i] for i in range(6)], 6, 2)  # n* there is 6 bond per site

        h = [
            _h + utils.sum_ham(h_bond, [[0, 5]], 6, 2),
            _h + utils.sum_ham(h_bond, [[0, 4]], 6, 2),
            _h + utils.sum_ham(h_bond, [[2, 4]], 6, 2),
        ]
        return [_h for _h in h], 8
        # rewrite this with forloop

    if lt == "3siteDiag":
        _h0 = utils.sum_ham(h_bond, [[0, 1], [1, 2], [2, 0]], 3, 2)
        E, V = np.linalg.eigh(_h0)
        U = np.kron(V, V)
        _h = utils.sum_ham(h_bond / 6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], 6, 2)
        _h += utils.sum_ham(h_single / 6, [[i] for i in range(6)], 6, 2)  # n* there is 6 bond per site

        h = [
            _h + utils.sum_ham(h_bond, [[0, 5]], 6, 2),
            _h + utils.sum_ham(h_bond, [[0, 4]], 6, 2),
            _h + utils.sum_ham(h_bond, [[2, 4]], 6, 2),
        ]
        # h = [_h] * 3
        h = [U.T @ _h @ U for _h in h]

        # n* symmetrize
        h = [(_h + _h.T) / 2 for _h in h]
        return h, 8
        # rewrite this with forloop

    if lt == "6site":
        print("impossible")
    error_message = f"Other unitary_algorithms mode not implemented yet lattice type = {lt}"
    raise NotImplementedError(error_message)
    return None, None


def system(_L: list[int], lt: str, params: dict, separate: bool = False) -> np.ndarray:
    if lt not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")

    L = np.array(_L)
    [L1, L2] = _L
    N = L1 * L2 * 3
    # print(f"L      : {L} (3 site per unit cell)")
    # print(f"params : {params}")
    # print(f"Kagome Heisenberg model ({N} sites)")
    logging.info(f"L      : {L} (3 site per unit cell)")
    logging.info(f"params : {params}")
    logging.info(f"Kagome Heisenberg model ({N} sites)")
    if len(L) != 2:
        raise ValueError(f"Only 2D model is supported")
    if N > 12:
        raise ValueError(f"Only N <= 12 is supported")
    H = np.zeros((2**N, 2**N))
    H_list, sps = local(lt, params)

    if lt == "original":
        bonds = None
        if L[0] == L[1] == 2:
            bonds = [
                [
                    0,
                    1,
                ],
                [
                    0,
                    2,
                ],
                [
                    1,
                    2,
                ],
                [
                    0,
                    4,
                ],
                [
                    1,
                    11,
                ],
                [
                    0,
                    8,
                ],
                [
                    3,
                    4,
                ],
                [
                    3,
                    5,
                ],
                [
                    4,
                    5,
                ],
                [
                    3,
                    1,
                ],
                [
                    4,
                    8,
                ],
                [
                    3,
                    11,
                ],
                [
                    6,
                    7,
                ],
                [
                    6,
                    8,
                ],
                [
                    7,
                    8,
                ],
                [
                    6,
                    10,
                ],
                [
                    7,
                    5,
                ],
                [
                    6,
                    2,
                ],
                [
                    9,
                    10,
                ],
                [
                    9,
                    11,
                ],
                [
                    10,
                    11,
                ],
                [
                    9,
                    7,
                ],
                [
                    10,
                    2,
                ],
                [
                    9,
                    5,
                ],
            ]
        elif L[0] == 3 and L[1] == 1:
            bonds = [
                [
                    0,
                    1,
                ],
                [
                    0,
                    2,
                ],
                [
                    1,
                    2,
                ],
                [
                    0,
                    7,
                ],
                [
                    1,
                    5,
                ],
                [
                    0,
                    2,
                ],
                [
                    3,
                    4,
                ],
                [
                    3,
                    5,
                ],
                [
                    4,
                    5,
                ],
                [
                    3,
                    1,
                ],
                [
                    4,
                    8,
                ],
                [
                    3,
                    5,
                ],
                [
                    6,
                    7,
                ],
                [
                    6,
                    8,
                ],
                [
                    7,
                    8,
                ],
                [
                    6,
                    4,
                ],
                [
                    7,
                    2,
                ],
                [
                    6,
                    8,
                ],
            ]
        else:
            print("Not implemented")
        if len(H_list) != 1:
            raise RuntimeError("something wrong")

        H_bond = H_list[0]
        _H = utils.sum_ham(H_bond, bonds, N, 2)
        return _H

    if lt == "3site" or lt == "3siteDiag":
        _bonds_prime = [[1, 1, 0], [0, 2, 0], [2, 2, 1], [0, 3, 1], [1, 3, 2], [2, 3, 0]]
        bonds = [[], [], []]
        for bond in _bonds_prime:
            bonds[bond[0]].append(bond[1:])
            bonds[bond[0]].append(bond[1:][::-1])

        bonds_prime1 = [[], [], []]
        bonds_prime2 = [[], [], []]
        for bond in _bonds_prime:
            bonds_prime1[bond[0]].append(bond[1:])
            bonds_prime2[bond[0]].append(bond[1:][::-1])

        if len(H_list) != 3:
            raise RuntimeError("something wrong")
        if separate == True:
            _H_list = []
            _H = utils.sum_ham(H_list[0], bonds_prime1[0], 4, 8)
            _H += utils.sum_ham(H_list[1], bonds_prime1[1], 4, 8)
            _H += utils.sum_ham(H_list[2], bonds_prime1[2], 4, 8)
            _H_list.append(_H)
            _H = utils.sum_ham(H_list[0], bonds_prime2[0], 4, 8)
            _H += utils.sum_ham(H_list[1], bonds_prime2[1], 4, 8)
            _H += utils.sum_ham(H_list[2], bonds_prime2[2], 4, 8)
            _H_list.append(_H)
            return np.array(_H_list), P
        else:
            _H = utils.sum_ham(H_list[0], bonds[0], 4, 8)
            _H += utils.sum_ham(H_list[1], bonds[1], 4, 8)
            _H += utils.sum_ham(H_list[2], bonds[2], 4, 8)
            return _H

    else:
        raise ValueError(f"lattice type = {lt} not supported")
