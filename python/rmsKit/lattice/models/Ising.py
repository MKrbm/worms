
# quantum ising model

import sys
import os
import numpy as np
from ..core.constants import *
from .. import utils
import rms
import logging

unitary_algorithms = ["original"]


def local(ua: str, params: dict):
    if ua not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")
    Jz = params["Jz"]
    hx = params["hx"]
    h_bond = -Jz*SzSz  # ZZ interaction term
    h_single = -hx*Sx  # transverse field term

    if ua == "original":
        h = h_bond + (np.kron(h_single, I2) + np.kron(I2, h_single)) / \
            2.0  # there is 2 site per bond
        sps = 2
        return [h], sps
    error_message = f"Other unitary_algorithms mode not implemented yet u = {ua}"
    raise NotImplementedError(error_message)
    return None, None


def system(_L: list[int], ua: str, params: dict) -> np.ndarray:
    if ua not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")

    if (len(_L) != 1):
        raise ValueError(f"Currently only 1D is supported")
    L = _L[0]
    logging.info(f"L      : {L} (1 site per unit cell)")
    logging.info(f"params : {params}")
    H = np.zeros((2**L, 2**L))
    H_list, sps = local(ua, params)

    if ua == "original":
        bonds = [[i, (i+1) % L] for i in range(L)]
        if (len(H_list) != 1):
            raise RuntimeError("something wrong")
        _H = rms.sum_ham(H_list[0], bonds, L, 2)
        return _H
    else:
        raise ValueError(f"ua = {ua} not supported")
