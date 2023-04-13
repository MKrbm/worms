# kagome heisenberg model

import sys,os
import numpy as np
from ..core.constants import *
from ..core.utils import *
import rms
import logging
unitary_algorithms = ["original", "3site", "3siteDiag"]


def local(ua : str, params : dict):
    if ua not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")
    Jz = params["Jz"]
    Jx = params["Jx"]
    Jy = params["Jy"]
    hz = params["hz"]
    hx = params["hx"]
    h_bond = Jz*SzSz + Jx*SxSx + Jy*SySy 
    h_single = hz*Sz + hx*Sx

    if ua == "original":
        h = h_bond + (np.kron(h_single, I2) +  np.kron(I2, h_single)) / 4.0 #n* there is 4 bond per site
        sps = 2
        return [h], sps
    error_message = f"Other unitary_algorithms mode not implemented yet u = {ua}"
    raise NotImplementedError(error_message)
    return None, None



def system(_L : list[int], ua : str, params : dict) -> np.ndarray:
    if ua not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")
    
    if (len(_L) != 1):
        raise ValueError(f"L must be 1D array")
    L = _L[0]
    logging.info(f"L      : {L} (3 site per unit cell)")
    logging.info(f"params : {params}")
    H = np.zeros((2**L, 2**L))
    H_list, sps = local(ua, params)

    if ua == "original":
        bonds = [[i, (i+1)%L] for i in range(L)]
        if (len(H_list) != 1):
            raise RuntimeError("something wrong")
        _H = rms.sum_ham(H_list[0], bonds, L, 2)
        return _H 
    else:
        raise ValueError(f"ua = {ua} not supported")