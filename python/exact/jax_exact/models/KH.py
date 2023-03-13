# kagome heisenberg model

import sys,os
import numpy as np
from jax import numpy as jnp
import jax
# from ..core import *
from ..core.constants import *
from ..core.utils import *
sys.path.append('/home/user/project/python/reduce_nsp')
from nsp.utils import l2nl, sum_ham, cast_local_jax


def KH(_L : list[int], params : dict):
    L = np.array(_L)
    [L1, L2] = _L
    N = L1 * L2 * 3
    print(f"L : {L} (3 site per unit cell)")
    print(f"params : {params}")
    print(f"Kagome Heisenberg model ({N} sites)")
    if (len(L) != 2):
        raise ValueError(f"Only 2D model is supported")
    Jz = params["Jz"]
    Jx = params["Jx"]
    Jy = params["Jy"]
    hz = params["h"]
    hx = 0
    if (np.any(L <= 0)):
        raise ValueError(f"L = {L} is not valid")
    H_bond = Jz*SzSz + Jx*SxSx + Jy*SySy
    H_single = hz*Sz + hx*Sx
    H = np.zeros((2**N, 2**N))

    bonds = None
    if (L[0] == L[1] == 2):
        bonds = [ [ 0, 1, ], [ 0, 2, ], [ 1, 2, ], [ 0, 4, ], [ 1, 11, ], [ 0, 8, ], [ 3, 4, ], [ 3, 5, ], [ 4, 5, ], [ 3, 1, ], [ 4, 8, ], [ 3, 11, ], [ 6, 7, ], [ 6, 8, ], [ 7, 8, ], [ 6, 10, ], [ 7, 5, ], [ 6, 2, ], [ 9, 10, ], [ 9, 11, ], [ 10, 11, ], [ 9, 7, ], [ 10, 2, ], [ 9, 5, ], ]
    elif (L[0] == 3 and L[1] == 1):
        bonds = [ [ 0, 1, ], [ 0, 2, ], [ 1, 2, ], [ 0, 7, ], [ 1, 5, ], [ 0, 2, ], [ 3, 4, ], [ 3, 5, ], [ 4, 5, ], [ 3, 1, ], [ 4, 8, ], [ 3, 5, ], [ 6, 7, ], [ 6, 8, ], [ 7, 8, ], [ 6, 4, ], [ 7, 2, ], [ 6, 8, ], ]
    else:
        print("Not implemented")

    # for bond in bonds:
    # H += l2nl(bond_H, N, bond, sps=2)
    # _H = cast_local_jax(jnp.array(H_bond), bonds, N, 2)
    _H = sum_ham(H_bond, bonds, N, 2)
    if (hx != 0 or hz != 0):
        _H += sum_ham(H_single, [i for i in range(N)], N, 2)
    return jnp.array(_H)

    # solve eigenvalue problem with jax
