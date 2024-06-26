"""Shastry-Sutherland model.

Normaly this model is next nearest neighbor interaction on a square lattice.
"""
import numpy as np
from ..core.paulis.spin import SzSz, SxSx, SySy
from .. import utils
import logging
from typing import List, Any, Tuple, Dict
from numpy._typing import NDArray


def local(params: Dict[str, Any], D: int = 1) -> Tuple[List[NDArray[Any]], int]:
    """Generate the local Hamiltonian of the Shastry-Sutherland model."""
    J0 = params["J0"]
    J1 = params["J1"]
    J2 = params["J2"]
    # TODO: hx = params["hx"]
    lt = params["lt"]  # lattice type
    h_bond = SzSz + SxSx + SySy
    # TODO: h_single = hx * Sx

    lh = h_bond

    # n: for bond type 1.
    H1 = utils.sum_ham(J1*lh, [[1, 2], [1, 3]], 4, 2)
    H1 += utils.sum_ham(J2*lh, [[0, 2], [0, 3]], 4, 2)
    H1 += utils.sum_ham(J0*lh/4, [[0, 1], [2, 3]], 4, 2)

    H2 = utils.sum_ham(J1*lh, [[0, 2], [1, 2]], 4, 2)
    H2 += utils.sum_ham(J2*lh, [[0, 3], [1, 3]], 4, 2)
    H2 += utils.sum_ham(J0*lh/4, [[0, 1], [2, 3]], 4, 2)

    sps = 4

    u_dimer = np.array([
        [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        [1, 0, 0, 0],
        [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
        [0, 0, 0, 1],
    ])

    Ud = np.kron(u_dimer, u_dimer)

    if lt == 1:
        logging.info("Lattice type: 1 is used (diagonal unit)")
        return [H1, H2], sps
    elif lt == -1:
        logging.info("Lattice type: -1 is used (dimer unit)")
        return [Ud @ H1 @ Ud.T, Ud @ H2 @ Ud.T], sps
    else:
        raise ValueError("Shastry-Sutherland only accept lattice type 1")


def system(_L: list[int], params: dict) -> Tuple[NDArray[Any], int]:
    """Generate the Hamiltonian of the Shastry-Sutherland model.

    Note that here we use diagonal basis.
    # Number of sites will be L1 * L2, and each site as 4 states.
    Number of sites will be 4 * L1 * L2.
    Also both L1 and L2 must be even.
    """

    if len(_L) == 2:
        H_list, sps = local(params, D=2)
        L1 = _L[0] * 2
        L2 = _L[1] * 2
        # if (L1 % 2 != 0) or (L2 % 2 != 0):
        #     raise ValueError("L1 and L2 must be even")
        S = np.arange(L1 * L2)
        x = S % L1
        y = S // L1
        T_x = (x + 1) % L1 + y * L1
        T_y = x + ((y + 1) % L2) * L1
        N2 = int(L1 * L2 / 2)
        # bonds1 = [[2*i, T_x[2*i]] for i in range((N2))] + \
        #     [[2*i+1, T_y[2*i+1]] for i in range((N2))]
        # bonds2 = [[2*i+1, T_x[2*i+1]] for i in range((N2))] + \
        #     [[2*i, T_y[2*i]] for i in range((N2))]

        bonds1 = []
        bonds2 = []
        for i in range(L1):
            for j in range(L2):
                idx = i+j*L1
                if (i+j) % 2 == 0:
                    bonds1.append([idx, T_x[idx]])
                    bonds2.append([idx, T_y[idx]])
                else:
                    bonds1.append([idx, T_y[idx]])
                    bonds2.append([idx, T_x[idx]])

        _H = utils.sum_ham(H_list[0], bonds1, L1 * L2, sps)
        _H += utils.sum_ham(H_list[1], bonds2, L1 * L2, sps)

        logging.info(f"L      : {L1}x{L2}")
        logging.info(f"params : {params}")
        if len(H_list) != 2:
            raise RuntimeError("something wrong")

        return _H, sps

    raise ValueError("Shastry-Sutherland model is 2D lattice")
