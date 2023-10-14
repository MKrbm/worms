# kagome heisenberg model

import sys,os
import numpy as np
import random
from ..core.constants import *
from ..core.utils import *
import logging
import tensornetwork as tn
import utils

def block(*dimensions):
    """Construct a new matrix for the MPS with random numbers from 0 to 1"""
    size = tuple([x for x in dimensions])
    return np.random.random_sample(size)


def create_MPS(s, n, dimension, bd):
    """Build the MPS tensor"""
    A = block(bd, s, bd)
    mps = [tn.Node(np.copy(A)) for _ in range(n)]
    # connect edges to build mps
    connected_edges = []
    for k in range(0, n):
        conn = mps[k][2] ^ mps[(k + 1) % n][0]
        connected_edges.append(conn)

    return mps, connected_edges

unitary_algorithms = ["original"]
def local(ua : str, params : dict):
    if ua not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")
    s = params["sps"]
    bond_dim = params["rank"]
    n = params["length"]
    d = params["dimension"]
    seed = params["seed"]
    logging.info(f"seed: {seed}")

    np.random.seed(seed)
    random.seed(seed)

    mps_nodes, mps_edges = create_MPS(s, n, d, bond_dim)
    if d != 1:
        raise ValueError("d != 1 not supported")

    for k in range(len(mps_edges)):
        A = tn.contract(mps_edges[k])

    y = A.tensor.reshape(-1)
    rho = y[:, None] @ y[None, :]
    rho_ = rho.reshape(s**2, s ** (n - 2), s**2, s ** (n - 2))
    prho = np.einsum("jiki->jk", rho_)
    e, V = np.linalg.eigh(prho)
    e = np.round(e, 10)
    P = np.diagflat((e == 0)).astype(np.float64)
    vp = V @ P
    h = vp @ vp.T
    bonds = [[i, (i + 1) % n] for i in range(n)]
    H = utils.sum_ham(-h, bonds, n, s)
    e0 = np.linalg.eigh(H)[0][0]    
    logging.info(f"e0: {e0}")

    return [-h], s



def system(_L : list[int], ua : str, params : dict) -> np.ndarray:
    if ua not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")
    _d = len(_L)
    s = params["sps"]
    d = params["dimension"]
    L = _L[0]
    if (d!=_d):
        raise ValueError("dimension not match")
    H = np.zeros((s**L, s**L))
    h_list, sps = local(ua, params) 
    bonds = [[i, (i + 1) % L] for i in range(L)]
    H = utils.sum_ham(h_list[0], bonds, L, s)
    return H


