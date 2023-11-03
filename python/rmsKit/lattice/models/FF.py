# kagome heisenberg model
import sys, os
import numpy as np
import random
from ..core.constants import *
from ..core.utils import *
import logging
import tensornetwork as tn
import utils
from typing import List


def block1D(*dimensions, normal = True):
    """Construct a new matrix for the MPS with random numbers from 0 to 1"""
    size = tuple([x for x in dimensions])
    if normal:
        A = np.random.normal(size=size)
    else:
        A = np.random.random_sample(size)
    A = (A.transpose(2, 1, 0) + A) / 2
    return A


def create_MPS(n, A):
    """Build the MPS tensor"""
    mps = [tn.Node(np.copy(A)) for _ in range(n)]
    # connect edges to build mps
    connected_edges = []
    for k in range(0, n):
        conn = mps[k][2] ^ mps[(k + 1) % n][0]
        connected_edges.append(conn)
    return mps, connected_edges


unitary_algorithms = ["original"]


def local(params: dict, check_L: List[int] = []):
    logging.info("params: {}".format(params))
    sps = params["sps"]
    bond_dim = params["rank"]
    d = params["dimension"]
    seed = params["seed"]
    lt = params["lt"]  # number of sites in unit cell
    logging.info(f"generate ff with seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    if check_L:
        if d != len(check_L):
            raise ValueError("dimension not match")

    if d == 1:
        A = block1D(bond_dim, sps, bond_dim)
        A2 = np.einsum("ijk,klm->jlim", A, A).reshape(sps**2, bond_dim**2)
        U, s, V = np.linalg.svd(A2)
        Up = U[:, len(s):]
        h = Up @ Up.T
        if lt == 2:
            _h = utils.sum_ham(h / 2, [[0, 1], [2, 3]], 4, sps)
            _h += utils.sum_ham(h, [[1, 2]], 4, sps)
            h = _h
            sps = sps**2
        if check_L:
            L = check_L[0]
            H = utils.sum_ham(h, [[i, (i + 1) % L] for i in range(L)], L, sps)
            e = np.linalg.eigvalsh(H)
            logging.info(f"eigenvalues: {e}")
            return [h], sps
        else:
            return [h], sps
    else:
        raise NotImplementedError("not implemented")


def system(_L: list[int], params: dict) -> np.ndarray:
    _d = len(_L)
    s = params["sps"]
    d = params["dimension"]
    L = _L[0]
    if d != _d:
        raise ValueError("dimension not match")
    H = np.zeros((s**L, s**L))
    h_list, sps = local(ua, params)
    bonds = [[i, (i + 1) % L] for i in range(L)]
    H = utils.sum_ham(h_list[0], bonds, L, s)
    return H
