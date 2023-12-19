# FF model
import sys
import os
import numpy as np
import random
from ..core.paulis.spin import *
from .. import utils
import logging
import tensornetwork as tn
from typing import List, Any, Tuple, Dict


def block1D(*dimensions, normal=True, seed=None, canonical=True, sym=True):
    """Construct a new matrix for the MPS with random numbers from 0 to 1"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    size = tuple([x for x in dimensions])
    if normal:
        A = np.random.normal(size=size)
    else:
        A = np.random.random_sample(size)
    if sym:
        A = (A.transpose(2, 1, 0) + A) / 2

    if canonical:
        if not sym:
            raise NotImplementedError("Currently MPS must be symmetric to be canonical ")

        A = get_canonical_form(A)
        if np.linalg.norm(A.imag) > 1E-8:
            raise ValueError("A is not real")
        return A.real
    else:
        return A


def block2D(sps, bd, seed=None, normal=True):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    '''Construct a new matrix for the MPS with random numbers from 0 to 1'''
    size = (bd,) * 4 + (sps,)
    if normal:
        A = np.random.normal(size=size)
    else:
        A = np.random.random_sample(size)
    A = (A.transpose(2, 1, 0, 3, 4) + A.transpose(0, 3,
         2, 1, 4) + A.transpose(2, 3, 0, 1, 4) + A) / 4
    A = A + A.transpose([1, 2, 3, 0, 4])
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


def create_PEPS(L1, L2, A):
    '''Build the PEPS tensor'''
    if A.ndim != 5:
        raise ValueError("hi")
    peps = [tn.Node(np.copy(A)) for _ in range(L1*L2)]
    s = np.arange(L1 * L2)
    x = s % L1
    y = s // L1
    T_x = (x + 1) % L1 + y * L1
    T_y = x + ((y + 1) % L2) * L1
    bonds = [([i, T_x[i]], [0, 2]) for i in range(L1*L2)] + \
        [([i, T_y[i]], [1, 3]) for i in range(L1*L2)]
    connected_edges = []
    for bond in bonds:
        conn = peps[bond[0][0]][bond[1][0]] ^ peps[bond[0][1]][bond[1][1]]
        connected_edges.append(conn)
    return peps, connected_edges


unitary_algorithms = ["original"]


def local(params: Dict[Any, Any], check_L: List[int] = []) -> Tuple[Any, int]:
    logging.info("params: {}".format(params))
    sps = params["sps"]
    bond_dim = params["rank"]
    d = params["dimension"]
    seed = params["seed"]
    L = params["lt"]  # number of sites in unit cell
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
        if L >= 2:
            _h = utils.sum_ham(h / 2, [[i, i+1] for i in range(2*L-1)], 2 * L, sps)
            _h += utils.sum_ham(h / 2, [[L-1, L]], 2 * L, sps)
            h = _h
            sps = sps**L

        if check_L:
            L = check_L[0]
            H = utils.sum_ham(h, [[i, (i + 1) % L] for i in range(L)], L, sps)
            e = np.linalg.eigvalsh(H)
            logging.info(f"eigenvalues: {e}")
            return [h], sps
        else:
            return [h], sps
    elif d == 2:
        if L != 1:
            raise NotImplementedError(
                "2D PEPS are not implemented for lt != 1")

        AP = block2D(sps, bond_dim)
        AP2 = np.einsum("ijklm,abide->mejklabd", AP, AP).reshape(sps ** 2, -1)
        U, s, V = np.linalg.svd(AP2)
        s = np.round(s, 10)
        s = s[s != 0]
        Up = U[:, len(s):]
        h = Up @ Up.T
        h = (h + h.T)/2
        if np.linalg.norm(h) < 1E-8:
            raise ValueError("h is null operator")
        return [h], sps
    else:
        raise NotImplementedError("not implemented")


def system(_L: list[int], params: dict) -> np.ndarray:
    _d = len(_L)
    s = params["sps"]
    d = params["dimension"]
    if _d == 1:
        L = _L[0]
        if d != _d:
            raise ValueError("dimension not match")
        H = np.zeros((s**L, s**L))
        h_list, sps = local(params)
        bonds = [[i, (i + 1) % L] for i in range(L)]
        H = utils.sum_ham(h_list[0], bonds, L, s ** params["lt"])
        return H
    elif _d == 2:
        L1 = _L[0]
        L2 = _L[1]
        s_ = np.arange(L1 * L2)
        x = s_ % L1
        y = s_ // L1
        T_x = (x + 1) % L1 + y * L1
        T_y = x + ((y + 1) % L2) * L1
        bonds = [[i, T_x[i]]
                 for i in range(L1*L2)] + [[i, T_y[i]] for i in range(L1*L2)]
        h_list, sps = local(params)
        H = utils.sum_ham(h_list[0], bonds, L1*L2, s)
        return H
    else:
        raise NotImplementedError("not implemented")


def get_canonical_form(A):

    check_validity = True
    if A.ndim != 3:
        raise ValueError("A must be a 3-rank tensor")
    if A.shape[0] != A.shape[2]:
        raise ValueError(
            "middle index should represent physical index and the side indices should be virtual indices")

    bd = A.shape[0]
    A = A.transpose(1, 0, 2)
    A_tilde = np.einsum("ijk,ilm->jlkm", A, A)
    A_tilde = A_tilde.reshape(bd**2, bd**2)
    e, V = np.linalg.eigh(A_tilde)
    rho = e[-1]
    A_tilde = A_tilde / rho

    e, V = np.linalg.eigh(A_tilde)
    x = V[:, -1].reshape(bd, bd)

    e, U = np.linalg.eigh(x)
    x_h = U @ np.diag(np.sqrt(e + 0j)) @ U.T
    x_h_inv = U @ np.diag(1/np.sqrt(e + 0j)) @ U.T

    B = x_h_inv @ A @ x_h / np.sqrt(rho)  # canonical form
    B = B.transpose(1, 0, 2)

    if check_validity:
        check_cano = np.einsum("jik, lik->jl", B, B)
        if np.linalg.norm(np.eye(check_cano.shape[0]) - check_cano) > 1E-8:
            raise ValueError("B is not a canonical")
        B_ = B.transpose(1, 0, 2)
        B_tilde = np.einsum("ijk,ilm->jlkm", B_, B_).reshape(bd**2, bd ** 2)
        Eb = np.sort(np.linalg.eigvals(B_tilde))
        Ea = np.sort(np.linalg.eigvals(A_tilde))
        # print("Ea: " ,Ea)
        if np.linalg.norm(Ea.real - Eb.real) > 1E-8:
            raise ValueError("B is not a canonical")
    return B
