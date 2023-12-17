"""Functions used for hamiltonian manipulation. """
import numpy as np
from typing import List


def sum_ham(h: np.ndarray, bonds: List, L: int, sps: int, stoquastic_=False, out=None):
    """Sum of local hamiltonian terms."""
    local_size = h.shape[0]
    h = np.kron(h, np.eye(int((sps**L) / h.shape[0])))
    ori_shape = h.shape
    H = np.zeros_like(h)
    h = h.reshape(L * 2 * (sps,))
    for bond in bonds:
        if sps ** len(bond) != local_size:
            raise ValueError("local size is not consistent with bond")
        trans = np.arange(L)
        for i, b in enumerate(bond):
            trans[b] = i
        l_tmp = len(bond)
        for i in range(L):
            if i not in bond:
                trans[i] = l_tmp
                l_tmp += 1
        trans = np.concatenate([trans, trans + L])
        H += h.transpose(trans).reshape(ori_shape)
    if stoquastic_:
        return stoquastic(H)
    return H


def stoquastic(X: np.ndarray) -> np.ndarray:
    """Make a hamiltonian stoquastic matrix."""
    a = np.eye(X.shape[0]) * np.max(np.diag(X))
    return -np.abs(X - a) + a
