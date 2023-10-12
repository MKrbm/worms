# * below for torch
import torch
import numpy as np
from typing import Any, Tuple, Union, List
from nptyping import NDArray

# * Type hinting change from Array to torch.Tensor


def check_is_unitary_torch(X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        I_prime = X @ X.T.conj()
        identity_matrix = torch.eye(X.shape[0], device=X.device)
        return (torch.abs(I_prime - identity_matrix) < 1e-8).all()


def riemannian_grad_torch(u: torch.Tensor, euc_grad: torch.Tensor) -> torch.Tensor:
    if not check_is_unitary_torch(u):
        raise ValueError("u must be unitary matrix")
    if u.dtype != euc_grad.dtype:
        raise ValueError("dtype of u and euc_grad must be same")

    if u.shape != euc_grad.shape:
        raise ValueError("shape of u and euc_grad must be same")
    rg = euc_grad @ u.T.conj() - u @ euc_grad.T.conj()
    return (rg - rg.T.conj()) / 2


def sum_ham(h: np.ndarray, bonds: List, L: int, sps: int, stoquastic_=False, out=None):
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
    a = np.eye(X.shape[0]) * np.max(np.diag(X))
    return -np.abs(X - a) + a
