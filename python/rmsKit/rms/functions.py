import jax
import jax.numpy as jnp
import abc
from typing import Union, Tuple
import numpy as np
from jax._src.basearray import Array


@jax.jit
def stoquastic(X: Array) -> Array:
    a = jnp.eye(X.shape[0]) * jnp.max(jnp.diag(X))
    return -jnp.abs(X - a) + a


@jax.jit
def check_is_unitary(X: Array) -> bool:
    I = X @ X.T.conj()
    return (jnp.abs(I - jnp.eye(X.shape[0])) < 1e-8).all()
    # return jnp.linalg.norm(X - jnp.conj(X.T)) < 1E-5


@jax.jit
def _riemannian_grad(u, euc_grad):
    """
    Calculate riemannian gradient from euclidean gradient

    Args:
        u (Array): unitary matrix
        euc_grad (Array): euclidean gradient of a loss cuntion
    """
    riemannianGradient = euc_grad @ u.T.conj() - u @ euc_grad.T.conj()
    return riemannianGradient


def sum_ham(h, bonds, L, sps, stoquastic_=False, out=None):
    local_size = h.shape[0]
    h = np.kron(h, np.eye(int((sps**L) / h.shape[0])))
    ori_shape = h.shape
    H = np.zeros_like(h)
    h = h.reshape(L * 2 * (sps,))
    for bond in bonds:
        if sps ** len(bond) != local_size:
            raise ValueError("local size is not consistent with bond")
        trans = np.arange(L)
        _trans = [i for i in range(L)]
        for i, b in enumerate(bond):
            trans[b] = i
        l = len(bond)
        for i in range(L):
            if i not in bond:
                trans[i] = l
                l += 1
        trans = np.concatenate([trans, trans + L])
        H += h.transpose(trans).reshape(ori_shape)
    if stoquastic_:
        return stoquastic(H)
    return H


def riemannian_grad(u: Array, euc_grad: Array) -> Array:
    if not check_is_unitary(u):
        raise ValueError("u must be unitary matrix")
    if u.dtype != euc_grad.dtype:
        raise ValueError("dtype of u and euc_grad must be same")

    if u.shape != euc_grad.shape:
        raise ValueError("shape of u and euc_grad must be same")
    r = _riemannian_grad(u, euc_grad)
    return (r - r.T.conj()) / 2


