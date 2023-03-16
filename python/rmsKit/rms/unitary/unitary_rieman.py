import jax
import jax.numpy as jnp
import abc
from typing import Union, Tuple, Type
import numpy as np
from jax._src.basearray import Array
from jax._src.config import config
from ..functions import _riemannian_grad, check_is_unitary
config.update('jax_enable_x64', True)  # * enable 64bit float

TYPES = [np.float32, np.float64, np.complex64, np.complex128]


class UnitaryRiemanGenerator:
    """
    parameter space is the same as D by D square matrix. No constrants but inital matrix is unitary.
    Random number is explicitly needed to initialize the matrix.

    params
    ------
    D : int
    key : jax.random.PRNGKey object
    dtype : one of np.float32, np.float64, np.complex64, np.complex128
    """

    _matrix: Array

    def __init__(self, D: int, key: jax.random.KeyArray, dtype: Union[Type[np.float64], Type[np.float32]]):
        print("UnitaryRiemanGenerator is initialized")
        self.key = key
        self.D = D
        self.complex = False
        if dtype in TYPES:
            self.dtype = dtype
            if (self.dtype == np.complex128) or (self.dtype == np.complex64):
                self.complex = True
                raise NotImplementedError(
                    "complex unitary matrix is not implemented yet")
        else:
            raise ValueError(
                "dtype must be either np.float64 or np.complex128")

    @property
    def size(self) -> tuple[int, int]:
        return (self.D, self.D)

    def reset_matrix(self, key: jax.random.KeyArray = None) -> Array:
        """
        reset to random unitary matrix.
        Uniformly random in haar measure.
        """
        if not key:
            self.key, key = jax.random.split(self.key)
        if not self.complex:
            randnMatrix = jax.random.normal(key, shape=(self.D, self.D), dtype=self.dtype)
        else:
            raise NotImplementedError("complex unitary matrix is not implemented yet")

        Q, R = jnp.linalg.qr(randnMatrix)
        dR = jnp.diag(R)
        haar_orth = Q.dot(jnp.diag(jnp.sign(dR)))
        return haar_orth
        # self._set_matrix(haar_orth)

    def riemannian_grad(self, u: Array, euc_grad: Array) -> Array:
        if not check_is_unitary(u):
            raise ValueError("u must be unitary matrix")
        if (u.dtype != euc_grad.dtype):
            raise ValueError("dtype of u and euc_grad must be same")

        if (u.shape != euc_grad.shape):
            raise ValueError("shape of u and euc_grad must be same")
        if (u.shape != (self.D, self.D)):
            raise ValueError("shape of u and euc_grad must be same")
        r = _riemannian_grad(u, euc_grad)
        return (r - r.T.conj()) / 2

    @staticmethod
    @jax.jit
    def unitarize(X: Array) -> Array:
        """
        Return a closest unitary matrix to X.
        """
        V, _, W = jnp.linalg.svd(X)
        return V @ W
