import jax
import jax.numpy as jnp
import abc
from typing import Union, Tuple
import numpy as np
from jax._src.basearray import Array
from jax._src.config import config
import abc
from ..unitary import UnitaryRiemanGenerator
from ..functions import stoquastic

config.update("jax_enable_x64", True)  # * enable 64bit float

TYPES = [np.float32, np.float64, np.complex64, np.complex128]


class BaseLoss(abc.ABC):
    """
    Base class for loss function
    """

    def __init__(self, H: Array, D: int, dtype):
        """
        params
        ------
        """
        if not (isinstance(H, Array)):
            raise ValueError("H must be jax.numpy.ndarray")
        if (H.T != H).all():
            raise ValueError("H must be hermitian matrix")

        if not (isinstance(D, int)):
            raise ValueError("D must be int")
            # raise ValueError("H must be stoquastic matrix")

        self.dtype = dtype
        self.H = H.astype(self.dtype)
        self.D = D

        if H.shape != (D * D, D * D):
            raise ValueError(
                "size of local hamiltonian and one-site unitary matrix are incompatible"
            )

    def __call__(self, M: Array, stq: bool = True) -> Array:
        """
        loss function
        """
        if M.shape != (self.D, self.D):
            raise ValueError("size of M is not appropriate")
        if M.dtype != self.dtype:
            raise ValueError("dtype of M is not appropriate")
        return self._loss(M, stq)

    @abc.abstractmethod
    def _loss(self, M: Array, stq: bool) -> Array:
        """
        loss function
        """
        pass


class QuasiEnergy(BaseLoss):
    """
    loss function for unitary matrix
    """

    def __init__(
        self,
        H: Array,
        X: Array,
        D: int,
        dtype,
    ):
        super().__init__(H, D, dtype)
        # if (jnp.linalg.norm(H - stoquastic(H)) > 1E-5):
        #     print("Warning! H is not stoquastic matrix")
        #     print("Automatically convert H to stoquastic matrix")
        #     H = stoquastic(H)
        if not (isinstance(X, Array)):
            raise ValueError("X must be jax.numpy.ndarray")
        if X.ndim != 1:
            raise ValueError("X must be 1D array")
        self.X = X
        self.X = self.X.reshape(self.H.shape[0], -1)
        self.X = self.X.astype(self.dtype)
        self.localD = self.X.shape[0]
        self.systemD = X.shape[0]
        exp = np.emath.logn(self.D, self.systemD)
        if abs(exp - round(exp)) > 1e-5:
            raise ValueError("size of X must be the power of D")
        self.n_units = round(exp)
        exp = np.emath.logn(self.D, self.localD)
        if abs(exp - round(exp)) > 1e-5:
            raise ValueError("size of X must be the power of D")
        self.n_units_local = round(exp)
        self.target = -jnp.trace(self.X @ self.X.T @ self.H)

    @staticmethod
    def _global_unitary(M: Array, n: int) -> Array:
        """
        loss function
        """
        U = M
        for _ in range(n - 1):
            U = jnp.kron(U, M)

        # def kron(i, U):
        #     return jnp.kron(U, M)
        # jax.lax.fori_loop(0, n - 1, kron, U)
        return U

    def _loss(self, M: Array, stq: bool) -> Array:
        """
        loss function
        """
        n = self.n_units_local
        m = self.n_units - n
        if self.D ** (n + m) != self.systemD:
            raise RuntimeError("something go wrong, size of X is not appropriate")
        return self._stq_loss(M, self.H, self.X, n, m) if stq else self.target

    def _stq_loss(self, u: Array, H: Array, X: Array, n: int, m: int) -> Array:
        """
        loss function
        """
        U1 = QuasiEnergy._global_unitary(u, n)
        U2 = QuasiEnergy._global_unitary(u, m)
        X = jnp.abs(U1 @ X @ U2.T)
        return -jnp.trace(X @ X.T @ stoquastic(U1 @ H @ U1.T))

    # @jax.jit
    # def _loss(self, M) -> Array:
    #     """
    #     loss function
    #     """
    #     if (self.X.shape[0] != M.shape[0]**self.n_units):
    #         raise ValueError("size of X is not appropriate")
    #     U = self._global_unitary(M, self.n_units)
    #     X = jnp.abs(U @ self.X)
    #     X = X.reshape(self.H.shape[0], -1)
    #     return jnp.trace(X @ X.T @ self.H)


class MinimumEnergy(BaseLoss):
    """
    loss function for unitary matrix
    """

    def __init__(
        self,
        H: Array,
        D: int,
        dtype,
    ):
        super().__init__(H, D, dtype)

    def _loss(self, M: Array, stq: bool) -> Array:
        """
        loss function
        """
        return self._stq_loss(M, self.H) if stq else self._nstq_loss(M, self.H)

    @staticmethod
    @jax.jit
    def _stq_loss(u: Array, H: Array) -> Array:
        """
        loss function
        """
        U = jnp.kron(u, u)
        E = jnp.linalg.eigvalsh(stoquastic(U.T @ H @ U))
        # print(E[0])
        return -E[0]

    @staticmethod
    @jax.jit
    def _nstq_loss(u: Array, H: Array) -> Array:
        """
        loss function
        """
        U = jnp.kron(u, u)
        E = jnp.linalg.eigvalsh(U.T @ H @ U)
        return -E[0]
