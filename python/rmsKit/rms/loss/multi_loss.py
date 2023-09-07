import jax
import jax.numpy as jnp
import abc
from typing import Union, Tuple, List, Any, TypeVar, NamedTuple
import numpy as np
from jax._src.basearray import Array
from jax._src.config import config
import abc
from ..unitary import UnitaryRiemanGenerator
from ..functions import stoquastic
from .loss import BaseLoss, MES, QES, SEL, SMEL, mes, qes, system_el, system_mel

config.update("jax_enable_x64", True)  # * enable 64bit float

TYPES = [np.float32, np.float64, np.complex64, np.complex128]


QESMulti = List[QES]
MESMulti = List[MES]
SELMulti = List[SEL]
SMELMulti = List[SMEL]


@jax.jit
def qes_multi(state: QESMulti, u: Array) -> Array:
    loss = jnp.array(0)
    for i in range(len(state)):
        loss += qes(state[i].H, state[i].X, u)
    return loss


@jax.jit
def mes_multi(state: MESMulti, u: Array) -> Array:
    loss = jnp.array(0)
    for i in range(len(state)):
        loss += mes(state[i].H, u)
    return loss


@jax.jit
def system_el_multi(state: SELMulti, u: Array) -> Array:
    loss = jnp.array(0)
    for i in range(len(state)):
        loss += system_el(state[i].H, state[i].beta, u)
    return loss


#minimize energy loss for multiple hamiltonian
@jax.jit
def system_mel_multi(state: SMELMulti, u: Array) -> Array:
    loss = jnp.array(0)
    for i in range(len(state)):
        loss += system_mel(state[i].H, u)
    return loss


T = TypeVar("T", bound=BaseLoss)


class BaseMultiLoss(abc.ABC):
    def __init__(self, loss_list: List[T]):
        self.loss_list = loss_list
        for loss in self.loss_list:
            if not (isinstance(loss, BaseLoss)):
                raise ValueError("loss_list must be list of BaseLoss")
            if loss.dtype != self.loss_list[0].dtype:
                raise ValueError("dtype of loss_list must be same")
            if loss.D != self.loss_list[0].D:
                raise ValueError("D of loss_list must be same")
        self.D = self.loss_list[0].D
        self.dtype = self.loss_list[0].dtype
        self.Hs = [loss.H for loss in self.loss_list]
        self.upper_bounds = self.__call__(jnp.eye(self.D, dtype=self.dtype))
        self.target = self.__call__(jnp.eye(self.D, dtype=self.dtype), stq=False)
        print(f"upper bounds : {self.upper_bounds:.6f}")
        print(f"target       : {self.target:.6f}")

    def __call__(
        self,
        M: Array,
        stq: bool = True,
    ) -> Array:
        return self._multi_loss(M, stq)

    @abc.abstractmethod
    def _multi_loss(self, M: Array, stq: bool) -> Array:
        pass


class MeanMultiLoss(BaseMultiLoss):
    def _multi_loss(self, M: Array, stq: bool) -> Array:
        loss = jnp.array(0)
        for loss_func in self.loss_list:
            loss += loss_func(M, stq)
        return loss
