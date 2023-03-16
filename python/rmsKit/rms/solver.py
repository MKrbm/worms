import jax
import jax.numpy as jnp
import numpy as np
from jax._src.basearray import Array
from typing import Union, Tuple, NamedTuple, Callable, Type
from .loss import BaseLoss, BaseMultiLoss
from .unitary import UnitaryRiemanGenerator
from .optimizer import cg, momentum, LION
from .functions import check_is_unitary, riemannian_grad
from tqdm.auto import tqdm
import abc
from jax.example_libraries.optimizers import (
    OptimizerState,
    UpdateFn,
    ParamsFn,
    Optimizer,
    Schedule,
    make_schedule,
)


def default_schedule(step: int) -> float:
    return 1.0 / (1.0 + step)


class BaseSolver(abc.ABC):
    def __init__(
        self,
        loss: Union[BaseMultiLoss, BaseLoss],
    ):
        self.loss = loss

    def _iter(
        self,
        u: Array,
        opt_state: OptimizerState,
        opt_update: UpdateFn,
        get_unitary: ParamsFn,
        nstep: int,
        return_bext: bool,
        cutoff_cnt: int,
        cout: bool,
        offset: float,
    ):
        if not check_is_unitary(u):
            raise ValueError("u is not unitary")

        def step(step, opt_state):
            value, grads = jax.value_and_grad(self.loss)(get_unitary(opt_state))
            rg = riemannian_grad(u, grads)
            opt_state = opt_update(step, rg, opt_state)
            return value, opt_state

        value = self.loss(u)
        best_u = u
        best_value = value
        bad_cnt = 0
        with tqdm(range(nstep), disable=not cout) as pbar:
            for t, ch in enumerate(pbar):
                value, opt_state = step(t, opt_state)
                u = get_unitary(opt_state)
                pbar.set_postfix_str(
                    "iter={}, loss={:.5f}, bad_cnt={}".format(t, value, bad_cnt)
                )
                if value > best_value - offset:
                    bad_cnt += 1
                    if bad_cnt > cutoff_cnt:
                        break
                else:
                    bad_cnt = 0
                if value < best_value:
                    best_value = value
                    best_u = u

        if return_bext:
            return best_u, best_value
        else:
            return u, value

    def __call__(
        self,
        u: Array,
        nstep: int,
        step_size: Union[Schedule, float],
        return_bext: bool = True,
        cutoff_cnt: int = 50,
        cout: bool = False,
        offset: float = 1e-5,
        **kwargs,
    ):
        """
        solve the momentum equation
        """

        if offset < 0:
            raise ValueError("offset must be positive")
        if cutoff_cnt < 0:
            raise ValueError("cutoff_cnt must be positive")

        opt_init, opt_update, get_unitary = self._get_Optimizer(step_size, **kwargs)
        opt_state = opt_init(u)
        return self._iter(
            u,
            opt_state,
            opt_update,
            get_unitary,
            nstep,
            return_bext,
            cutoff_cnt,
            cout,
            offset,
        )

    @abc.abstractmethod
    def _get_Optimizer(self, step_size: Union[Schedule, float], **kwargs) -> Optimizer:
        pass


class momentumSolver(BaseSolver):
    def _get_Optimizer(self, step_size: Union[Schedule, float], mass: float) -> Optimizer:
        return momentum(step_size, mass)


class lionSolver(BaseSolver):
    def _get_Optimizer(
        self, step_size: Union[Schedule, float], mass1: float, mass2: float
    ) -> Optimizer:
        return LION(step_size, mass1, mass2)


class cgSolver(BaseSolver):
    def _get_Optimizer(self, step_size: Union[Schedule, float], mass: float) -> Optimizer:
        return cg(step_size, mass, self.loss)


# def momentum_solver(
#     loss: Union[BaseMultiLoss, BaseLoss],
#     u: Array,
#     nstep: int,
#     step_size: float,
#     mass: float,
#     return_bext: bool = True,
#     cutoff_cnt: int = 50,
#     **kwargs,
# ) -> Tuple[Array, Array]:
#     return solver(
#         momentum, loss, u, nstep, step_size, mass, return_bext, cutoff_cnt, **kwargs
#     )


# def cg_solver(
#     loss: Union[BaseMultiLoss, BaseLoss],
#     u: Array,
#     nstep: int,
#     step_size: float,
#     mass: float,
#     return_bext: bool = True,
#     cutoff_cnt: int = 50,
#     **kwargs,
# ) -> Tuple[Array, Array]:
#     return solver(cg, loss, u, nstep, step_size, mass, return_bext, cutoff_cnt, **kwargs)


def solver(
    optimizer: Callable,
    loss: Union[BaseMultiLoss, BaseLoss],
    u: Array,
    nstep: int,
    step_size: float,
    mass: float,
    return_bext: bool = True,
    cutoff_cnt: int = 50,
    **kwargs,
) -> Tuple[Array, Array]:
    """
    solve the momentum equation
    """

    if not check_is_unitary(u):
        raise ValueError("u is not unitary")
    opt_init, opt_update, get_unitary = optimizer(step_size, mass)
    opt_state = opt_init(u)

    def step(step, opt_state):
        value, grads = jax.value_and_grad(loss)(get_unitary(opt_state))
        rg = riemannian_grad(u, grads)
        opt_state = opt_update(step, rg, opt_state)
        return value, opt_state

    value = loss(u)
    best_u = u
    best_value = value
    bad_cnt = 0
    with tqdm(range(nstep), disable=False) as pbar:
        for t, ch in enumerate(pbar):
            value, opt_state = step(t, opt_state)
            u = get_unitary(opt_state)
            pbar.set_postfix_str(
                "iter={}, loss={:.5f}, bad_cnt={}".format(t, value, bad_cnt)
            )
            if value > best_value:
                bad_cnt += 1
                if bad_cnt > cutoff_cnt:
                    break
            else:
                bad_cnt = 0
            if value < best_value:
                best_value = value
                best_u = u

    if return_bext:
        return best_u, best_value
    else:
        return u, value
