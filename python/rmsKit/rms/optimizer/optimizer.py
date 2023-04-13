from scipy import optimize
import jax
import jax.numpy as jnp
import abc
from typing import Union, Tuple, NamedTuple, Callable
import numpy as np
import math
from jax._src.basearray import Array
from jax._src.config import config
from jax.example_libraries.optimizers import (
    optimizer,
    make_schedule,
    Schedule,
    InitFn,
    UpdateFn,
    ParamsFn,
)
from jax.scipy.linalg import expm
from ..functions import check_is_unitary
from ..unitary import UnitaryRiemanGenerator
from ..loss import BaseLoss, BaseMultiLoss

config.update("jax_enable_x64", True)  # * enable 64bit float


class Optimizer(NamedTuple):
    init_fn: InitFn
    update_fn: UpdateFn
    unitary_fn: ParamsFn


@optimizer
def sgd(step_size):
    """Construct optimizer triple for stochastic gradient descent.

    Args:
        step_size: positive scalar, or a callable representing a step size schedule
            that maps the iteration index to a positive scalar.

    Returns:
        An (init_fun, update_fun, get_params) triple.

    update_fun(step, riemannian_grads, opt_state)

    Args:
      step: integer representing the step index.
      grads: a pytree with the same structure as `get_params(opt_state)`
        representing the gradients to be used in updating the optimizer state.
      opt_state: a pytree representing the optimizer state to be updated.

    Returns:
      A pytree with the same structure as the `opt_state` argument representing
      the updated optimizer state.

    """
    step_size = make_schedule(step_size)

    def init(x0):
        return x0

    def update(i, rg, u):
        return expm(-step_size(i) * rg) @ u

    def get_unitary(x):
        return x

    return Optimizer(init, update, get_unitary)


@optimizer
def momentum(step_size: Schedule, mass: float):
    """Construct optimizer triple for SGD with momentum.

    Args:
        step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.
        mass: positive scalar representing the momentum coefficient.

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        v0 = jnp.zeros_like(x0)
        return x0, v0

    def update(i, rg, state):
        u, vrg = state
        vrg = mass * vrg + rg
        u = expm(-step_size(i) * vrg) @ u
        return u, vrg

    def get_unitary(state):
        x, _ = state
        return x

    return init, update, get_unitary


@optimizer
def LION(
    step_size: Schedule,
    mass1: float,
    mass2: float,
):
    """Construct optimizer triple for SGD with momentum.

    Args:
        step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.
        mass: positive scalar representing the momentum coefficient.

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        v0 = jnp.zeros_like(x0)
        return x0, v0

    def update(i, rg, state):
        u, vrg = state
        c = mass1 * vrg + (1 - mass1) * rg
        delta = jnp.sign(c)
        u = expm(-step_size(i) * delta) @ u
        vrg = mass2 * vrg + (1 - mass2) * rg
        return u, vrg

    def get_unitary(state):
        x, _ = state
        return x

    return init, update, get_unitary


@optimizer
def cg(step_size: float, mass: float, loss: Union[BaseMultiLoss, BaseLoss]):
    """Construct optimizer triple for SGD with momentum.

    Args:
        step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.
        mass: positive scalar representing the momentum coefficient.

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """

    def init(x0):
        v0 = jnp.zeros_like(x0)
        return x0, v0

    def update(i, rg, state):
        u, vrg = state
        vrg = mass * vrg + rg
        lr = golden(u, vrg, loss, delta=step_size)
        u = unitarize(expm(-lr * vrg) @ u)
        return u, vrg

    def get_unitary(state):
        x, _ = state
        return x

    return init, update, get_unitary


def golden(
    u: Array, rg: Array, loss: Union[BaseMultiLoss, BaseLoss], delta=0.001, cout=False
) -> float:
    def objective(t):
        return loss(jax.scipy.linalg.expm(-t * rg) @ u)

    t = jnp.array(1.0)

    g = jax.grad(objective)(t)
    g2 = jax.grad(jax.grad(objective))(t)
    if math.isnan(g2):
        g2 = 1
    step = g / g2 * delta
    step = abs(step.item())
    t = 10
    for _ in range(10):
        # print(objective(step))
        if objective(step) < objective(0):
            while True:
                if objective(step) < objective(step * t):
                    # print(objective(0) ,objective(step), objective(step*t))
                    a = optimize.golden(objective, brack=(0, step, step * t))
                    if isinstance(a, float):
                        return a
                    else:
                        raise TypeError("return value is supposed to be float")
                else:
                    step *= t
        else:
            step /= t
    if cout:
        print("No local minimum found")
    a = optimize.golden(objective)
    if isinstance(a, float):
        return a
    else:
        raise TypeError("return value is supposed to be float")


@jax.jit
def unitarize(X: Array) -> Array:
    """
    Return a closest unitary matrix to X.
    """
    V, _, W = jnp.linalg.svd(X)
    return V @ W
