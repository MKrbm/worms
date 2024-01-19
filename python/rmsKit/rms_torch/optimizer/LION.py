from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import logging

# functions


def exists(val):
    return val is not None


# update functions


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    # p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.data[:] = torch.matrix_exp(-lr * update) @ p.data[:]
    # p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


# class


class LION(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        amsgrad=False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        self.epoch = 0

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        logging.info("LION optimizer")
        param_info = "lr = {}, betas = {}, weight_decay = {}".format(
            lr, betas, weight_decay)
        logging.info(param_info)

        super().__init__(params, defaults)

        self.update_fn = update_fn

    @staticmethod
    def lr_decay(epoch: int, lr: float) -> float:
        def f(x): return np.exp(-5 * np.tanh(x * 0.04))
        lr = f(epoch) * lr
        # epoch = (epoch // 10) * 10
        return lr

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure() if callable(closure) else closure

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, self.lr_decay(self.epoch, lr), wd, beta1, beta2)

        self.epoch += 1
        return loss
