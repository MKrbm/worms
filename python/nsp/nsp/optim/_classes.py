import abc
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class optimizer(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        self._cuda_graph_capture_health_check()

        loss = None

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Sgin does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)


        return loss