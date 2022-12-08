import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional

class Sign(torch.optim.SGD):

    def __init__(self,  params, lr, decay_rate = 1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False):
        """
        gamma is decay rate for gradient
        """
        self.decay = decay_rate
        if not 0 <= self.decay <= 1:
            raise ValueError("Invalid decay_rate value : {}".format(decay_rate))
        self.r = 1
        super().__init__(params, lr, momentum , dampening,
                weight_decay, nesterov, maximize=maximize)

    @torch.no_grad()
    def step(self, closure=None):

        # self._cuda_graph_capture_health_check()
        loss = None
        self.r *= self.decay
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad[:] = torch.sign(p.grad)[:] * self.r
        return super().step(closure)