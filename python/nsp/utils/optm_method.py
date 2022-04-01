import numpy as np
from pyrsistent import v
from scipy.linalg import expm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


class scheme1:
    def __init__(self, param, gamma, r = 0.99):
        self.gamma = gamma
        self.param = list(param)
        assert 0 < r <= 1, "0 < r <= 1"
        self.r = r
    def step(self, alpha = None):
        for p in self.param:
            if alpha is None:
                # alpha = 5*np.random.rand(p.shape[0])
                alpha = np.random.normal(1, 1, p.shape[0])
                # alpha = np.random.rand(1)
            alpha = torch.tensor(alpha)
            assert alpha.shape[0] == p.shape[0] or alpha.shape[0] == 1
            grad = torch.sign(p.grad.data)
            p.data.add_(-alpha * self.gamma * grad)
            self.gamma *= self.r
    def reset(self, gamma, r = 0.99):
        self.gamma = gamma
        self.r = r

    def zero_grad(self):
        for p in self.param:
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()