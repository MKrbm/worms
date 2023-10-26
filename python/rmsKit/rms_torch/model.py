import torch
from torch import nn
import numpy as np
import math
from typing import Union, Tuple


def random_unitary_matrix(size: int, device: torch.device) -> torch.Tensor:
    random_matrix = np.random.randn(size, size)
    q, _ = np.linalg.qr(random_matrix)
    return torch.tensor(q, dtype=torch.float64, device=device)


class UnitaryRieman(nn.Module):
    def __init__(self, H_size: int, unitary_size: int, device=torch.device("cpu"), u0=None):
        super(UnitaryRieman, self).__init__()

        if H_size <= 0 or unitary_size <= 0:
            raise ValueError("Both H_size and unitary_size should be positive integers.")

        self.H_size = H_size
        self.unitary_size = unitary_size
        self.device = device
        self.u0 = u0

        self.initialize_params()

    def initialize_params(self):
        n_us = round(math.log2(self.H_size) / math.log2(self.unitary_size))
        if self.u0 is None:
            self.u = nn.ParameterList(
                [nn.Parameter(random_unitary_matrix(self.unitary_size, self.device), requires_grad=True)]
            )
        else:
            self.u = nn.ParameterList(
                [nn.Parameter(torch.tensor(self.u0, dtype=torch.float64, device=self.device), requires_grad=True)]
            )

    def reset_params(self, u0: Union[torch.Tensor, None] = None):
        if u0 is not None and u0.shape != (self.unitary_size, self.unitary_size):
            raise ValueError("The shape of u0 is not correct.")
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
            p.data = (
                random_unitary_matrix(self.unitary_size, self.device)
                if u0 is None
                # else torch.tensor(u0.clone().detach(), dtype=torch.float64, device=self.device)
                else u0.clone().detach().to(torch.float64).to(self.device)
            )

    def forward(self) -> torch.Tensor:
        # calculate kron of unitaries (result size must be H_size x H_size)

        num_repeat = round(math.log2(self.H_size) / math.log2(self.unitary_size))
        U = self.u[0]
        for i in range(num_repeat - 1):
            U = torch.kron(U, self.u[0])
        return U


class UnitaryRiemanNonSym(nn.Module):
    def __init__(self, H_size: int, unitary_size: int, device=torch.device("cpu"), u0_list=None):
        super(UnitaryRiemanNonSym, self).__init__()

        if H_size <= 0 or unitary_size <= 0:
            raise ValueError("Both H_size and unitary_size should be positive integers.")

        self.H_size = H_size
        self.unitary_size = unitary_size
        self.device = device
        self.u0_list = u0_list

        self.initialize_params()

    def initialize_params(self):
        n_us = round(math.log2(self.H_size) / math.log2(self.unitary_size))
        if self.u0_list is None:
            self.us = nn.ParameterList(
                [
                    nn.Parameter(random_unitary_matrix(self.unitary_size, self.device), requires_grad=True)
                    for _ in range(n_us)
                ]
            )
        elif len(self.u0_list) == 1:
            u0_tensor = torch.tensor(self.u0_list[0], dtype=torch.float64, device=self.device)
            self.us = nn.ParameterList([nn.Parameter(u0_tensor, requires_grad=True) for _ in range(n_us)])
        else:
            self.us = nn.ParameterList(
                [
                    nn.Parameter(torch.tensor(u0, dtype=torch.float64, device=self.device), requires_grad=True)
                    for u0 in self.u0_list
                ]
            )

    def reset_params(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
            p.data = random_unitary_matrix(self.unitary_size, self.device)

    def forward(self) -> torch.Tensor:
        U = self.us[0]
        # for u in self.us[1:]:
        #     U = torch.kron(U, u)
        U = torch.kron(U, self.us[1])
        U = torch.kron(U, self.us[2])
        U = torch.kron(U, self.us[3])

        return U
