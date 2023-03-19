import torch
import numpy as np
from torch import nn
import math


def random_unitary_matrix(size: int, device: torch.device) -> torch.Tensor:
    random_matrix = np.random.randn(size, size)
    q, _ = np.linalg.qr(random_matrix)
    return torch.tensor(q, dtype=torch.float64, device=device)

class CustomModel(nn.Module):
    def __init__(self, H_size: int, unitary_size: int, device = torch.device("cpu"), u0 = None):
        super(CustomModel, self).__init__()

        if H_size <= 0 or unitary_size <= 0:
            raise ValueError("Both H_size and unitary_size should be positive integers.")

        n_us = round(math.log2(H_size) / math.log2(unitary_size))
        if u0 is None:
            self.us = nn.ParameterList([nn.Parameter(random_unitary_matrix(unitary_size, device), requires_grad=True) for _ in range(n_us)])
        else:
            self.us = nn.ParameterList([
                nn.Parameter(torch.tensor(u0, dtype = torch.float64, device = device), requires_grad=True)
                for _ in range(n_us)])

    def forward(self) -> torch.Tensor:
        U = self.us[0]
        # for u in self.us[1:]:
        #     U = torch.kron(U, u)
        U = torch.kron(U, self.us[1])
        U = torch.kron(U, self.us[2])
        U = torch.kron(U, self.us[3])

        return U