import torch
import numpy as np
from torch import nn
import math
from typing import Union, Tuple

class CustomLoss(nn.Module):
    def __init__(self, H: Union[np.ndarray, torch.Tensor], device: torch.device = torch.device("cpu")):
        super(CustomLoss, self).__init__()

        if isinstance(H, np.ndarray):
            self.H = torch.from_numpy(H).to(device)
        elif isinstance(H, torch.Tensor):
            self.H = H.to(device)
        else:
            raise TypeError("H should be of type np.ndarray or torch.Tensor.")
        self.eye = torch.eye(self.H.shape[0], device=device)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        A = torch.matmul(U, torch.matmul(self.H, U.T))

        a = torch.max(torch.diag(A)) * self.eye
        result_abs = -torch.abs(A - a) + a

        E = torch.linalg.eigvalsh(result_abs)
        z = torch.exp(-E * 1).sum()
        return torch.log(z)