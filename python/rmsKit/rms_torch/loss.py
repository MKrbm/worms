import torch
import numpy as np
from torch import nn
import math
from typing import Union, Tuple





class SystemEnergyLoss(nn.Module):
    def __init__(self, H: Union[np.ndarray, torch.Tensor], device: torch.device = torch.device("cpu")):
        super(SystemEnergyLoss, self).__init__()

        if isinstance(H, np.ndarray):
            self.H = torch.from_numpy(H).to(device)
        elif isinstance(H, torch.Tensor):
            self.H = H.to(device)
        else:
            raise TypeError("H should be of type np.ndarray or torch.Tensor.")
        E, V = torch.linalg.eigh(self.H)
        self.offset = E[-1]
        self.eye_offset = self.offset * torch.eye(self.H.shape[0], device=device)
        self.X = V[:,0].to(device)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        A = torch.matmul(U, torch.matmul(self.H, U.T))
        result_abs = self.stoquastic(A)
        E = torch.linalg.eigvalsh(result_abs)
        z = torch.exp(-E * 1).sum()
        return torch.log(z)

    def stoquastic(self, A : torch.Tensor):
        return -torch.abs(A - self.eye_offset) + self.eye_offset

class SystemQuasiEnergyLoss(SystemEnergyLoss):
    def __init__(
        self, 
        H: Union[np.ndarray, torch.Tensor], 
        N: int = 10,
        r: float = 0, #* regularization
        device: torch.device = torch.device("cpu")):
        super(SystemQuasiEnergyLoss, self).__init__(H, device)
        self.H = self.H - self.eye_offset
        self.N = int(N)


    def forward(self, U: torch.Tensor, r: float = 0) -> torch.Tensor:
        SUx = torch.abs(U @ self.X)
        A = U @ self.H @ U.T
        SUH = self.stoquastic(A)
        y = SUx
        for _ in range(self.N):
            y = SUH @ y
            if _ % 4 == 0:
                y = y / torch.norm(y)
        quasi_Sgs = torch.abs(y / torch.norm(y))

        # gap = (SUx - quasi_Sgs) @ SUH @ (quasi_Sgs + SUx) 
        # return gap - SUx @ y - self.offset
        z = SUH @ quasi_Sgs
        #* if H is real and symmetric
        return - (quasi_Sgs @ z + self.offset) + r * (1 - torch.abs(quasi_Sgs.dot(z) / torch.norm(z)))