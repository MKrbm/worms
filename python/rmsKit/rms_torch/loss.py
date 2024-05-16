import torch
import numpy as np
from torch import nn
from typing import Union, List
from .functions import is_hermitian_torch
import logging

logger = logging.getLogger(__name__)

class MinimumEnergyLoss(nn.Module):
    """Calculate the minimum energy of a system using the reverse iteration method."""

    def __init__(
        self,
        h_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        decay: float = np.infty,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize the minimum energy loss function."""
        super(MinimumEnergyLoss, self).__init__()

        if isinstance(h_tensor, list):
            h_tensor = torch.stack(h_tensor)


        if decay <= 0:
            raise ValueError("decay should be strictly positive.")
        if not all(is_hermitian_torch(h_tensor[i]) for i in range(h_tensor.shape[0])):
            raise ValueError("h_tensor should be a tensor of Hermitian matrices.")
        

        self.weight_decay = decay
        self.device = device
        self.dtype = dtype
        self.initializer()

        if h_tensor.dtype != dtype: #raise warning  
            logger.warning(f"h_tensor is not of type {dtype}. It will be converted to {dtype}.")
        self.h_tensor = h_tensor.clone().detach().to(dtype).to(device)
        E, V = torch.linalg.eigh(self.h_tensor)
        self.offset = E.max(dim=1).values
        self.h_tensor -= self.offset[:, None, None] * torch.eye(h_tensor.shape[-1], device=device, dtype=dtype)
        self.shift_origin_offset = self.offset - E.min(dim=1).values
        self.X = V[:, :, 0]

        logger.info(f"Initialized MinimumEnergyLoss")
        logger.info(f"\tnumber of local hamiltonians: {self.h_tensor.shape[0]}")
        logger.info(f"\tdtype: {self.dtype}")
        logger.info(f"\tdevice: {self.device}")
        logger.info(f"\tweight decay: {self.weight_decay}")
        logger.info(f"\tinitial weight: {self.weight}")
        logger.info(f"\tshift_origin_offset value: {self.shift_origin_offset}")

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """Return loss value for the minimum eigen loss."""
        if U.dtype != self.dtype:
            raise ValueError(f"U should be of type {self.dtype}.")
        loss = torch.tensor(0, dtype=torch.float64, device=self.device)
        for i in range(self.h_tensor.shape[0]):
            loss += self.minimum_energy_loss(self.h_tensor[i], U) - self.shift_origin_offset[i]

        self.weight *= np.exp(-1 / self.weight_decay)
        return torch.abs(loss)

    def minimum_energy_loss(self, H: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Calculate the minimum energy of a system with ED."""
        A = U @ H @ U.H
        result_abs = self.get_stoquastic(A)
        negativity = torch.abs(A - result_abs).mean() / H.shape[0] if self.weight > 0 else 0
        E = torch.linalg.eigvalsh(result_abs)
        return -E[0] + self.weight * negativity

    def stoquastic(self, A: torch.Tensor):
        """Change the sign of all non-diagonal elements into negative."""
        return -torch.abs(A)

    def get_stoquastic(self, A: torch.Tensor) -> torch.Tensor:
        """Return the stoquastic matrix of a given matrix."""
        return self.stoquastic(A)

    def initializer(self) -> None:
        """Initialize the unitary matrix."""
        self.weight = 0 if self.weight_decay == np.infty else 1


class SystemQuasiEnergyLoss(MinimumEnergyLoss):
    def __init__(
        self,
        h_list: Union[List[torch.Tensor], List[np.ndarray]],
        device: torch.device = torch.device("cpu"),
    ):
        # h_list must be 1 element list
        if len(h_list) != 1:
            raise ValueError("There should be only one local Hamiltonian for SystemQuasiEnergyLoss")

        super(SystemQuasiEnergyLoss, self).__init__(h_list, device)

    def forward(self, U: torch.Tensor, nr: int = 2, regularizer: float = 0.0) -> torch.Tensor:
        """Return loss value for the System Quasi Energy Loss.

        Quasi Energy loss is approximated value of System Energy Loss.
        """
        if nr < 1:
            raise ValueError("nr should be positive integer.")

        X = self.X[0]
        H = self.h_list[0]
        offset = self.shift_origin_offset[0]
        return self.system_quasi_energy_loss(U, X, H, nr, regularizer) - offset

    def system_quasi_energy_loss(
            self,
            U: torch.Tensor,
            X: torch.Tensor,
            H: torch.Tensor,
            nr: int = 2,
            regularizer: float = 0.0) -> torch.Tensor:
        """Return loss value for the System Quasi Energy Loss.

        Quasi Energy loss is approximated value of System Energy Loss.
        """
        if nr < 1:
            raise ValueError("nr should be positive integer.")

        SUx = torch.abs(U @ X)
        SUH = self.get_stoquastic(U @ H @ U.T)
        y = SUx
        for _ in range(10):
            y = SUH @ y
            if _ % 4 == 0:
                y = y / torch.norm(y)

        quasi_Sgs = torch.abs(y / torch.norm(y))
        z = SUH @ quasi_Sgs
        res = -(quasi_Sgs @ z)
        res += regularizer * (1 - torch.abs(quasi_Sgs.dot(z) / torch.norm(z)))
        return res

    def initializer(self, U: Union[torch.Tensor, None] = None) -> None:
        """Initialize the unitary matrix.

        If the loss is mel, just initialze decay weight to 1 if decay > 0.
        """
        return


class SystemStoquastic(MinimumEnergyLoss):
    """Calculate the minimum energy of a system using the reverse iteration method."""

    def __init__(
        self,
        h_list: Union[List[torch.Tensor], List[np.ndarray]],
        device: torch.device = torch.device("cpu"),
        # n : after "decay" step, the regularization term will e^(-1) times smaller
    ):
        """System Stoquastic Loss."""
        super(SystemStoquastic, self).__init__(h_list, device, decay=0)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """Return loss value for the minimum eigen loss. (Or minimum eigen local loss).

        Local Hamiltonian will be shifted so that miminum value will be 0
        """
        loss = torch.zeros(1, device=U.device)
        for i in range(len(self.h_list)):
            H = self.h_list[i]
            loss += (H - self.get_stoquastic(U @ H @ U.T)).sum()
        return torch.abs(loss)

    def initializer(self, U: Union[torch.Tensor, None] = None) -> None:
        """Initialize the unitary matrix.

        If the loss is mel, just initialze decay weight to 1 if decay > 0.
        """
        self.weight = 1 if self.weight_decay > 0 else 0
        return
