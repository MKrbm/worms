import torch
import numpy as np
from torch import nn

# import math
from typing import Union, List
from .functions import is_hermitian_torch
import logging


class MinimumEnergyLoss(nn.Module):
    """Calculate the minimum energy of a system using the reverse iteration method."""

    def __init__(
        self,
        h_list: List[torch.Tensor],
        device: torch.device = torch.device("cpu"),
        decay: float = 0.1,
        # n : after "decay" step, the regularization term will e^(-1) times smaller
    ):
        """Initialize the minimum energy loss function."""
        super(MinimumEnergyLoss, self).__init__()

        logging.info("Initializing MinimumEnergyLoss")
        logging.info(f"\tnumber of local hamiltonians: {len(h_list)}")
        logging.info(f"\tdecay time: {decay}")
        self.shift_origin_offset = []
        self.h_list = []
        self.X = []
        self.weight = 1 if decay > 0 else 0
        self.offset = []
        if decay < 0:
            raise ValueError("decay should be positive.")
        self.weight_decay = decay
        for i in range(len(h_list)):
            if not is_hermitian_torch(h_list[i]):
                raise ValueError("h_list should be a list of Hermitian matrices.")
            self.h_list.append(h_list[i].to(device))
            E, V = torch.linalg.eigh(self.h_list[i])
            logging.info(f"\tmaximum energy of local hamiltonian {i}: {E[-1]:.3f}")
            logging.info(f"\tminimum energy of local hamiltonian {i}: {E[0]:.3f}")
            self.offset.append(E[-1])
            self.h_list[i][:] = self.h_list[i] - self.offset[i] * \
                torch.eye(h_list[i].shape[1], device=device)
            logging.info(f"\toffset of local hamiltonian {i}: {self.offset[i]:.3f}")
            self.X.append(V[:, 0].to(device))
            self.shift_origin_offset.append(self.offset[i] - E[0])

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """Return loss value for the minimum eigen loss. (Or minimum eigen local loss).

        Local Hamiltonian will be shifted so that miminum value will be 0
        """
        # add minimum energy of each local hamiltonian for calculating the total energy
        # initialize with torch tensor
        loss = torch.zeros(1, device=U.device)
        for i in range(len(self.h_list)):
            loss += self.minimum_energy_loss(self.h_list[i], U) - self.shift_origin_offset[i]

        self.weight = (self.weight * np.exp(-1 / self.weight_decay)) \
            if (self.weight_decay != 0) else 0
        return torch.abs(loss)

    def minimum_energy_loss(self, H: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Calculate the minimum energy of a system using the reverse iteration method.

        The first ground state is calculated using the eigendecomposition of the Hamiltonian.
        """
        A = U @ H @ U.T
        result_abs = self.get_stoquastic(A)

        # n: corresponds to caluclate energy of maximum superposition state
        negativity = torch.abs(A - result_abs).mean() / H.shape[0]
        try:
            E = torch.linalg.eigvalsh(result_abs)
        except RuntimeError:
            # If there are some errors during the eigen decomposition.
            result_abs = (result_abs + result_abs.T)/2
            E = torch.linalg.eigvalsh(result_abs)

        return - E[0] + self.weight * negativity

    def stoquastic(self, A: torch.Tensor):
        """Change the sign of all non-diagonal elements into negative.

        If A is already negative definite matrix,
        then just taking negative absolute value sufficies.
        """
        return -torch.abs(A)

    def get_stoquastic(self, A: torch.Tensor) -> torch.Tensor:
        """Return the stoquastic matrix of a given matrix.

        First, calculate the matrix A = U @ h @ U.T. Then apply stoquastic function.
        """
        return self.stoquastic(A)

    def initializer(self, U: Union[torch.Tensor, None] = None) -> None:
        """Initialize the unitary matrix.

        If the loss is mel, just initialze decay weight to 1 if decay > 0.
        """
        self.weight = 1 if self.weight_decay > 0 else 0
        return


class SystemQuasiEnergyLoss(MinimumEnergyLoss):
    def __init__(
        self,
        h_list: Union[List[torch.Tensor], List[np.ndarray]],
        device: torch.device = torch.device("cpu"),
    ):
        # h_list must be 1 element list
        if len(h_list) != 1:
            raise ValueError("There should be only one local Hamiltonian for SystemQuasiEnergyLoss")

        super(SystemQUasiEnergyLoss, self).__init__(h_list, device)

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
