"""UnitaryRiemann and UnitaryRiemanNonSym classes for PyTorch."""

import torch
from torch import nn
import numpy as np
import math
import logging
from typing import Union, Optional
from .functions import check_is_unitary_torch, riemannian_grad_torch
from .typing import is_complex, is_numerical

logger = logging.getLogger(__name__)

def random_unitary_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a random unitary matrix of size size x size in Haar measure."""
    if not is_complex(dtype):
        random_matrix = np.random.randn(size, size)
    else:
        random_matrix = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    q, _ = np.linalg.qr(random_matrix)
    return torch.tensor(q, dtype=dtype, device=device)

class UnitaryRiemann(nn.Module):
    """UnitaryRiemann class for PyTorch."""

    def __init__(
        self,
        H_size: int,
        unitary_size: int,
        device: torch.device = torch.device("cpu"),
        u0: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
        only_phase: bool = False,
    ):
        """Initialize UnitaryRiemann class."""
        super(UnitaryRiemann, self).__init__()

        if H_size <= 0 or unitary_size <= 0:
            raise ValueError(
                "Both H_size and unitary_size should be positive integers."
            )

        if not is_numerical(dtype):
            raise ValueError("The dtype is not correct.")
        if not np.emath.logn(unitary_size, H_size).is_integer():
            raise ValueError("H_size should be a power of unitary_size.")
        if u0 is not None:
            if u0.shape != (unitary_size, unitary_size):
                raise ValueError("u0 must be a square matrix of shape (unitary_size, unitary_size).")
            if u0.dtype != dtype:
                raise ValueError("The dtype of u0 does not match the specified dtype.")
            if not check_is_unitary_torch(X=u0):
                raise ValueError("u0 must be a unitary matrix.")
        if only_phase and not is_complex(dtype):
            logger.warning("dtype == float contains phase information. If you want to use only_phase, please set dtype == complex. only_phase will be set to False.")
            only_phase = False

        self.H_size = H_size
        self.unitary_size = unitary_size
        self.device = device
        self.u0 = u0
        self.dtype = dtype
        self.num_repeat = round(np.emath.logn(unitary_size, H_size))
        self.only_phase = only_phase

        self.initialize_params()

    def initialize_params(self):
        """Initialize parameters of UnitaryRiemann class."""
        # n_us = round(math.log2(self.H_size) / math.log2(self.unitary_size))
        if self.only_phase:
            dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
        else:
            dtype = self.dtype
        if self.u0 is None:
            self.u = nn.ParameterList(
                [
                    nn.Parameter(
                        random_unitary_matrix(self.unitary_size, self.device, dtype),
                        requires_grad=True,
                    )
                ]
            )
        else:
            self.u = nn.ParameterList(
                [
                    nn.Parameter(
                        self.u0.clone().detach().to(dtype).to(self.device),
                        requires_grad=True,
                    )
                ]
            )
        
        if self.only_phase:
            self.phase = nn.Parameter(torch.tensor(np.random.randn(), dtype=torch.float64, device=self.device), requires_grad=True)

    def reset_params(self, u0: Union[torch.Tensor, None] = None):
        """Reset parameters of UnitaryRiemann class."""
        if u0 is not None and u0.shape != (self.unitary_size, self.unitary_size):
            raise ValueError("The shape of u0 is not correct.")
        for i in range(len(self.u)):
            self.u[i].data[:] = (
                random_unitary_matrix(self.unitary_size, self.device, self.dtype)
                if u0 is None
                else u0.clone().detach().to(self.dtype).to(self.device)
            )
        if self.only_phase:
            self.phase.data[:] = torch.tensor(np.random.randn(), dtype=self.dtype, device=self.device)

    def forward(self) -> torch.Tensor:
        """Calculate kron of unitary matrix (result size must be H_size x H_size)."""
        U = self.u[0]
        if self.only_phase:
            U = U * torch.exp(1j * self.phase)
        for i in range(self.num_repeat - 1):
            U = torch.kron(U, self.u[0])
        return U
    
    def update_riemannian_gradient(self):
        """
        Update the Riemannian gradient of the unitary matrix for each parameter.
        """
        # for p in self.parameters():
        #     if p.grad is not None:
        #         if not self.is_params_phase(p):
        #             p.grad[:] = riemannian_grad_torch(p.data, p.grad)
        for i in range(len(self.u)):
            self.u[i].grad[:] = riemannian_grad_torch(self.u[i], self.u[i].grad)
    
    def update_phase(self, step: float):
        """
        Update the phase of the unitary matrix for each parameter.
        """
        if self.phase is None:
            raise ValueError("The phase is not defined.")
        if self.phase.grad is None:
            raise ValueError("The phase.grad is not initialized.")
        self.phase.data[:] = self.phase.data[:] - step * self.phase.grad[:]
    
    def is_params_phase(self, p : nn.Parameter):
        return p.shape == torch.Size([])
        

    


class UnitaryRiemanNonSym(nn.Module):
    """UnitaryRiemanNonSym class for PyTorch."""

    def __init__(
        self, H_size: int, unitary_size: int, device=torch.device("cpu"), u0_list=None
    ):
        """Initialize UnitaryRiemanNonSym class."""
        super(UnitaryRiemanNonSym, self).__init__()

        if H_size <= 0 or unitary_size <= 0:
            raise ValueError(
                "Both H_size and unitary_size should be positive integers."
            )

        self.H_size = H_size
        self.unitary_size = unitary_size
        self.device = device
        self.u0_list = u0_list

        self.initialize_params()

    def initialize_params(self):
        """Initialize parameters of UnitaryRiemanNonSym class."""
        n_us = round(math.log2(self.H_size) / math.log2(self.unitary_size))
        if self.u0_list is None:
            self.us = nn.ParameterList(
                [
                    nn.Parameter(
                        random_unitary_matrix(self.unitary_size, self.device, self.dtype),
                        requires_grad=True,
                    )
                    for _ in range(n_us)
                ]
            )
        elif len(self.u0_list) == 1:
            u0_tensor = torch.tensor(
                self.u0_list[0], dtype=torch.float64, device=self.device
            )
            self.us = nn.ParameterList(
                [nn.Parameter(u0_tensor, requires_grad=True) for _ in range(n_us)]
            )
        else:
            self.us = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor(u0, dtype=torch.float64, device=self.device),
                        requires_grad=True,
                    )
                    for u0 in self.u0_list
                ]
            )

    def reset_params(self):
        """Reset parameters of UnitaryRiemanNonSym class."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
            p.data = random_unitary_matrix(self.unitary_size, self.device, self.dtype)

    def forward(self) -> torch.Tensor:
        """Calculate kron of unitaries (result size must be H_size x H_size)."""
        U = self.us[0]
        # for u in self.us[1:]:
        #     U = torch.kron(U, u)
        U = torch.kron(U, self.us[1])
        U = torch.kron(U, self.us[2])
        U = torch.kron(U, self.us[3])

        return U
