"""UnitaryRiemann and UnitaryRiemanNonSym classes for PyTorch."""


import torch
from torch import nn
import numpy as np
import math
import logging
from typing import Union, Optional
import geoopt
from .functions import check_is_unitary_torch, riemannian_grad_torch
from .typing import is_complex, is_numerical

logger = logging.getLogger(__name__)

def _check_power_of(base: int, target: int) -> bool:
    """
    Check if 'target' is an integer power of 'base'.
    """
    # handle edge cases
    if base <= 1 or target <= 0:
        return False
    # Check if log_base(target) is an integer
    val = np.emath.logn(base, target)
    return np.isclose(val, round(val))


class UnitaryRiemann(nn.Module):
    """
    A class that builds a repeated Kronecker product of a single orthonormal (real "unitary")
    matrix to match the dimension H_size x H_size. Uses geoopt for manifold-constrained optimization.
    """

    def __init__(
        self,
        H_size: int,
        unitary_size: int,
        device: torch.device = torch.device("cpu"),
        u0: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
        manifold: geoopt.Manifold = geoopt.CanonicalStiefel(),
    ):
        """
        Initialize the UnitaryRiemann class.

        Args:
            H_size (int): The total dimension of the final Kronecker product matrix.
            unitary_size (int): The size of the single orthonormal matrix that will be repeated.
            device (torch.device, optional): Device on which the data is located.
            u0 (Optional[torch.Tensor], optional): An initial orthonormal (real unitary) matrix
                                                   of shape (unitary_size, unitary_size).
            dtype (torch.dtype, optional): Data type to use.
        """
        super().__init__()

        if H_size <= 0 or unitary_size <= 0:
            raise ValueError("Both H_size and unitary_size should be positive integers.")
        if not _check_power_of(unitary_size, H_size):
            raise ValueError("H_size should be a power of unitary_size.")

        self.H_size = H_size
        self.unitary_size = unitary_size
        self.device = device
        self.dtype = dtype

        # Number of times we must Kronecker-product the base matrix
        self.num_repeat = round(np.emath.logn(unitary_size, H_size))

        # Define a Stiefel manifold (real orthonormal constraint).
        # You can choose among geoopt.CanonicalStiefel, geoopt.EuclideanStiefel, etc.
        if manifold is None:
            self.manifold = geoopt.CanonicalStiefel()
        else:
            if not isinstance(manifold, (geoopt.CanonicalStiefel, geoopt.EuclideanStiefel, geoopt.EuclideanStiefelExact)):
                raise ValueError("Manifold must be one of: geoopt.CanonicalStiefel, geoopt.EuclideanStiefel,  geoopt.EuclideanStiefelExact")
            self.manifold = manifold

        # Construct the manifold parameter
        # shape = (unitary_size, unitary_size) for the real "unitary."
        if u0 is not None:
            if u0.shape != (self.unitary_size, self.unitary_size):
                raise ValueError(
                    "u0 must be of shape (unitary_size, unitary_size)."
                )
            # Optionally check if it's orthonormal:
            # if not torch.allclose(u0.T @ u0, torch.eye(self.unitary_size, dtype=dtype), atol=1e-7):
            #     raise ValueError("Provided u0 is not orthonormal (or nearly so).")

            u_data = u0.clone().detach().to(dtype=dtype, device=device)
        else:
            # Initialize with random orthonormal matrix from geoopt's Stiefel manifold
            # shape = (unitary_size, unitary_size)
            u_data = self.manifold.random((self.unitary_size, self.unitary_size), dtype=dtype, device=device)

        # Create a ManifoldParameter so that geoopt-optimizers will handle retractions properly
        self.u = geoopt.ManifoldParameter(u_data, manifold=self.manifold, requires_grad=True)

    def forward(self) -> torch.Tensor:
        """
        Compute the Kronecker product repeated self.num_repeat times.
        Returns a tensor of shape (H_size, H_size).
        """
        # Start with self.u
        U = self.u
        # Repeatedly compute the Kronecker product
        for _ in range(self.num_repeat - 1):
            U = torch.kron(U, self.u)
        return U

    def reset_params(self, u0: Optional[torch.Tensor] = None):
        """
        Reset parameters. If u0 is provided, use it; else randomly sample from the manifold.
        """
        if u0 is not None:
            if u0.shape != (self.unitary_size, self.unitary_size):
                raise ValueError("Provided u0 has incorrect shape.")
            u_data = u0.clone().detach().to(device=self.device, dtype=self.dtype)
            # (Optional) check orthonormality
            # ...
        else:
            # Random from the Stiefel manifold
            u_data = self.manifold.random((self.unitary_size, self.unitary_size),
                                          dtype=self.dtype, device=self.device)
        with torch.no_grad():
            self.u.data = u_data
    
    


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
