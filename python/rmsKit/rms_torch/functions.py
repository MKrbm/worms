# * below for torch
import torch
import numpy as np
from typing import Any, Tuple, Union, List



def check_is_unitary_torch(X: torch.Tensor) -> bool:
    """
    Check if a given matrix X is unitary.
    
    A matrix is unitary if the product of X and its conjugate transpose is the identity matrix.
    
    Parameters:
        X (torch.Tensor): A square matrix to be checked for unitarity.
    
    Returns:
        torch.Tensor: A boolean tensor indicating if the matrix is unitary.
    """
    with torch.no_grad():
        I_prime = X @ X.H
        identity_matrix = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        return torch.allclose(I_prime, identity_matrix, atol=1e-6)

def is_hermitian_torch(X: torch.Tensor) -> bool:
    """
    Check if a given matrix X is Hermitian.
    
    A matrix is Hermitian if it is equal to its conjugate transpose.
    
    Parameters:
        X (torch.Tensor): A matrix to be checked for Hermitian property.
    
    Returns:
        bool: A boolean indicating if the matrix is Hermitian.
    """
    return torch.allclose(X, X.H, atol=1e-9)


def riemannian_grad_torch(u: torch.Tensor, euc_grad: torch.Tensor) -> torch.Tensor:
    """
    Compute the Riemannian gradient for a unitary matrix.
    
    Parameters:
        u (torch.Tensor): A unitary matrix.
        euc_grad (torch.Tensor): The Euclidean gradient at matrix u.
    
    Returns:
        torch.Tensor: The Riemannian gradient at matrix u.
    
    Raises:
        ValueError: If u is not unitary, or if the shapes or dtypes of u and euc_grad do not match.
    """
    if not check_is_unitary_torch(u):
        raise ValueError("u must be unitary matrix")
    if u.dtype != euc_grad.dtype:
        raise ValueError("dtype of u and euc_grad must be same")

    if u.shape != euc_grad.shape:
        raise ValueError("shape of u and euc_grad must be same")
    rg = euc_grad @ u.H
    return (rg - rg.H) / 2


def kron_complex(ur1: torch.Tensor, ui1: torch.Tensor, ur2: torch.Tensor,
                 ui2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Kronecker product of two complex matrices represented by their real and imaginary parts.
    
    Parameters:
        ur1 (torch.Tensor): Real part of the first matrix.
        ui1 (torch.Tensor): Imaginary part of the first matrix.
        ur2 (torch.Tensor): Real part of the second matrix.
        ui2 (torch.Tensor): Imaginary part of the second matrix.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The real and imaginary parts of the Kronecker product.
    """
    return torch.kron(ur1, ur2) - torch.kron(ui1, ui2), torch.kron(ur1, ui2) + torch.kron(ui1, ur2)


def matmal_complex(ur1: torch.Tensor, ui1: torch.Tensor, ur2: torch.Tensor,
                   ui2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform matrix multiplication on two complex matrices represented by their real and imaginary parts.
    
    Parameters:
        ur1 (torch.Tensor): Real part of the first matrix.
        ui1 (torch.Tensor): Imaginary part of the first matrix.
        ur2 (torch.Tensor): Real part of the second matrix.
        ui2 (torch.Tensor): Imaginary part of the second matrix.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The real and imaginary parts of the resulting matrix.
    """
    return ur1 @ ur2 - ui1 @ ui2, ur1 @ ui2 + ui1 @ ur2


def unitary_transform_complex(H: torch.Tensor, ur: torch.Tensor, ui: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a unitary transformation to a Hermitian matrix using a complex matrix represented by its real and imaginary parts.
    
    Parameters:
        H (torch.Tensor): Hermitian matrix to be transformed.
        ur (torch.Tensor): Real part of the unitary matrix.
        ui (torch.Tensor): Imaginary part of the unitary matrix.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The real and imaginary parts of the transformed matrix.
    """
    return ur @ H @ ur.T + ui @ H @ ui.T, - ur @ H @ ui.T + ui @ H @ ur.T

def nonstoq_complex(H: torch.Tensor, ur: torch.Tensor, ui: torch.Tensor) -> torch.Tensor:
    """
    Calculate the non-stoquasiticy of a Hermitian matrix after a unitary transformation using a complex matrix.
    
    Parameters:
        H (torch.Tensor): Hermitian matrix to be transformed.
        ur (torch.Tensor): Real part of the unitary matrix.
        ui (torch.Tensor): Imaginary part of the unitary matrix.
    
    Returns:
        torch.Tensor: The sum of the square roots of the squares of the real and imaginary parts of the transformed matrix.
    """
    Ur, Ui = kron_complex(ur, ui, ur, ui)
    Hr, Hi = unitary_transform_complex(H, Ur, Ui)
    return - torch.sqrt(Hr**2 + Hi**2).sum()