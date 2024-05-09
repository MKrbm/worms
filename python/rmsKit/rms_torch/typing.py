from dataclasses import dataclass
import torch
from typing import Union, Tuple, List, Optional 


@dataclass
class ComplexMat:
    real: torch.Tensor
    imag: torch.Tensor

    def __post_init__(self):
        """
        Post-initialization to check if the matrix is square and if the dimensions of real and imaginary parts match.
        """
        if self.real.shape[0] != self.real.shape[1] or self.imag.shape[0] != self.imag.shape[1]:
            raise ValueError("Both real and imaginary parts must be square matrices.")
        if self.real.shape != self.imag.shape:
            raise ValueError("Real and imaginary parts must have the same dimensions.")



def is_complex(dtype: torch.dtype) -> bool:
    return dtype in [torch.complex64, torch.complex128]

def is_numerical(dtype: torch.dtype) -> bool:
    return dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]
