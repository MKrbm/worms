from dataclasses import dataclass
import torch

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