import torch 
import numpy 
from ..functions import (
    kron_complex,
    matmal_complex,
    unitary_transform_complex,
    nonstoq_complex
)
import pytest 

class TestComplexOperations:
    """
    This class contains tests for complex number operations implemented in the rms_torch library.
    It tests the following functions:
    - `kron_complex`: Kronecker product of two complex matrices.
    - `matmal_complex`: Matrix multiplication of two complex matrices.
    - `unitary_transform_complex`: Unitary transformation of a Hermitian matrix using a complex matrix.
    - `stoquastic_complex`: Calculation of the stoquastic property of a transformed Hermitian matrix.
    
    Attributes:
        ur1 (torch.Tensor): Real part of the first unitary matrix.
        ui1 (torch.Tensor): Imaginary part of the first unitary matrix.
        ur2 (torch.Tensor): Real part of the second unitary matrix.
        ui2 (torch.Tensor): Imaginary part of the second unitary matrix.
        u1 (torch.Tensor): First complex unitary matrix.
        u2 (torch.Tensor): Second complex unitary matrix.
        H (torch.Tensor): Hermitian matrix.
        h (torch.Tensor): Smaller Hermitian matrix used for transformations.
    """
    def setup_method(self):
        # Generate random complex tensors
        self.ur1 = torch.randn(2, 2, dtype=torch.float32)
        self.ui1 = torch.randn(2, 2, dtype=torch.float32)
        self.ur2 = torch.randn(2, 2, dtype=torch.float32)
        self.ui2 = torch.randn(2, 2, dtype=torch.float32)
        self.u1 = torch.complex(self.ur1, self.ui1)
        self.u2 = torch.complex(self.ur2, self.ui2)
        self.H = torch.randn(4, 4, dtype=torch.float32)
        self.H = 0.5 * (self.H + self.H.T)  # Making H Hermitian
        self.h = torch.randn(2, 2, dtype=torch.float32)
        self.h = 0.5 * (self.h + self.h.T)  # Making H Hermitian

    def test_kron_complex(self):
        expected = torch.kron(self.u1, self.u2)
        actual_real, actual_imag = kron_complex(self.ur1, self.ui1, self.ur2, self.ui2)
        actual = torch.complex(actual_real, actual_imag)
        assert torch.allclose(actual, expected, atol=1e-8), "kron_complex failed"

    def test_matmal_complex(self):
        expected = torch.matmul(self.u1, self.u2)
        actual_real, actual_imag = matmal_complex(self.ur1, self.ui1, self.ur2, self.ui2)
        actual = torch.complex(actual_real, actual_imag)
        assert torch.allclose(actual, expected, atol=1e-8), "matmal_complex failed"

    def test_unitary_transform(self):
        expected = self.u1 @ self.h.to(torch.complex64) @ self.u1.H
        actual_real, actual_imag = unitary_transform_complex(self.h, self.ur1, self.ui1)
        actual = torch.complex(actual_real, actual_imag)
        assert torch.allclose(actual, expected, atol=1e-8), "unitary_transform failed"

    def test_stoquastic_complex(self):
        U1 = torch.kron(self.u1, self.u1)
        H_ = self.H.to(torch.complex64)
        H_ = U1 @ H_ @ U1.H
        expected = - torch.abs(H_).sum()
        actual = nonstoq_complex(self.H, self.ur1, self.ui1)
        assert torch.isclose(actual, expected, atol=1e-8), "stoquastic_complex failed"
