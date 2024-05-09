import torch
import numpy as np
from ..functions import (
    kron_complex,
    matmal_complex,
    unitary_transform_complex,
    nonstoq_complex,
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

    def setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def setup_method(self):
        torch.manual_seed(42)  # Set the seed for PyTorch
        np.random.seed(42) 
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
        actual_real, actual_imag = matmal_complex(
            self.ur1, self.ui1, self.ur2, self.ui2
        )
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
        expected = -torch.abs(H_).sum()
        actual = nonstoq_complex(self.H, self.ur1, self.ui1)
        assert torch.isclose(actual, expected, atol=1e-8), "stoquastic_complex failed"

    def test_backward_ambient(self):
        ur1 = self.ur1.clone().detach().to(device="cpu").requires_grad_(True)
        ui1 = self.ui1.clone().detach().to(device="cpu").requires_grad_(True)

        loss = nonstoq_complex(self.H, ur1, ui1)

        loss.backward()

        # Check if gradients are computed
        assert ur1.grad is not None, "Gradient for ur1 not computed"
        assert ui1.grad is not None, "Gradient for ui1 not computed"

        eps = 1e-7 

        first_order_expected = (
            torch.linalg.norm(ur1.grad) ** 2 + torch.linalg.norm(ui1.grad) ** 2
        ).item() * eps

        ur2 = ur1 - eps * ur1.grad
        ui2 = ui1 - eps * ui1.grad
        loss2 = nonstoq_complex(self.H, ur2, ui2)
        first_order_numerical = -(loss2 - loss).item()


        assert np.isclose(
            first_order_expected, first_order_numerical, atol=1e-4
        ), "First order approximation incorrect"

    def test_cuda_backward_ambient(self):
        self.setup_device()
        assert self.device == torch.device("cuda")
        ur1 = self.ur1.clone().detach().to(device="cuda").requires_grad_(True)
        ui1 = self.ui1.clone().detach().to(device="cuda").requires_grad_(True)
        H = self.H.clone().detach().to(device="cuda")

        loss = nonstoq_complex(H, ur1, ui1)

        loss.backward()

        assert ur1.grad is not None, "Gradient for ur1 not computed"
        assert ui1.grad is not None, "Gradient for ui1 not computed"
