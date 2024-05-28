import torch
import numpy as np
from ..model import random_unitary_matrix, UnitaryRiemann

from ..functions import check_is_unitary_torch

import pytest


class TestUnitaryRieman:

    def setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def setup_method(self):
        torch.manual_seed(42)  # Set the seed for PyTorch
        np.random.seed(42)
        self.setup_device()

    def test_wrong_dtype_error(self):
        with pytest.raises(ValueError):
            UnitaryRiemann(H_size=8, unitary_size=2, dtype=torch.int32)

    def test_h_size_not_power_of_unitary_size_error(self):
        with pytest.raises(ValueError):
            UnitaryRiemann(H_size=10, unitary_size=2)

    def test_inconsistent_u0_error(self):
        # Test case 1: When u0 is not unitary
        u0_not_unitary = torch.randn(2, 2)  # Random matrix, not unitary
        with pytest.raises(ValueError):
            UnitaryRiemann(H_size=8, unitary_size=2, u0=u0_not_unitary)

        # Test case 2: When u0 is not a square matrix
        u0_not_square = torch.randn(2, 3)  # Non-square matrix
        with pytest.raises(ValueError):
            UnitaryRiemann(H_size=8, unitary_size=2, u0=u0_not_square)

        # Test case 3: When dtype is inconsistent
        u0_inconsistent_dtype = torch.randn(
            2, 2, dtype=torch.float32
        )  # Different dtype
        with pytest.raises(ValueError):
            UnitaryRiemann(
                H_size=8, unitary_size=2, u0=u0_inconsistent_dtype, dtype=torch.float64
            )

    def test_unitary_output(self):
        model = UnitaryRiemann(
            H_size=8, unitary_size=2, dtype=torch.float64, device=self.device
        )
        U = model.forward()
        assert check_is_unitary_torch(U)
        assert U.shape == (8, 8)
        assert U.dtype == torch.float64
        assert U.device == self.device

        model = UnitaryRiemann(
            H_size=16, unitary_size=4, dtype=torch.complex128, device=self.device
        )
        U = model.forward()
        assert check_is_unitary_torch(U)
        assert U.shape == (16, 16)
        assert U.dtype == torch.complex128
        assert U.device == self.device

    def test_riemannian_gd(self):
        H = torch.randn(8, 8, dtype=torch.complex64, device=self.device)
        model = UnitaryRiemann(
            H_size=8, unitary_size=2, dtype=torch.complex64, device=self.device
        )
        U = model.forward()

        def loss_fn(U, H):
            loss = torch.mm(U, torch.mm(H, U.H))
            loss = torch.abs(loss).sum()
            return loss

        loss = loss_fn(U, H)
        loss.backward()
        model.update_riemannian_gradient()

        step = 0.001
        pdata_old = []
        for p in model.u:
            assert p.grad is not None
            assert p.grad.device == self.device
            pdata_old.append(p.data.clone().detach())
            rg = p.grad.clone()
            rg /= torch.norm(rg)
            p.data[:] = torch.linalg.matrix_exp(-step * rg) @ p.data
            assert check_is_unitary_torch(p.data)

        Up = model.forward()
        assert check_is_unitary_torch(Up)
        loss_p = loss_fn(Up, H)
        assert loss_p < loss

        for _ in range(100):
            # Create a skew-Hermitian matrix
            skew_hermitian = torch.randn(
                2, 2, dtype=torch.complex64, device=self.device
            )
            skew_hermitian = skew_hermitian - skew_hermitian.T.conj()
            skew_hermitian /= torch.norm(skew_hermitian)
            # print(skew_hermitian)

            # Update pdata_old with the generated skew-Hermitian matrix
            for i, p in enumerate(model.u):
                p.data[:] = (
                    torch.linalg.matrix_exp(-step * skew_hermitian) @ pdata_old[i]
                )

            # Recalculate the loss after updating with skew-Hermitian matrix
            Up_updated = model.forward()
            loss_updated = loss_fn(Up_updated, H)
            assert (
                loss_updated > loss_p
            ), "Updated loss should be greater than previous loss_p"

