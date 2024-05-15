import torch
import numpy as np
import pytest
import logging
from typing import List

from ..loss import MinimumEnergyLoss


logger = logging.getLogger(__name__)



class TestMinimumEnergyLoss:

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

        # Create a tensor of random Hermitian matrices
        h_tensor = torch.randn(3, 4, 4, device=self.device, dtype=torch.float64)
        h_tensor = h_tensor + h_tensor.transpose(-2, -1)  # Make the tensor Hermitian

        self.h_tensor = h_tensor

        # Create a tensor of negative identity matrices
        self.h_tensor_an = -torch.eye(4, device=self.device, dtype=torch.float64).repeat(3, 1, 1)

        # Create a tensor of shifted Hermitian matrices
        h_tensor_shifted = torch.randn(3, 4, 4, device=self.device, dtype=torch.float64)
        h_tensor_shifted = h_tensor_shifted + h_tensor_shifted.transpose(-2, -1)
        max_eigenvalues = torch.linalg.eigvalsh(h_tensor_shifted).max(dim=1).values
        self.h_tensor_shifted = h_tensor_shifted - max_eigenvalues[:, None, None] * torch.eye(4, device=self.device)

    def test_initializer(self):
        mel = MinimumEnergyLoss(h_tensor=self.h_tensor, device=self.device)
        for original_h, h, offset, shift_origin_offset in zip(self.h_tensor, mel.h_tensor, mel.offset, mel.shift_origin_offset):
            # Check if the matrix is shifted by its maximum eigenvalue
            E, _ = torch.linalg.eigh(original_h)
            max_eigenvalue = E[-1]

            assert offset.dtype == torch.float64, "Tensor is not of the correct type: torch.float64"
            assert shift_origin_offset.dtype == torch.float64, "Tensor is not of the correct type: torch.float64"
            assert h.device.type == self.device.type, f"Tensor is not on the correct device: {self.device}"
            assert h.dtype == torch.float64, "Tensor is not of the correct type: torch.float64"
            assert torch.allclose(h, original_h - offset * torch.eye(4, device=self.device)), \
                "Matrix is not shifted correctly by the origin offset."
            assert torch.allclose(offset, max_eigenvalue), "Offset is not the maximum eigenvalue."

    def test_non_hermitian(self):
        # Create a non-Hermitian matrix tensor
        non_hermitian_tensor = torch.randn(3, 4, 4, device=self.device)
        with pytest.raises(ValueError):
            MinimumEnergyLoss(h_tensor=non_hermitian_tensor, device=self.device)

    def test_weight_decay_effect(self):
        # Test the effect of weight decay
        mel = MinimumEnergyLoss(h_tensor=self.h_tensor, device=self.device, decay=0.1)
        initial_weight = mel.weight
        mel.forward(torch.eye(4, device=self.device, dtype=torch.float64))  # Trigger weight update
        assert mel.weight < initial_weight, "Weight did not decay as expected."
        assert mel.weight == 1 * np.exp(-1 / mel.weight_decay), "Weight did not decay as expected."

    def test_weight_zero_when_decay_zero(self):
        # Test that weight is zero when decay is zero
        mel = MinimumEnergyLoss(h_tensor=self.h_tensor, device=self.device)
        assert mel.weight == 0, "Weight should be zero when decay is zero."
    
    def test_return_toy(self):
        mel = MinimumEnergyLoss(h_tensor=self.h_tensor_an, device=self.device)
        I = torch.eye(4, device=self.device, dtype=torch.float64)
        loss = mel(I)
        assert loss.dtype == self.h_tensor_an.dtype, "Tensor is not of the correct type: torch.float64"
        assert torch.allclose(loss.cpu(), torch.tensor(0.0).double()), "Loss must be 0 for identity hamiltonian."
        assert loss.device.type == self.device.type, f"Tensor is not on the correct device: {self.device}"

        # Create a tensor of random matrices and make them all negative definite
        h_tensor_all_neg = torch.randn(3, 4, 4, device=self.device, dtype=torch.float64)
        h_tensor_all_neg = -torch.abs(h_tensor_all_neg + h_tensor_all_neg.transpose(-2, -1))

        mel = MinimumEnergyLoss(h_tensor=h_tensor_all_neg, device=self.device)
        I = torch.eye(4, device=self.device, dtype=torch.float64)
        loss = mel(I)
        assert torch.allclose(loss.cpu(), torch.tensor(0.0).double(), atol=1e-7), "Since the matrices are all negative, the loss must be 0."

    def test_loss_computation(self):
        # Calculate the sum of the largest eigenvalues of self.h_tensor_shifted
        loss0 = torch.tensor(0.0).double()
        loss_an0 = torch.tensor(0.0).double()
        for h in self.h_tensor_shifted:
            h_an = -torch.abs(h)
            loss0 += torch.linalg.eigvalsh(h)[0].double().cpu()
            loss_an0 += torch.linalg.eigvalsh(h_an)[0].double().cpu()
        expected_loss = loss0 - loss_an0
        mel = MinimumEnergyLoss(h_tensor=self.h_tensor_shifted, device=self.device)
        for i in range(len(self.h_tensor_shifted)):
            assert torch.allclose(mel.offset[i], torch.tensor(0.0).double(), atol=1e-6), "Offset should be 0 for all matrices."
        I = torch.eye(4, device=self.device, dtype=torch.float64)
        computed_loss = mel(I).detach().cpu()
        print(f"Computed loss: {computed_loss}")
        print(f"Expected loss: {expected_loss}")
        # Check if the computed loss matches the expected loss
        assert torch.isclose(computed_loss, expected_loss), "Computed loss does not match expected loss."



class TestMinimumEnergyLossComplex:

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

        # Create a tensor of random Hermitian matrices with complex values
        real_part = torch.randn(3, 4, 4, device=self.device, dtype=torch.float64)
        imag_part = torch.randn(3, 4, 4, device=self.device, dtype=torch.float64)
        h_tensor_complex = torch.complex(real_part, imag_part)
        h_tensor_complex = h_tensor_complex + h_tensor_complex.transpose(-2, -1).conj()  # Make the tensor Hermitian

        self.h_tensor_complex = h_tensor_complex

        # Create a tensor of shifted Hermitian matrices with complex values
        h_tensor_shifted_complex = torch.complex(real_part, imag_part)
        h_tensor_shifted_complex = h_tensor_shifted_complex + h_tensor_shifted_complex.transpose(-2, -1).conj()
        max_eigenvalues = torch.linalg.eigvalsh(h_tensor_shifted_complex).max(dim=1).values
        self.h_tensor_shifted_complex = h_tensor_shifted_complex - max_eigenvalues[:, None, None] * torch.eye(4, device=self.device, dtype=torch.float64)

    def test_complex_hermitian(self):
        mel = MinimumEnergyLoss(h_tensor=self.h_tensor_complex, device=self.device, dtype=torch.complex128)
        for original_h, h, offset in zip(self.h_tensor_complex, mel.h_tensor, mel.offset):
            # Check if the matrix is shifted by its maximum eigenvalue
            E, _ = torch.linalg.eigh(original_h)
            max_eigenvalue = E[-1]

            assert offset.dtype == torch.float64, "Offset is not of the correct type: torch.float64"
            assert h.device.type == self.device.type, f"Tensor is not on the correct device: {self.device}"
            assert h.dtype == torch.complex128, "Tensor is not of the correct type: torch.complex128"
            assert torch.allclose(h, original_h - offset * torch.eye(4, device=self.device, dtype=torch.complex128)), \
                "Matrix is not shifted correctly by the origin offset."
            assert torch.allclose(offset, max_eigenvalue), "Offset is not the maximum eigenvalue."
        
        I = torch.eye(4, device=self.device, dtype=torch.complex128)
        loss = mel(I)
        assert loss.dtype == torch.float64, "Loss is not of the correct type: torch.float64"
        assert loss.device.type == self.device.type, f"Tensor is not on the correct device: {self.device}"

    def test_complex_non_hermitian(self):
        # Create a non-Hermitian matrix tensor
        non_hermitian_tensor = torch.randn(3, 4, 4, device=self.device, dtype=torch.complex128)
        non_hermitian_tensor = non_hermitian_tensor + non_hermitian_tensor.transpose(-2, -1)
        with pytest.raises(ValueError):
            MinimumEnergyLoss(h_tensor=non_hermitian_tensor, device=self.device, dtype=torch.complex128)
    
    def test_dtype_mismatch(self):
        with pytest.raises(ValueError):
            mel = MinimumEnergyLoss(h_tensor=self.h_tensor_complex, device=self.device, dtype=torch.complex64)
            I = torch.eye(4, device=self.device, dtype=torch.float64)
            mel(I)

    def test_loss_computation_complex(self):
        # Calculate the sum of the largest eigenvalues of self.h_tensor_shifted_complex
        loss = torch.tensor(0.0).double()
        loss_stoq = torch.tensor(0.0).double()
        for h in self.h_tensor_shifted_complex:
            h_stoq = -torch.abs(h)
            loss += torch.linalg.eigvalsh(h)[0].double().cpu()
            loss_stoq += torch.linalg.eigvalsh(h_stoq)[0].double().cpu()
        expected_loss = loss - loss_stoq
        mel = MinimumEnergyLoss(h_tensor=self.h_tensor_shifted_complex, device=self.device, dtype=torch.complex128)
        I = torch.eye(4, device=self.device, dtype=torch.complex128)
        computed_loss = mel(I).detach().cpu()
        print(f"Computed loss: {computed_loss}")
        print(f"Expected loss: {expected_loss}")
        # Check if the computed loss matches the expected loss
        assert torch.isclose(computed_loss, expected_loss), "Computed loss does not match expected loss."

    def test_complex_shifted_comparison(self):
        h_tensor_real = self.h_tensor_shifted_complex.real
        h_tensor_real_complex = h_tensor_real.clone().detach().to(torch.complex128)
        mel_real = MinimumEnergyLoss(h_tensor=h_tensor_real, device=self.device, dtype=torch.float64)
        mel_complex = MinimumEnergyLoss(h_tensor=h_tensor_real_complex, device=self.device, dtype=torch.complex128)
        I_complex = torch.eye(4, device=self.device, dtype=torch.complex128)
        loss_complex = mel_complex(I_complex)
        loss_real = mel_real(I_complex.real)
        assert torch.allclose(loss_complex, loss_real), "Losses from complex and real shifted tensors do not match."

