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

        # Create a list of random Hermitian matrices
        self.h_list = [torch.randn(4, 4, device=self.device, dtype=torch.float64)  for _ in range(3)]
        self.h_list = [h + h.H for h in self.h_list]

        self.h_list_an : List[torch.Tensor] = [-torch.eye(4, device=self.device, dtype=torch.float64) for _ in range(3)]
        self.h_list_shifted : List[torch.Tensor] = [torch.randn(4, 4, device=self.device, dtype=torch.float64)  for _ in range(3)]
        self.h_list_shifted = [h + h.H for h in self.h_list_shifted]
        self.h_list_shifted = [h - torch.linalg.eigvalsh(h).max() * torch.eye(4, device=self.device) for h in self.h_list_shifted]

    def test_initializer(self):
        mel = MinimumEnergyLoss(h_list=self.h_list, device=self.device)
        for original_h, h, offset, shift_origin_offset in zip(self.h_list, mel.h_list, mel.offset, mel.shift_origin_offset):
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
        # Create a non-Hermitian matrix
        non_hermitian_matrix = torch.randn(4, 4, device=self.device)
        with pytest.raises(ValueError):
            MinimumEnergyLoss(h_list=[non_hermitian_matrix], device=self.device)

    def test_weight_decay_effect(self):
        # Test the effect of weight decay
        mel = MinimumEnergyLoss(h_list=self.h_list, device=self.device, decay=0.1)
        initial_weight = mel.weight
        mel.forward(torch.eye(4, device=self.device, dtype=torch.float64))  # Trigger weight update
        assert mel.weight < initial_weight, "Weight did not decay as expected."

    def test_weight_zero_when_decay_zero(self):
        # Test that weight is zero when decay is zero
        mel = MinimumEnergyLoss(h_list=self.h_list, device=self.device, decay=0)
        assert mel.weight == 0, "Weight should be zero when decay is zero."
    
    def test_return_toy(self):

        mel = MinimumEnergyLoss(h_list=self.h_list_an, device=self.device, decay=0.0)
        I = torch.eye(4, device=self.device, dtype=torch.float64)
        loss = mel(I)
        assert loss.dtype == self.h_list_an[0].dtype, "Tensor is not of the correct type: torch.float64"
        assert torch.allclose(loss.cpu(), torch.tensor(0.0).double()), "Loss must be 0 for identity hamiltonian."
        assert loss.device.type == self.device.type, f"Tensor is not on the correct device: {self.device}"


        h_list_all_neg = [torch.randn(4, 4, device=self.device, dtype=torch.float64)  for _ in range(3)]
        h_list_all_neg = [-torch.abs(h + h.H) for h in h_list_all_neg]
        mel = MinimumEnergyLoss(h_list=h_list_all_neg, device=self.device, decay=0.0)
        I = torch.eye(4, device=self.device, dtype=torch.float64)
        loss = mel(I)
        assert torch.allclose(loss.cpu(), torch.tensor(0.0).double(), atol=1e-7), "Since the matrices are all negative, the loss must be 0."


    def test_loss_computation(self):
        # Calculate the sum of the largest eigenvalues of self.h_list
        loss0 = torch.tensor(0.0).double()
        loss_an0 = torch.tensor(0.0).double()
        for h in self.h_list_shifted:
            h_an = -torch.abs(h)
            loss0 += torch.linalg.eigvalsh(h)[0].double().cpu()
            loss_an0 += torch.linalg.eigvalsh(h_an)[0].double().cpu()
        expected_loss = loss0 - loss_an0
        mel = MinimumEnergyLoss(h_list=self.h_list_shifted, device=self.device, decay=0.0)
        for i in range(len(self.h_list_shifted)):
            assert torch.allclose(mel.offset[i], torch.tensor(0.0).double(), atol=1e-6), "Offset should be 0 for all matrices."
        I = torch.eye(4, device=self.device, dtype=torch.float64)
        computed_loss = mel(I).detach().cpu()
        print(f"Computed loss: {computed_loss}")
        print(f"Expected loss: {expected_loss}")
        # Check if the computed loss matches the expected loss
        assert torch.isclose(computed_loss, expected_loss), "Computed loss does not match expected loss."

