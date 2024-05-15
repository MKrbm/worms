import torch
import numpy as np
import pytest
import logging

from ..model import UnitaryRieman
from ..loss import MinimumEnergyLoss
from ..functions import check_is_unitary_torch


logger = logging.getLogger(__name__)

class TestRiemannianOptimization:

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

        # Generate a random symmetric Hamiltonian of size 16x16
        self.H = torch.randn(16, 16, dtype=torch.float64, device=self.device)
        self.H = (self.H + self.H.T) / 2

    def test_riemannian_optimization(self):
        # Initialize UnitaryRieman class with dtype=torch.float64
        model = UnitaryRieman(H_size=16, unitary_size=4, dtype=torch.float64, device=self.device)
        U = model.forward()

        # Calculate loss with MinimumEnergyLoss
        mel = MinimumEnergyLoss(h_tensor=self.H.unsqueeze(0), device=self.device, dtype=torch.float64)
        loss = mel(U)

        step = 0.001
        num_iterations = 100

        for i in range(num_iterations):
            loss.backward()

            # Calculate Riemannian gradient
            model.update_riemannian_gradient()

            for p in model.u:
                assert p.grad is not None
                assert p.grad.device == self.device 
                rg = p.grad.clone()
                p.data[:] = torch.linalg.matrix_exp(-step * rg) @ p.data
                assert check_is_unitary_torch(p.data)

            # Recalculate the loss after the Riemannian optimization step
            U = model.forward()
            assert check_is_unitary_torch(U)
            loss_prev = loss.item()
            loss = mel(U)

            # Check if the loss is reduced compared to the previous iteration
            assert loss.item() <= loss_prev, f"Riemannian optimization should reduce the loss at iteration {i+1}"
            logger.debug(f"Loss at iteration {i+1}: {loss.item()}")

            # Reset gradients for the next iteration
            model.zero_grad()

    def test_comp_float_comp(self):
        modelR = UnitaryRieman(H_size=16, unitary_size=4, dtype=torch.float64, device=self.device)
        u0 = [p.detach().clone().to(torch.complex128) for p in modelR.u][0]
        modelC = UnitaryRieman(H_size=16, 
                            unitary_size=4,
                            u0=u0,
                            dtype=torch.complex128, device=self.device)
        UC = modelC.forward()
        UR = modelR.forward()

        assert torch.allclose(UC, UR.to(torch.complex128), rtol=1e-6)

        melC = MinimumEnergyLoss(h_tensor=self.H.unsqueeze(0), device=self.device, dtype=torch.complex128)
        lossC = melC(UC)
        lossC.backward()
        modelC.update_riemannian_gradient()
        uc_grad = [p.grad for p in modelC.u]

        melR = MinimumEnergyLoss(h_tensor=self.H.unsqueeze(0), device=self.device, dtype=torch.float64)
        lossR = melR(UR)
        lossR.backward()
        modelR.update_riemannian_gradient()
        ur_grad = [p.grad for p in modelR.u]

        assert torch.allclose(lossC, lossR, rtol=1e-6)
        for i in range(len(uc_grad)):
            assert torch.allclose(uc_grad[i], ur_grad[i].to(torch.complex128), rtol=1e-6)
            logger.debug(f"Gradient {i} is {uc_grad[i]} \n and \n {ur_grad[i]}")

    def test_rg_optimal(self):
        # Initialize UnitaryRieman class with dtype=torch.float64
        model = UnitaryRieman(H_size=16, unitary_size=4, dtype=torch.float64, device=self.device)
        U = model.forward()

        # Calculate loss with MinimumEnergyLoss and do backward
        mel = MinimumEnergyLoss(h_tensor=self.H.unsqueeze(0), device=self.device, dtype=torch.float64)
        loss = mel(U)
        loss.backward()

        # Calculate Riemannian gradient
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
        
        # Recalculate the loss after the Riemannian optimization step
        Up = model.forward()
        assert check_is_unitary_torch(Up)
        loss_p = mel(Up)

        num_random_directions = 100
        for _ in range(num_random_directions):
            # Create random skew-Hermitian matrices
            skew_hermitian_matrices = []
            for p in model.u:
                skew_hermitian = torch.randn_like(p.data)
                skew_hermitian = skew_hermitian - skew_hermitian.T
                skew_hermitian /= torch.norm(skew_hermitian)
                skew_hermitian_matrices.append(skew_hermitian)

            # Update pdata_old with the generated skew-Hermitian matrices
            for i, p in enumerate(model.u):
                p.data[:] = torch.linalg.matrix_exp(-step * skew_hermitian_matrices[i]) @ pdata_old[i]

            # Recalculate the loss after updating with skew-Hermitian matrices
            Up_updated = model.forward()
            loss_updated = mel(Up_updated)
            assert loss_updated >= loss_p, "Riemannian gradient should be the optimal direction"
            logger.debug(f"Loss after updating with random"
                         f"skew-Hermitian matrices: {loss_updated.item() - loss_p.item()}")
    
    def test_adam(self):
        model = UnitaryRieman(H_size=16, unitary_size=4, dtype=torch.complex128, device=self.device)
        U = model.forward()
        mel = MinimumEnergyLoss(h_tensor=self.H.unsqueeze(0),
                                device=self.device,
                                dtype=torch.complex128)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        num_iterations = 30
        prev_loss = float('inf')
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            U = model.forward()
            loss = mel(U)
            model.update_riemannian_gradient()
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            assert current_loss <= prev_loss, f"Loss increased at iteration {i+1}"
            prev_loss = current_loss
            
            logger.debug(f"Loss at iteration {i+1}: {current_loss}")


