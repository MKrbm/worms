import torch
import numpy as np
import pytest
import logging
from typing import List

from ..lattice import FF
from ..utils import sum_ham
from ..rms_torch import UnitaryRieman, MinimumEnergyLoss, Adam, check_is_unitary_torch


logger = logging.getLogger(__name__)



class TestOptimizeFF1D:
    def setup_method(self):

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.params = {
            "sps": 3,
            "rank": 2,
            "dimension": 1,
            "seed": 42,
            "lt": 1
        }
    
    def test_optimize_orthogonal(self):
        h, sps = FF.local(self.params)
        h_torch = torch.tensor(h, dtype=torch.float64)

        model = UnitaryRieman(H_size=h.shape[1], unitary_size=sps, dtype=torch.float64, device=self.device)

        # Calculate loss with MinimumEnergyLoss and do backward
        mel = MinimumEnergyLoss(h_tensor=h_torch, device=self.device, dtype=torch.float64)

        step = 0.001
        num_iterations = 10

        optimizer = Adam(model.parameters(), lr=step, betas=(0.9, 0.999))

        prev_loss = mel(model.forward()).item()
        for i in range(num_iterations):
            optimizer.zero_grad()
            U = model.forward()
            loss = mel(U)
            loss.backward()
            model.update_riemannian_gradient()
            optimizer.step()
            current_loss = loss.item()
            assert current_loss <= prev_loss, f"Loss increased at iteration {i+1}"
            prev_loss = current_loss
            logger.debug(f"Loss at iteration {i+1}: {current_loss}")

    def test_optimize_unitary(self):
        h, sps = FF.local(self.params)
        h_torch = torch.tensor(h, dtype=torch.complex128)

        model = UnitaryRieman(H_size=h.shape[1], unitary_size=sps, dtype=torch.complex128, device=self.device)

        # Calculate loss with MinimumEnergyLoss and do backward
        mel = MinimumEnergyLoss(h_tensor=h_torch, device=self.device, dtype=torch.complex128)

        step = 0.001
        num_iterations = 10

        optimizer = Adam(model.parameters(), lr=step, betas=(0.9, 0.999))

        prev_loss = mel(model.forward()).item()
        for i in range(num_iterations):
            optimizer.zero_grad()
            U = model.forward()
            loss = mel(U)
            loss.backward()
            model.update_riemannian_gradient()
            optimizer.step()
            current_loss = loss.item()
            assert current_loss <= prev_loss, f"Loss increased at iteration {i+1}"
            prev_loss = current_loss
            logger.debug(f"Loss at iteration {i+1}: {current_loss}")

    def test_optimize_unitary_orthogonal_init(self):
        h, sps = FF.local(self.params)
        h_torch = torch.tensor(h, dtype=torch.float64)

        modelR = UnitaryRieman(H_size=h.shape[1], unitary_size=sps, dtype=torch.float64, device=self.device)
        u0 = [p.detach().clone().to(torch.complex128) for p in modelR.u][0]
        modelC = UnitaryRieman(H_size=h.shape[1], unitary_size=sps, dtype=torch.complex128, device=self.device, u0=u0)

        # Calculate loss with MinimumEnergyLoss and do backward
        melR = MinimumEnergyLoss(h_tensor=h_torch, device=self.device, dtype=torch.float64, decay=1e-2)
        melC = MinimumEnergyLoss(h_tensor=h_torch.to(torch.complex128), device=self.device, dtype=torch.complex128, decay=1e-2)

        step = 0.001
        num_iterations = 10

        optimizerR = Adam(modelR.parameters(), lr=step, betas=(0.9, 0.999))
        optimizerC = Adam(modelC.parameters(), lr=step, betas=(0.9, 0.999))

        for i in range(num_iterations):
            optimizerR.zero_grad()
            optimizerC.zero_grad()
            UR = modelR.forward()
            UC = modelC.forward()
            lossR = melR(UR)
            lossC = melC(UC)
            lossR.backward()
            lossC.backward()
            modelR.update_riemannian_gradient()
            modelC.update_riemannian_gradient()
            optimizerR.step()
            optimizerC.step()
            assert lossR.item() == pytest.approx(lossC.item(), abs=1e-6)
            logger.debug(f"Loss at iteration {i+1}: {lossR.item()} and {lossC.item()}")

