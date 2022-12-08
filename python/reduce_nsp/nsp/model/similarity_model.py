import numpy as np
import torch
from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import copy
import abc
from typing import Union

from ..utils.func import *
from .. import n_sphere
from .unitary_model import BaseMatrixGenerator


class SlRiemanGenerator(BaseMatrixGenerator):

    """
    class for generating square matrix
    initial matrix is unitary matrix or orthogonal 

    W <- W - \Delta S
    """

    def __init__(self, D, dtype = np.float64, seed = None):
        super().__init__(D, dtype, seed)
        self.reset_params(seed=seed)

        if self._complex:
            raise NotImplementedError("similarity transformation for complex matrix is not implemented")


    def _get_n_params(self) -> int:
        D = self.D
        n_params = D*D
        return int(n_params)


    def _get_matrix(self, params = None):
        if params is None:
            params = self._params
        if len(params) != len(self._params):
            raise ValueError("given parameters are not appropriate len(params) = {}".format(len(params)))
        return view_tensor(params[:self.D**2], [self.D]*2)

    def reset_params(self, seed = None, unitary=False):
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if self._complex:
            # randnMatrix = np.random.randn(self.D, self.D) + 1j*np.random.randn(self.D, self.D)
            pass
        else:
            randnMatrix = np.random.randn(self.D, self.D)
        if unitary:
            Q, R = np.linalg.qr(randnMatrix)
            haar_orth = Q.dot(np.diag(np.diag(R)/np.abs(np.diag(R))))
            haar_orth /= np.linalg.det(haar_orth)   
            self.set_params(haar_orth.reshape(-1))
        else:
            if self._complex:
                # randnMatrix = np.random.randn(self.D, self.D) + 1j*np.random.randn(self.D, self.D)
                pass
            else:
                randnMatrix = np.random.randn(self.D, self.D)
                randnMatrix /= abs(np.linalg.det(randnMatrix)) ** (1/self.D)
                self.set_params(randnMatrix.reshape(-1))


    def _inv(self, U, buffer=False):
        if buffer:
            return self.U_inv
        self.U_inv = matinv(U)
        return self.U_inv