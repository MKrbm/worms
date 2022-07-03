import numpy as np
from pyrsistent import v
from scipy.linalg import expm
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import copy
from ..utils.func import *
from ..model.unitary_model import BaseMatrixGenerator
from ..loss.max_eig import BaseMatirxLoss

import abc
from typing import Union



class BaseMatrixSolver(abc.ABC):

    model : BaseMatrixGenerator
    loss : BaseMatirxLoss

    def __init__(self, model:BaseMatrixGenerator, loss:BaseMatirxLoss):
        self.model = model
        self.loss = loss
        self._type = model._type
        self.loss._type_convert(self._type)

        if not issubclass(type(model), BaseMatrixGenerator):
            raise TypeError("model need to inherit BaseMatrixGenerator")

        if not issubclass(type(loss), BaseMatirxLoss):
            raise TypeError("model need to inherit base_ulf")

        if not (callable(loss)):
            raise AttributeError("loss is required to be callable")
        
    @abc.abstractmethod
    def __call__(self, x):
        
        """
        return loss fucntion
        """


class SymmSolver(BaseMatrixSolver):

    def __init__(self, model, loss, zero_origin = True):
        super().__init__(model, loss)
        self.zero_origin = zero_origin
        
            

    def __call__(self, x):
        x = convert_type(x, self._type)
        if (self.zero_origin):
            print("Hi")
            return self.loss([self.model.matrix(x)]*self.loss._n_unitaries) - self.loss.target
        return convert_type(self.loss([self.model.matrix(x)]*self.loss._n_unitaries), np.ndarray)

    
