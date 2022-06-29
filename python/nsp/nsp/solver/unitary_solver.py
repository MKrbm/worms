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
from ..model.unitary_model import base_matrix_generator
from ..loss.max_eig import base_ulf

import abc
from typing import Union



class base_matrix_solver(abc.ABC):

    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        if not issubclass(type(model), base_matrix_generator):
            raise TypeError("model need to inherit base_matrix_generator")

        if not issubclass(type(loss), base_ulf):
            raise TypeError("model need to inherit base_ulf")

        if not (callable(loss)):
            raise AttributeError("loss is required to be callable")
        
    @abc.abstractmethod
    def __call__(self, x):
        
        """
        return loss fucntion
        """


class sym_solver(base_matrix_solver):

    def __init__(self, model, loss, zero_origin = True):
        super().__init__(model, loss)
        self.zero_origin = zero_origin
        
            

    def __call__(self, x):
        if (self.zero_origin):
            return self.loss([self.model.matrix(x)]*self.loss._n_unitaries) - self.loss.target
        return self.loss([self.model.matrix(x)]*self.loss._n_unitaries)

    
