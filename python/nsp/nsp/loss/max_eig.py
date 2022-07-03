from os import stat
import numpy as np
from pyrsistent import v
from scipy.linalg import expm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from ..utils.func import *
import abc
from typing import Union
from .base_class import BaseMatirxLoss


class MES(BaseMatirxLoss):
    """
    maximum eigenvalue of stoquastic map of matrix (U @ X @ U.T) 

    params
    ------
    act : if U is list of unitary matrix, the map will be like U[0] \otimes U[1] \otimes \cdots U[len(act)-1] @ X @ (c.c.)  . Each U[0].shape[0] == act[0]
    """
    def __init__(self, X, act):
        super().__init__(X, act)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.target = eigvalsh_(X)[-1]

    def forward(self, A):
        A = stoquastic(A)
        return eigvalsh_(A)[-1]

    @staticmethod
    def _inverse(U):
        return U.T.conj()

class ME(BaseMatirxLoss):
    """
    maximum eigenvalue of  matrix (U @ X @ U.T) 

    params
    ------
    act : if U is list of unitary matrix, the map will be like U[0] \otimes U[1] \otimes \cdots U[len(act)-1] @ X @ (c.c.)  . Each U[0].shape[0] == act[0]
    """
    def __init__(self, X, act):
        super().__init__(X, act)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.target = eigvalsh_(X)[-1]

    def forward(self, A):
        return eigvalsh_(A)[-1]
    
    @staticmethod
    def _inverse(U):
        return U.T.conj()


