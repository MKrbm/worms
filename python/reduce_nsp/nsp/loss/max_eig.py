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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.target = eigvalsh_(self.X)[-1]

    def forward(self, A):

        # print(torch.sum(torch.abs(torch.diag(A)-torch.diag(stoquastic(A)))))
        # print(torch.diag(A))
        # print(torch.diag(stoquastic(A, p_def = self.p_def)))
        # print("\n\n")
        A = stoquastic(A, p_def = self.p_def)
        return eigvalsh_(A)[-1]
    
class BES(BaseMatirxLoss):
    """
    Expectation value of eigenvalues weighted by boltzmann factor

    params
    ------
    act : if U is list of unitary matrix, the map will be like U[0] \otimes U[1] \otimes \cdots U[len(act)-1] @ X @ (c.c.)  . Each U[0].shape[0] == act[0]
    """
    def __init__(self, *args, beta = 1,  **kwargs):
        super().__init__(*args, **kwargs)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.beta = beta
        E = eigvalsh_(self.X)
        self.target = torch.sum(E * torch.exp(self.beta * E)) / torch.sum(torch.exp(self.beta * E))

    def forward(self, A):

        # print(torch.sum(torch.abs(torch.diag(A)-torch.diag(stoquastic(A)))))
        # print(torch.diag(A))
        # print(torch.diag(stoquastic(A, p_def = self.p_def)))
        # print("\n\n")
        A = stoquastic(A, p_def = self.p_def)
        E = eigvalsh_(A)
        return torch.sum(E * torch.exp(self.beta * E)) / torch.sum(torch.exp(self.beta * E))

class MES_SL(BaseMatirxLoss):
    """
    maximum eigenvalue of stoquastic map of matrix (U @ X @ U.T) 

    params
    ------
    act : if U is list of unitary matrix, the map will be like U[0] \otimes U[1] \otimes \cdots U[len(act)-1] @ X @ (c.c.)  . Each U[0].shape[0] == act[0]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.target = eigvalsh_(self.X)[-1]

    def forward(self, A):

        A = stoquastic(A, p_def = self.p_def)
        
        return torch.max(torch.linalg.eigvals(A).real)



class ME(BaseMatirxLoss):
    """
    maximum eigenvalue of  matrix (U @ X @ U.T) 

    params
    ------
    act : if U is list of unitary matrix, the map will be like U[0] \otimes U[1] \otimes \cdots U[len(act)-1] @ X @ (c.c.)  . Each U[0].shape[0] == act[0]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.target = eigvalsh_(self.X)[-1]

    def forward(self, A):
        return eigvalsh_(A)[-1]
    



