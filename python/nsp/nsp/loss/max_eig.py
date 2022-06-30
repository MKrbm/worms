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

class base_ulf(abc.ABC):
    """Abstract class for unitary(matrix) loss function. This class prototypes the methods
    needed by a class satisfying the Operator concept.
    """
    Xtarget : float
    def __init__(self, X, act : Union[list, np.ndarray]):

        self._type = type_check(X) # return np.ndarray or torch.Tensor
        self.dtype = X.dtype
        self.X = X
        self.act = np.array(act)
        self._n_unitaries = len(self.act)
        if (X.shape[0] != np.prod(self.act)):
            raise ValueError("act on list is inconsistent with X")
        

    def __call__(self, U_list : list):
        """
        if _U is not given by list, repete _U by len(act) times.
        """
        if not isinstance(U_list, list):
            U_list = [U_list] * self._n_unitaries
        self._unitary_check(U_list)
        return self.forward(U_list)

    @abc.abstractmethod
    def forward(self, U_list : list):
        """
        return loss value from list of unitaries.
        """


    def kron(self, U : list):

        if self._type == np.ndarray:
            self.U = np.eye(1)
            for u in U:
                self.U = np.kron(self.U, u)
        else:
            self.U = torch.eye(1)
            for u in U:
                self.U = torch.kron(self.U, u)
        return self.U


    def _unitary_check(self, U):
        # if len(U) != len(self.act):
        #     raise ValueError("U and act are inconsistent")
        for u, act in zip(U, self.act):
            if self._type != type_check(u):
                raise ValueError("Type of U and X are required to be same")
            if u.shape[0] != act:
                raise ValueError("dimension of local unitary matrix should be same as act[i]")

    def _transform(self, U):
        return self._inverse(U) @ cast_dtype(self.X, U.dtype) @ U
    
    @staticmethod
    @abc.abstractmethod
    def _inverse(U):
        """
        inverse of given matrix U
        """




class mes(base_ulf):
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

    def forward(self, U_list):
        U = self.kron(U_list)
        A = stoquastic(self._transform(U))
        return eigvalsh_(A)[-1]

    @staticmethod
    def _inverse(U):
        return U.T.conj()

class me(base_ulf):
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

    def forward(self, U_list):
        U = self.kron(U_list)
        A = self._transform(U)
        return eigvalsh_(A)[-1]
    
    @staticmethod
    def _inverse(U):
        return U.T.conj()


