
from copy import copy, deepcopy
from os import stat
from unittest import result
from matplotlib import container
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
from ..model.unitary_model import BaseMatrixGenerator

class BaseMatirxLoss(abc.ABC):
    """Abstract class for unitary(matrix) loss function. This class prototypes the methods
    needed by a class satisfying the Operator concept.
    """
    target : float
    model : BaseMatrixGenerator
    def __init__(self, 
        X : torch.Tensor,
        act : Union[list, np.ndarray], 
        inv = cc,
        mineig_zero = True,
        einsum=False):
        
        self._inverse = inv
        self.X = convert_type(X, torch.Tensor)
        # self.X_original = torch.from_numpy(np.copy(X))
        self.X_original = self.X.detach().clone()
        self.p_def = False
        if mineig_zero:
            self.X = set_mineig_zero(self.X)
            self.p_def = True

        self._type = type_check(self.X) # return np.ndarray or torch.Tensor
        if not is_hermitian(self.X):
            raise ValueError("initial matrix X is required to be hermitian matrix")

        self.act = np.array(act)
        if len(self.act) != 2:
            raise NotImplementedError("local hamiltonian must be a bond operator")
        self.dtype = self.X.dtype
        self.act_cumprod = np.cumprod(self.act)
        # self.act_cumprod.insert(1, 0)
        self.act_cumprod=np.insert(self.act_cumprod, 0, 1)
        
        self._n_unitaries = len(self.act)
        if (self.X.shape[0] != np.prod(self.act)):
            raise ValueError("act on list is inconsistent with X")
        if (einsum):
            self._transform = self._transform_einsum
        else:
            self._transform = self._transform_kron

        # if (X.shape)
        

    def __call__(self, U_list : list):
        """
        if _U is not given by list, repete _U by len(act) times.
        """
        
        if not isinstance(U_list, list):
            U_list = [U_list] * self._n_unitaries
        self._matrix_check(U_list)
        return self.forward(self._transform(U_list))

    @abc.abstractmethod
    def forward(self, A):
        """
        return loss value
        """
        
    def _transform_kron(self, U_list : list, original = False):
        U = self._kron(U_list)
        if self.X.dtype != U.dtype:
            print("dtype changes from {} to {}".format(self.X.dtype, U.dtype))
            self.X = cast_dtype(self.X, U.dtype)
        if original:
            return self._inverse(U) @ self.X_original @ U
        else:
            return self._inverse(U) @ self.X @ U
        

    def _transform_einsum(self, U_list : list, original = False):
        """
        perform return U @ cast_dtype(self.X, U.dtype) @ self._inverse(U)
        """
        if original:
            resultMatrix = deepcopy(self.X_original)
        else:
            resultMatrix = deepcopy(self.X)
        for i in range(len(self.act)):
            resultMatrix=self._multiplymatrix(resultMatrix, U_list, i, True, True)
            resultMatrix=self._multiplymatrix(resultMatrix, U_list, i, False, False)
    
        return resultMatrix

    def apply(self, U_list :list, order = "F"):
        R = self._transform(U_list).detach().numpy()

    def _multiplymatrix(self, Matrix, U_list, i, fromTheLeft=True, inv = False):
        """
        multiply matricies U_list[i] to self.X from left at self.act[i]

        Args:
            cc : complex conjugate
            fromTheLeft : If true, apply matrix from the left side. If false right side.
        """
        tmpTensor = view_tensor(Matrix, [self.act_cumprod[i], self.act[i], int(self.act_cumprod[-1]/self.act_cumprod[i+1])]*2)
        if fromTheLeft:
            contractedTensor = einsum_("ik, akbcjd->aibcjd", self._inverse(U_list[i]) if inv else U_list[i], tmpTensor)             
        else:
            contractedTensor = einsum_("aibcjd, jk->aibckd", tmpTensor, self._inverse(U_list[i]) if inv else U_list[i])
        

        return view_tensor(contractedTensor, [self.X.shape[0], self.X.shape[1]])


    def _kron(self, U_list : list):

        if self._type == np.ndarray:
            self.U = np.eye(1)
            for u in U_list:
                self.U = np.kron(u, self.U)
        else:
            self.U = torch.eye(1)
            for u in U_list:
                self.U = torch.kron(u, self.U)
        return self.U



    def _matrix_check(self, U):
        # if len(U) != len(self.act):
        #     raise ValueError("U and act are inconsistent")
        for u, act in zip(U, self.act):
            if self._type != type_check(u):
                # raise ValueError("Type of U and X are required to be same")
                self._type_convert(type_check(u))
                print("type of loss is converted so taht loss._type == model._type")
            if u.shape[0] != act:
                raise ValueError("dimension of local unitary matrix should be same as act[i]")

    def _type_convert(self, _type):
        self.X = convert_type(self.X, _type)
        self._type = _type


