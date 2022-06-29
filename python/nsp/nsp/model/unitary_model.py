import numpy as np
from pyrsistent import v
from scipy.linalg import expm
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import copy
import abc
from typing import Union

from ..utils.func import *
from .. import n_sphere



class base_matrix_generator(abc.ABC, torch.nn.Module):
    """Abstract class for matrix generator class. This class prototypes the methods
    needed by a class satisfying the Operator concept.

    params
    ------
    D : dimension of matrix
    """

    def __init__(self, D, dtype = np.float64, seed = 2022):
        super(base_matrix_generator, self).__init__()
        self.D = D
        self.dtype = dtype
        self._type, self._complex = dtype_check(dtype)
        self.npdtype = torch2numpy(self.dtype)
        self._n_params = self._get_n_params()


        
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self._type is torch.Tensor:
            self._params = torch.nn.Parameter(torch.rand(self._n_params))
        else:
            self._params = np.random.rand(self._n_params)

        self.generators = self.make_generators()


    def _type_check(self, X):
        if self._type != type_check(X):
            raise ValueError("Type of U and X are required to be same")

    @abc.abstractmethod
    def _get_n_params(self) -> int:
        """
        return number of parameters
        """

    @abc.abstractclassmethod
    def _get_matrix(self):
        """
        function return matrix from parameters.
        """

    @abc.abstractmethod
    def make_generators(self):
        """
        return generators
        """

    @property
    def params(self):
        return self._params
    

    @property
    def n_params(self):
        return self._n_params
    
    def set_params(self, params : Union[list, np.ndarray, torch.Tensor]):
        if (len(self._params) != len(params)):
            raise ValueError("given params is not appropriate")
        if self._type == torch.Tensor:
            self._params.data[:] = torch.Tensor(params)[:]
        else:
            self._params[:] = np.array(params)[:]

    def matrix(self, params = None):
        if (params is not None) and self._type == np.ndarray:
            self.set_params(params)
        elif (params is not None):
            NotImplementedError("matrix method doesn't reveive paramters as arguments for tensor type")
        return self._get_matrix()

    def reset_params(self, seed = 2022):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self._type is torch.Tensor:
            self._params = torch.nn.Parameter(torch.rand(self._n_params))
        else:
            self._params = np.random.rand(self._n_params)





class unitary_generator(base_matrix_generator):

    """
    class for generating unitary / orthogonal matrix from given params
    """

    def __init__(self, D, dtype = np.float64, seed = 2022, spherical = False):
        super().__init__(D, dtype, seed)
        self.spherical = spherical

    def _get_n_params(self) -> int:
        D = self.D
        if self._complex:
            n_params = D*D-1
        else:
            n_params = D*(D-1)/2
        return int(n_params)

    def make_generators(self):
        tmp_list = []
        D = self.D
        for i, j in [[i,j] for i in range(D) for j in range(i+1,D)]:
            tmp = np.zeros((D,D), dtype=self.npdtype)
            tmp[i,j] = 1
            tmp[j,i] = -1
            tmp_list.append(tmp)
            if self._complex:
                tmp = np.zeros((D,D), dtype=self.npdtype)
                tmp[i,j] = 1j
                tmp[j,i] = 1j
                tmp_list.append(tmp)
        if self._complex:
            for i in range(D-1):
                tmp = np.zeros((D,D), dtype=self.npdtype)
                tmp[i,i] = 1j
                tmp_list.append(tmp)
        tmp_list = np.array(tmp_list)
        if self._type == torch.Tensor:
            return torch.tensor(tmp_list)
        else:
            return tmp_list


    def _convert_params(self):
        if self.spherical and self.n_params != 1:
            return n_sphere.convert_rectangular(self._params)
        else:
            return self._params

    def _get_matrix(self):
        params = self._convert_params()
        tmp =  (params[:,None,None] * self.generators).sum(axis=0)
        return matrix_exp_(tmp)