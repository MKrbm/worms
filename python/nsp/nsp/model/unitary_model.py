import numpy as np
from pyrsistent import v
from scipy.linalg import expm
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



class BaseMatrixGenerator(abc.ABC, torch.nn.Module):
    """Abstract class for matrix generator class. This class prototypes the methods
    needed by a class satisfying the Operator concept.

    params
    ------
    D : dimension of matrix
    """

    def __init__(self, D, dtype = np.float64, seed = None):
        super(BaseMatrixGenerator, self).__init__()
        self.D = D
        self.dtype = dtype
        self._type, self._complex = dtype_check(dtype)
        self.npdtype = torch2numpy(self.dtype)
        self._n_params = self._get_n_params()


        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self._type is torch.Tensor:
            self._params = torch.nn.Parameter(torch.rand(self._n_params))
        else:
            self._params = np.random.rand(self._n_params)

        self.generators = self._make_generators()


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
    def _make_generators(self):
        """
        return generators
        """

    @staticmethod
    @abc.abstractclassmethod
    def _inv(U):
        """
        For unitary matrix, complex conjugate
        """
    @property
    def params(self):
        return self._params
    

    @property
    def n_params(self):
        return self._n_params
    
    def set_params(self, params : Union[list, np.ndarray, torch.Tensor], copy_grad = False):
        # if isinstance(params, list):
        #     params = np.array(params)
        # p_r = params.real
        # if self._complex:
        #     p_i = params.imag
        #     if type_check(params) == torch.Tensor:
        #         params = torch.concat([p_r, p_i])
        #     elif type_check(params) == np.np.ndarray:
        #         params = np.concatenate([p_r, p_i])
        # else:
        #     params = p_r

        if (len(self._params) != len(params)):
            raise ValueError("given params is not appropriate")
        if self._type == torch.Tensor:
            self._params.data = convert_type(params, torch.Tensor).data
            if copy_grad and hasattr(params, "grad"):
                self._params.grad = params.grad
        else:
            self._params[:] = np.array(params)[:]

    def matrix(self, params = None):
        if (params is not None) :
            self.set_params(params)
        return self._get_matrix()

    def reset_params(self, seed = None):
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if self._type is torch.Tensor:
            self._params = torch.nn.Parameter(torch.rand(self._n_params))
        else:
            self._params = np.random.rand(self._n_params)





class UnitaryGenerator(BaseMatrixGenerator):

    """
    class for generating unitary / orthogonal matrix from given params
    """

    def __init__(self, D, dtype = np.float64, seed = None, spherical = False):
        super().__init__(D, dtype, seed)
        self.spherical = spherical

    def _get_n_params(self) -> int:
        D = self.D
        if self._complex:
            n_params = D*D-1
        else:
            n_params = D*(D-1)/2
        return int(n_params)

    def _make_generators(self):
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

    @staticmethod
    def _inv(U):
        return U.T.conj()

class UnitaryRiemanGenerator(BaseMatrixGenerator):

    """
    class for generating square matrix
    initial matrix is unitary matrix or orthogonal 

    W <- W - \Delta S
    """

    def __init__(self, D, dtype = np.float64, seed = None):
        super().__init__(D, dtype, seed)
        self.reset_params(seed=seed)


    def set_params(self, params : Union[list, np.ndarray, torch.Tensor], copy_grad = False):
        if isinstance(params, list):
            params = np.array(params)
        p_r = params.real
        if self._complex:
            p_i = params.imag
            if type_check(params) == torch.Tensor:
                params = torch.concat([p_r, p_i])
            elif type_check(params) == np.ndarray:
                params = np.concatenate([p_r, p_i])
        else:
            params = p_r

        if (len(self._params) != len(params)):
            raise ValueError("given params is not appropriate")
        if self._type == torch.Tensor:
            self._params.data = convert_type(params, torch.Tensor).data
            if copy_grad and hasattr(params, "grad"):
                self._params.grad = params.grad
        else:
            self._params[:] = np.array(params)[:]


    def _get_n_params(self) -> int:
        D = self.D
        if self._complex:
            n_params = D*D*2
        else:
            n_params = D*D
        return int(n_params)

    def _make_generators(self):
        # raise NotImplementedError("generators are not used in this class")
        pass

    def _get_matrix(self):
        if self._complex:
            return view_tensor(self._params[:self.D**2], [self.D]*2) + 1j*view_tensor(self._params[self.D**2:], [self.D]*2)
        else:
            return view_tensor(self._params[:self.D**2], [self.D]*2)

    def reset_params(self, seed = None):
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if self._complex:
            randnMatrix = np.random.randn(self.D, self.D) + 1j*np.random.randn(self.D, self.D)
        else:
            randnMatrix = np.random.randn(self.D, self.D)
        Q, R = np.linalg.qr(randnMatrix)
        haar_orth = Q.dot(np.diag(np.diag(R)/np.abs(np.diag(R))))   
        if self._complex:
            params = haar_orth.reshape(-1)
            pr = params.real.astype(np.float64)
            pi = params.imag.astype(np.float64)
            self.set_params(np.concatenate([pr, pi]))
        else:
            self.set_params(haar_orth.reshape(-1).astype(np.float64))

    @staticmethod
    def _inv(U):
        return U.T.conj()
