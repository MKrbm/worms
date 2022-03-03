import numpy as np
from scipy.linalg import expm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

def loss_1(A):
#     A = -A**2
    A = torch.nn.ReLU()(-A)
    return - torch.trace(A) + A.sum()
    
    
class unitary_solver(torch.nn.Module):
    def __init__(self,sps_list):
        super(unitary_solver, self).__init__()
        self.sps_list = sps_list
        self.generators = []
        self._n_params = []
        for N in sps_list:
            self.generators.append(torch.tensor(self.make_generator(N), requires_grad=False))
            self._n_params.append(int(N*(N-1)/2))
        tmp = torch.rand(np.sum(self._n_params))
        self._params = torch.nn.Parameter(tmp)
        self._size = np.prod(sps_list)
        
    def forward(self, x):
        assert x.shape[0] == x.shape[1], "require square matrix"
        assert x.shape[0] == self._size
        x = self.matrix @ x @ self.matrix.T
        return x
    
    @property
    def params(self):
        return self._params
    
    @property
    def n_params(self):
        return self._n_params
    
    def set_params(self,params):
        self._params[:] = params[:]
    
    @property
    def matrix(self):
        ind = 0
        unitary_list = []
        for i, n in enumerate(self._n_params):
            tmp = self._params[ind:ind+n,None,None] * self.generators[i]
            ind += n
            unitary_list.append(torch.matrix_exp(tmp.sum(axis=0)))
        
        U_return = torch.eye(1)
        for U in unitary_list:
            U_return = torch.kron(U_return, U)
        return U_return
    
    
    @staticmethod
    def make_generator(N):
        tmp_list = []
        for i, j in [[i,j] for i in range(N) for j in range(i+1,N)]:
            tmp = np.zeros((N,N), dtype=np.float64)
            tmp[i,j] = 1
            tmp[j,i] = -1
            tmp_list.append(tmp)
        return np.array(tmp_list)

