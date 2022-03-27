import numpy as np
from pyrsistent import v
from scipy.linalg import expm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
# import torch.kron

def loss_1(A):
#     A = -A**2
    A = torch.nn.ReLU()(-A)
    return - torch.trace(A) + A.sum()


def loss_eig(A, beta):
#     A = -A**2
    B = torch.zeros(1)
    if A.ndim!=3:
        A = A[None,:]
    else:
        for a in A:
            a_ = torch.abs(a)
            eigs = torch.linalg.eigvalsh(a_)
            # Z = eigs*torch.exp(beta*(eigs.data))
            # B += torch.sum(eigs*Z) / torch.sum(Z)
            # print(B)
            B += eigs[-1]
    return B
    
    
class unitary_solver(torch.nn.Module):
    def __init__(self,sps_list, syms = False, seed = 2022, dtype = torch.float64):
        super(unitary_solver, self).__init__()

        self.sps_list = sps_list
        self.generators = []
        self._n_params = []
        self._matrix = None
        self._syms = syms
        self._loss = None
        self.dtype = dtype


        if dtype == torch.float64:
            self.denominator = 2
            self.complex = False
            self.npdtype = np.float64
        elif dtype == torch.complex128:
            self.complex = True
            self.denominator = 1
            self.npdtype = np.complex128
        if syms:
            N = sps_list[0]
            assert np.all(np.array(sps_list) == N), "parameter symmetrization can only be applied to the sps_list with all elements being the same"
            self._n_params = [int(N*(N-1)/self.denominator)]
            self.generators.append(torch.tensor(self.make_generator(N), requires_grad=False))
        else:
            for N in sps_list:
                self.generators.append(torch.tensor(self.make_generator(N), requires_grad=False))
                self._n_params.append(int(N*(N-1)/self.denominator))
        
        torch.manual_seed(seed)
        tmp = torch.rand(np.sum(self._n_params))
        # tmp = torch.zeros(np.sum(self._n_params))
        self._params = torch.nn.Parameter(tmp)
        self._size = np.prod(sps_list)
    
    def forward(self, x):
        if x.ndim != 3:
            x = x[None, :, :]

        x = x.type(self.dtype)
        assert x.shape[1] == x.shape[2], "require square matrix"
        assert x.shape[1] == self._size
        M = self.matrix[None, :, :]
        x = M @ x @ M.transpose(1,2).conj()

        # add = torch.zeros(1)
        return x

    def set_loss(self, loss):
        self._loss = loss


    @property
    def get_loss(self):
        return self._loss


    @property
    def params(self):
        return self._params
    
    @property
    def n_params(self):
        return self._n_params
    
    def set_params(self, params):
        self._params[:] = params[:]
    
    @property
    def matrix(self):
        if self._syms:
            return self._get_sym_matrix()
        else:
            return self._get_matrix()
    
    @property
    def one_site_matrix(self):
        if not self._syms:
            raise NameError("this method is unavailable for syms = False")
        else:
            return self._one_site_matrix


    def _get_sym_matrix(self):
        ind = 0
        U_return = torch.eye(1)
        N = self.n_params[0]
        tmp = self._params[:,None,None] * self.generators[0]
        tmp = tmp.sum(axis=0)
        self._one_site_matrix = torch.matrix_exp(tmp)
        for _ in range(len(self.sps_list)):
            U_return = torch.kron(U_return, self._one_site_matrix)
        return U_return     
    

    def _get_matrix(self):
        ind = 0
        U_return = torch.eye(1)
        for i, n in enumerate(self._n_params):
            tmp = self._params[ind:ind+n,None,None] * self.generators[i]
            ind += n
            U_return = torch.kron(U_return, torch.matrix_exp(tmp.sum(axis=0)))
        return U_return

    def make_generator(self, N):
        tmp_list = []
        for i, j in [[i,j] for i in range(N) for j in range(i+1,N)]:
            tmp = np.zeros((N,N), dtype=self.npdtype)
            tmp[i,j] = 1
            tmp[j,i] = -1
            tmp_list.append(tmp)
            if self.complex:
                tmp[i,j] = 1j
                tmp[j,i] = 1j
                tmp_list.append(tmp)
        return np.array(tmp_list)




class matrix_solver(torch.nn.Module):
    def __init__(self,sps_list, syms = False):
        super(matrix_solver, self).__init__()

        self.sps_list = sps_list
        self.generators = []
        self._n_params1 = []
        self._n_params2 = []
        self._n_params3 = []
        self._matrix = None
        self._syms = syms

        if syms:
            N = sps_list[0]
            assert np.all(np.array(sps_list) == N), "parameter symmetrization can only be applied to the sps_list with all elements being the same"
            self._n_params1 = [int(N*(N-1)/2)]
            self._n_params2 = [int(N*(N-1)/2)]
            self._n_params3 = [N]
            self.generators.append(torch.tensor(self.make_generator(N), requires_grad=False))

        else:
            for N in sps_list:
                self.generators.append(torch.tensor(self.make_generator(N), requires_grad=False))
                self._n_params1.append(int(N*(N-1)/2))
                self._n_params2.append(int(N*(N-1)/2))
                self._n_params3.append(N)


        

        tmp = torch.rand(np.sum(self._n_params1),dtype=torch.float64)
        self._params1 = torch.nn.Parameter(tmp)
        tmp = torch.rand(np.sum(self._n_params2),dtype=torch.float64)
        self._params2 = torch.nn.Parameter(tmp)

        tmp = torch.rand(np.sum(self._n_params3),dtype=torch.float64)
        self._params3 = torch.nn.Parameter(tmp)
    
        self._size = np.prod(sps_list)


    def forward(self, x):
        assert x.shape[0] == x.shape[1], "require square matrix"
        assert x.shape[0] == self._size
        M = self.matrix
        x = M @ x @ torch.inverse(M)
        return x
    
    @property
    def params(self):
        return self._params
    
    @property
    def n_params(self):
        return self._n_params
    
    def set_params(self, params):
        self._params[:] = params[:]
    
    @property
    def matrix(self):
        if self._syms:
            return self._get_sym_matrix()
        else:
            return self._get_matrix()
    
    @property
    def one_site_matrix(self):
        if not self._syms:
            raise NameError("this method is unavailable for syms = False")
        else:
            return self._one_site_matrix


    def _get_sym_matrix(self):
        U_return1 = torch.eye(1)
        tmp = self._params1[:,None,None] * self.generators[0]
        tmp = tmp.sum(axis=0)
        self._one_site_matrix1 = torch.matrix_exp(tmp)
        for _ in range(len(self.sps_list)):
            U_return1 = torch.kron(U_return1, self._one_site_matrix1)

        U_return2 = torch.eye(1)
        tmp = self._params2[:,None,None] * self.generators[0]
        tmp = tmp.sum(axis=0)
        self._one_site_matrix2 = torch.matrix_exp(tmp)
        for _ in range(len(self.sps_list)):
            U_return2 = torch.kron(U_return2, self._one_site_matrix2)

        U_return3 = torch.eye(1)
        tmp = torch.diag(self._params3[:])
        # tmp = tmp.sum(axis=0)x
        self._one_site_matrix3 = tmp
        for _ in range(len(self.sps_list)):
            U_return3 = torch.kron(U_return3, self._one_site_matrix3)
        return U_return1 @ U_return3 @ U_return2     
    

    def _get_matrix(self):
        ind = 0
        U_return1 = torch.eye(1)
        for i, n in enumerate(self._n_params1):
            tmp = self._params1[ind:ind+n,None,None] * self.generators[i]
            ind += n
            U_return1 = torch.kron(U_return1, torch.matrix_exp(tmp.sum(axis=0)))

        ind = 0
        U_return2 = torch.eye(1)
        for i, n in enumerate(self._n_params2):
            tmp = self._params2[ind:ind+n,None,None] * self.generators[i]
            ind += n
            U_return2 = torch.kron(U_return2, torch.matrix_exp(tmp.sum(axis=0)))
        
        ind = 0
        U_return3 = torch.eye(1)
        for i, n in enumerate(self._n_params3):
            tmp = torch.diag(self._params3[ind:ind+n])
            ind += n
            U_return3 = torch.kron(U_return3, tmp)
        return U_return1 @ U_return3 @ U_return2
    @staticmethod
    def make_generator(N):
        tmp_list = []
        for i, j in [[i,j] for i in range(N) for j in range(i+1,N)]:
            tmp = np.zeros((N,N), dtype=np.float64)
            tmp[i,j] = 1
            tmp[j,i] = -1
            tmp_list.append(tmp)
        return np.array(tmp_list)



class diagonal_solver(torch.nn.Module):
    def __init__(self,sps_list, syms = False):
        super(diagonal_solver, self).__init__()

        self.sps_list = sps_list
        self._n_params3 = []
        self._matrix = None
        self._syms = syms

        if syms:
            N = sps_list[0]
            assert np.all(np.array(sps_list) == N), "parameter symmetrization can only be applied to the sps_list with all elements being the same"
            self._n_params3 = [N]

        else:
            for N in sps_list:
                self._n_params3.append(N)


        

        tmp = torch.ones(np.sum(self._n_params3),dtype=torch.float64)
        self._params3 = torch.nn.Parameter(tmp)
    
        self._size = np.prod(sps_list)


    def forward(self, x):
        assert x.shape[0] == x.shape[1], "require square matrix"
        assert x.shape[0] == self._size
        M, M_inv = self.matrix()
        x = M @ x @ M_inv
        return x
    
    @property
    def params(self):
        return self._params
    
    @property
    def n_params(self):
        return self._n_params
    
    @property
    def set_params(self, params):
        self._params[:] = params[:]
    
    def matrix(self):
        if self._syms:
            return self._get_sym_matrix()
        else:
            return self._get_matrix()
    
    @property
    def one_site_matrix(self):
        if not self._syms:
            raise NameError("this method is unavailable for syms = False")
        else:
            return self._one_site_matrix



    def _get_sym_matrix(self):

        U_return = torch.eye(1)
        U_inv = torch.eye(1)
        tmp = torch.diag(self._params3)
        tmpinv = torch.diag(1/self._params3)
        # tmp = tmp.sum(axis=0)x
        self._one_site_matrix3 = tmp
        self._one_site_matrix_inv3 = tmpinv
        for _ in range(len(self.sps_list)):
            U_return = torch.kron(U_return, self._one_site_matrix3)
            U_inv = torch.kron(U_inv, self._one_site_matrix_inv3)
        return U_return, U_inv
    

    def _get_matrix(self):
        ind = 0
        U_return = torch.eye(1)
        U_inv = torch.eye(1)
        for i, n in enumerate(self._n_params3):
            tmp = torch.diag(self._params3[ind:ind+n])
            tmp_inv = torch.diag(1/self._params3[ind:ind+n])
            ind += n
            U_return = torch.kron(U_return, tmp)
            U_inv = torch.kron(U_inv,tmp_inv)
        self.U_return = U_return
        self.U_inv = U_inv



    @staticmethod
    def make_generator(N):
        tmp_list = []
        for i, j in [[i,j] for i in range(N) for j in range(i+1,N)]:
            tmp = np.zeros((N,N), dtype=np.float64)
            tmp[i,j] = 1
            tmp[j,i] = -1
            tmp_list.append(tmp)
        return np.array(tmp_list)


def get_mat_status(X):
    X = np.array(X)
    L = X.shape[1]
    lps = int(np.sqrt(L))
    if lps**2 != L:
        print("! ---- This matrix might not be the bond operator ---- !")


    assert X.ndim==3, "dimension of X must be 3"

    E = []    
    for x in X:
        if not np.all(np.diag(x)>=0):
            print("! ---- diagonal elements of the local hamiltonian should be non negative ---- !")
        e, V = np.linalg.eigh(x)
        E.append(e)
    return L, lps, np.array(E)

def energy(E, beta):
    Z = np.exp(-beta*E)
    return (E*Z).sum()/Z.sum()

def optim_matrix_symm(X, N_iter,
        optm_method = torch.optim.Adam, 
        loss_func = loss_eig,
        print_status = True,
        seed = 0,
        beta = 10,
        **kwargs,
        ):
    X = torch.stack(X)
    if X.ndim!=3:
        X = X[None,:]
    L, lps, E = get_mat_status(X)
    model = unitary_solver([lps,lps],syms=True, seed = seed)
    optimizer = optm_method(model.parameters(), **kwargs)
    
    np.set_printoptions(precision=3)

    grad_list = []

    E_exp = 0
    for e in E:
        E_exp += e[-1]
    print("target loss : {:.3f}\n".format(E_exp))
    
    print("-"*10, "iteration start", "-"*10)
    
    mu = 10

    beta_list = np.linspace(10, 1000, N_iter)
    for t in range(N_iter):
        y= model(X)
        loss = loss_func(y,beta_list[t])
        loss_ = loss
        if t % 1000 == 0:
            print("iteration : {:4d}   loss : {:.3f}".format(t,loss_.item()))
        optimizer.zero_grad()
        loss_.backward()
        model.set_loss(loss.data)
        optimizer.step()
        grad_list.append(model._params.grad)
        # model._params = torch.nn.Parameter(model._params.data - 0.001 * model._params.grad)


    if print_status:

        E2 = []
        E3 = []
        for x, yy in zip(X.data,y.data):
            e2, V2 = np.linalg.eigh(np.abs(np.array(x.data)))
            e3, V3 = np.linalg.eigh(np.abs(np.array(yy.data)))
            E2.append(e2)
            E3.append(e3)

        E2 = np.array(E2)
        E3 = np.array(E3)

        print("\n","-"*14, "results", "-"*14)
        print("target loss      : {:.3f}".format(np.sum(E[:,-1])))
        print("loss before optm : {:.3f}".format(np.sum(E2[:,-1])))
        print("loss after optm  : {:.3f}".format(np.sum(E3[:,-1])))
    return model, grad_list