from typing import Type
import numpy as np
import torch
from scipy.optimize import OptimizeResult
import random

def exp_energy(E, beta):
    Z = np.exp(-beta*E)
    EZ = E*Z
    return np.sum(EZ)/np.sum(Z)

def _absolute_map(X, sign):
    for i in range(X.shape[0]):
        for j in range(i,X.shape[1]):
            X[i,j] = np.abs(X[i,j])*sign
            X[j,i] = np.abs(X[i,j])*sign
    return X


def _absolute_map_torch(X, sign):
    for i in range(X.shape[0]):
        for j in range(i,X.shape[1]):
            X[i,j] = torch.abs(X[i,j])*sign
            X[j,i] = torch.abs(X[i,j])*sign
    return X


def _abs(X, sign = 1):
    if not(sign in [1, -1]):
        ValueError("sign is either 1 or -1")
    
    if isinstance(X, np.ndarray):
        return sign * np.abs(X)
    elif isinstance(X, torch.Tensor):
        return sign * torch.abs(X)
    else:
        raise TypeError("type of X is inappropriate")

"""
min function for torch.complex dtype tensor.
choose the elemtent with the biggest real value
"""
def _torch_complex_min(A):
    idx = torch.argmin(A.real).item()
    return A[idx]


def _positive_map_torch(A, p_def = False):
    if p_def:
        # print("iside func",torch.diag(A))
        # print("insde func abs ", torch.diag(torch.abs(A)))
        return torch.abs(A)
    else:
        a = (_torch_complex_min(torch.diag(A))-1)* torch.eye(A.shape[0])
        return torch.abs(A-a) + a

def _positive_map(A, p_def = False):
    if p_def:
        return np.abs(A)
    else:
        a = np.eye(A.shape[0])*np.min(np.diag(A))
        return np.abs(A-a) + a 


def stoquastic(X, abs_ : bool = False, p_def = False):
    """
    take absolute except diagonal part.

    arags:
        p_def : matrix is positive definite or not. If true, it is garanteed that all diagonal elements are positive.
    """
    if abs_:
        return abs(X)
    if isinstance(X, np.ndarray):
        return _positive_map(X, p_def)
    elif isinstance(X, torch.Tensor):
        return _positive_map_torch(X, p_def)
    else:
        raise TypeError("type of X is inappropriate")


def type_check(X):
    if not (X.ndim == 2 and X.shape[0]==X.shape[1]):
        raise ValueError("X must be a 2D square matrix")
        
    if isinstance(X, np.ndarray):
        if (X.ndim != 2):
            raise ValueError("dimension should be 2") 
        return type(X)
    elif isinstance(X, torch.Tensor):
        if (X.ndim != 2):
            raise ValueError("dimension should be 2")
        return type(X)
    else:
        raise TypeError("type of X is inappropriate")
    



def dtype_check(dtype):
    if dtype == np.float64:
        return np.ndarray, False
    elif dtype == np.complex128:
        return np.ndarray, True
    elif dtype == torch.float64:
        return torch.Tensor, False
    elif dtype == torch.complex128: 
        return torch.Tensor, True
    else:
        TypeError("dtype is not valid")

numpy_to_torch_dtype_dict = {
    np.bool       : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

torch_to_numpy_dtype_dict = {
    torch.bool       : np.bool,
    torch.uint8      : np.uint8,
    torch.int8       : np.int8,
    torch.int16      : np.int16,
    torch.int32      : np.int32,
    torch.int64      : np.int64,
    torch.float16    : np.float16,
    torch.float32    : np.float32,
    torch.float64    : np.float64,
    torch.complex64  : np.complex64,
    torch.complex128 : np.complex128
}
def numpy2torch(dtype):
    if dtype in numpy_to_torch_dtype_dict.keys():
        return numpy_to_torch_dtype_dict[dtype]
    else:
        return dtype

def torch2numpy(dtype):
    if dtype in torch_to_numpy_dtype_dict.keys():
        return torch_to_numpy_dtype_dict[dtype]
    else:
        return dtype


def convert2numpy(X):
    if isinstance(X, torch.Tensor):
        return np.array(X.data)
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise TypeError("X should be either np_array or tensor")

def convert2tensor(X):
    if isinstance(X, torch.Tensor):
        return X
    elif isinstance(X, np.ndarray):
        return torch.from_numpy(X)
    else:
        raise TypeError("X should be either np_array or tensor")

def convert_type(X, _type):
    if not (isinstance(X, np.ndarray) or isinstance(X, torch.Tensor)):
        return X
    if _type == torch.Tensor:
        return convert2tensor(X)
    elif _type == np.ndarray:
        return convert2numpy(X)
    else:
        raise TypeError("type should be either np_array or tensor : {}".format(_type))

def eigvalsh_(A):
    _type = type(A)
    if _type == np.ndarray:
        return np.linalg.eigvalsh(A)
    elif _type == torch.Tensor:
        return torch.linalg.eigvalsh(A)
    else:
        raise TypeError("A should be either np_array or tensor")



def zeros_(D, _type):
    if _type == np.ndarray:
        return np.zeros(D)
    elif _type == torch.Tensor:
        return  torch.zeros(D)
    else:
        raise TypeError("A should be either np_array or tensor")

from scipy.linalg import expm

def matrix_exp_(A):
    _type = type(A)
    if _type == np.ndarray:
        return expm(A)
    elif _type == torch.Tensor:
        return torch.matrix_exp(A)
    else:
        TypeError("A should be either np_array or tensor")

def cast_dtype(X, dtype):
    if isinstance(X, torch.Tensor):
        return X.to(dtype)
    elif isinstance(X, np.ndarray):
        return X.astype(dtype)
    else:
        raise TypeError("X should be either np_array or tensor")

def view_tensor(X, view_ : list):
    if (isinstance(view_, np.ndarray)):
        view_ = view_.tolist()
    if (not isinstance(view_, list)):
        raise TypeError("view_ must be a list or nparray")
    if isinstance(X, torch.Tensor):
        return X.view(view_)
    elif isinstance(X, np.ndarray):
        return X.reshape(view_)
    else:
        raise TypeError("X should be either np_array or tensor")

def einsum_(string, *arg):
    if isinstance(arg[0], torch.Tensor):
        return torch.einsum(string, *arg).contiguous()  
    elif isinstance(arg[0], np.ndarray):
        return np.einsum(string, *arg)
    else:
        raise TypeError("X should be either np_array or tensor")

def is_hermitian(X):
    if isinstance(X, torch.Tensor):
        return torch.all(X==X.T.conj())
    elif isinstance(X, np.ndarray):
        return np.all(X==X.T.conj())
    else:
        raise TypeError("X should be either np_array or tensor")    


def pick_negative(X):
    if isinstance(X, torch.Tensor):
        return torch.minimum(torch.tensor(0), X)
    elif isinstance(X, np.ndarray):
        return np.minimum(0, X)
    else:
        raise TypeError("X should be either np_array or tensor")  

def set_mineig_zero(X):
    if isinstance(X, torch.Tensor):
        E = torch.linalg.eigvalsh(X)[0]
        return X - torch.eye(X.shape[0]) * E
    elif isinstance(X, np.ndarray):
        E = np.linalg.eigvalsh(X)[0]
        return X - np.eye(X.shape[0]) * E
    else:
        raise TypeError("X should be either np_array or tensor") 


def l2_measure(X):
    X = X - (_abs(X))
    X = (X*X.conj()).real

    return (X).sum() - X.trace()

def l1_measure(X):
    X = _abs(X - (_abs(X)))

    return (X.sum() - X.trace())


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def is_identity_torch(X, is_complex=False, decimals = 10):
    X_r = torch.round(X.real, decimals=decimals)
    if is_complex:
        X_i = torch.round(X.imag, decimals=decimals)
        return (X_r == torch.eye(X_r.shape[0])).all() and (X_i == 0).all()
    else:
        return (X_r == torch.eye(X_r.shape[0])).all()


def cc(X):
    return X.T.conj()

def matinv(X):
    if isinstance(X, torch.Tensor):
        return torch.linalg.inv(X)
    elif isinstance(X, np.ndarray):
        return np.linalg.inv(X)
    else:
        raise TypeError("X should be either np_array or tensor") 



def is_square_matrix(X):
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")


def mul(H, h, al, from_left=True):
    
    is_square_matrix(H)
    is_square_matrix(h)
    L = H.shape[0]    
    if np.prod(al) != L:
        print("act list is inconsistent with given Hamiltonian")
    l = h.shape[0]
    if l != al[1] or len(al) !=3:
        print("al is not a proper list")
    ori_shape = (L,)
    conv_shape = tuple(al)
    H = H.reshape(2*conv_shape)
    if from_left:
        H = np.einsum("ijklmn, jg->igklmn", H, h).reshape(2*ori_shape)
    else:
        H = np.einsum("ijklmn, mg->ijklgn", H, h).reshape(2*ori_shape)
        
    return H

# def swap_axis(h, L, sps, axis):
#     N = sps ** L
#     assert N == h.shape[0]

#     ori_shape = (N, N)
#     trans = np.arange(2*L)
#     # if (axis[0][0] == axis[0][1] and axis[1][1] == axis[1][0]):
#     #     return h
#     for ax in axis:
#         (a1, a2) = ax
#         (b1, b2) = (a1, np.argwhere(trans==a2)[0,0])
#         (a1, a2) = (np.argwhere(trans==a1)[0,0], a2)
#         print(a2, b1, a1, b2)
#         trans[a2] = b1
#         trans[a1] = b2
#         trans[a2+L] = b1+L
#         trans[a1+L] = b2+L
#         print(trans)
#     return h.reshape(L*2*(sps,)).transpose(trans).reshape(ori_shape)


