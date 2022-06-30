from typing import Type
import numpy as np
import torch


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


def abs(X, sign = 1):
    if not(sign in [1, -1]):
        ValueError("sign is either 1 or -1")
    
    if isinstance(X, np.ndarray):
        ValueError("dimension should be 2") if (X.ndim != 2) else None
        return sign * np.abs(X)
    elif isinstance(X, torch.Tensor):
        ValueError("dimension should be 2") if (X.ndim != 2) else None
        return sign * torch.abs(X)
    else:
        TypeError("type of X is inappropriate")

"""
min function for torch.complex dtype tensor.
choose the elemtent with the biggest real value
"""
def _torch_complex_min(A):
    idx = torch.argmin(A.real).item()
    return A[idx]


def _positive_map_torch(A):
    a = _torch_complex_min(torch.diag(A))* torch.eye(A.shape[0])
    return torch.abs(A-a) + a

def _positive_map(A):
    a = np.eye(A.shape[0])*np.min(np.diag(A))
    return np.abs(A-a) + a

"""
take absolute except diagonal part.
"""
def stoquastic(X, abs : bool = False):
    if abs:
        return abs(X)
    if isinstance(X, np.ndarray):
        ValueError("dimension should be 2") if (X.ndim != 2) else None
        return _positive_map(X)
    elif isinstance(X, torch.Tensor):
        ValueError("dimension should be 2") if (X.ndim != 2) else None
        return _positive_map_torch(X)
        pass
    else:
        TypeError("type of X is inappropriate")


def type_check(X):
    if isinstance(X, np.ndarray):
        ValueError("dimension should be 2") if (X.ndim != 2) else None
        return type(X)
    elif isinstance(X, torch.Tensor):
        ValueError("dimension should be 2") if (X.ndim != 2) else None
        return type(X)
    else:
        TypeError("type of X is inappropriate")
    
    if not (X.ndim == 2 and X.shape[0]==X.shape[1]):
        raise ValueError("X must be a 2D square matrix")


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


def eigvalsh_(A):
    _type = type(A)
    if _type == np.ndarray:
        return np.linalg.eigvalsh(A)
    elif _type == torch.Tensor:
        return torch.linalg.eigvalsh(A)
    else:
        TypeError("A should be either np_array or tensor")

def zeros_(D, _type):
    if _type == np.ndarray:
        return np.zeros(D)
    elif _type == torch.Tensor:
        return  torch.zeros(D)
    else:
        TypeError("A should be either np_array or tensor")

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
        TypeError("X should be either np_array or tensor")

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
        TypeError("X should be either np_array or tensor")

def einsum_(string, *arg):
    if isinstance(arg[0], torch.Tensor):
        return torch.einsum(string, *arg).contiguous()  
    elif isinstance(arg[0], np.ndarray):
        return np.einsum(string, *arg)
    else:
        TypeError("X should be either np_array or tensor")

def is_hermitian(X):
    if isinstance(X, torch.Tensor):
        return torch.all(X==X.T.conj())
    elif isinstance(X, np.ndarray):
        return np.all(X==X.T.conj())
    else:
        TypeError("X should be either np_array or tensor")    


def pick_negative(X):
    if isinstance(X, torch.Tensor):
        return torch.minimum(torch.tensor(0), X)
    elif isinstance(X, np.ndarray):
        return np.minimum(0, X)
    else:
        TypeError("X should be either np_array or tensor")  


def l2_measure(X):
    X = pick_negative(X)
    return (X**2).sum()

def l1_measure(X):
    X = pick_negative(X)
    return -X.sum()