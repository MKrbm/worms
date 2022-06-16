import numpy as np
from pyrsistent import v
from scipy.linalg import expm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from .optm_method import *


def loss_1(A, add = False):
#     A = -A**2
    A = torch.nn.ReLU()(-A)
    B = torch.zeros(1)
    if A.ndim!=3:
        A = A[None,:]
    else:
        for a in A:
            B += - torch.trace(a) + a.sum()
    return B


def loss_eig(A, add):
#     A = -A**2
    B = torch.zeros(1)
    if A.ndim!=3:
        A = A[None,:]
    else:
        if add:
            for a in A:
                a_ = positive_map(a)
                eigs = torch.linalg.eigvalsh(a_)
                B += eigs[-1]
        else:
            a_ = torch.zeros_like(A[0])
            for a in A:
                a_ += positive_map(a)
            eigs = torch.linalg.eigvalsh(a_)
            B += eigs[-1]
    return B



def abs_map(A):
    return torch.abs(A)

def abs_map_np(A):
    return np.abs(A)

def positive_map(A, abs=False):
    if abs:
        return abs_map(A)
    a = torch.min(torch.diag(A))* torch.eye(A.shape[0])
    return torch.abs(A-a) + a

def positive_map_np(A, abs=False):
    if abs:
        return abs_map_np(A)
    a = np.eye(A.shape[0])*np.min(np.diag(A))
    return np.abs(A-a) + a



def loss_eig_np(A, add = False):
    if (A.ndim == 2):
        A = A[None, :, :]
    if not add:
        for i in range(A.shape[0]):
            A[i] = positive_map_np(A[i])
        return np.sum(np.linalg.eigvalsh(A)[:,-1])
    else:
        B = np.zeros_like(A[0])
        for a in A:
            B+=positive_map_np(a)
        return np.linalg.eigvalsh(B)[-1]
def loss_2(M, V):
    assert len(M) == len(V), "inconsistent"
    r = torch.zeros(1)
    for i in range(len(M)):
        m = torch.abs(M[i])
        vs = torch.abs(V[i])
        for v in vs.T:
            r += v @ m @ v
    return r


def reweight_loss(A, E_t, X = None, lam = 1, thres = 1E-8, type_ = 0, prod = 0):
    E = torch.max(torch.linalg.eigvals(A).real)
    
    loss = torch.maximum(E-E_t, torch.tensor(0))
    assert 0 <= prod <= 1, "prod is [0, 1)"
    if X is None:
        return loss
    else:
        index = np.argwhere(torch.abs(A) > thres)
        wr = torch.abs(X[index[0,:], index[1,:]] / A[index[0,:], index[1,:]])

        if type_ == 2:
            add = torch.max(wr)

        if prod and type_ == 1:
            r_index = np.random.choice(wr.shape[0],size=int(wr.shape[0]*prod), replace=True)

            add = torch.maximum(wr[r_index].prod()*lam, torch.tensor(lam))
            # print(wr[r_index])
        else:
            add = wr.sum()

        return loss + add * lam, loss
