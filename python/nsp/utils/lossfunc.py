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
                a_ = make_positive(a)
                eigs = torch.linalg.eigvalsh(a_)
                B += eigs[-1]
        else:
            a_ = torch.zeros_like(A[0])
            for a in A:
                a_ += make_positive(a)
            eigs = torch.linalg.eigvalsh(a_)
            B += eigs[-1]
    return B

def make_positive(A):
    # a = torch.min(torch.diag(A))* torch.eye(A.shape[0])
    # return torch.abs(A-a) + a
    return torch.abs(A)

def make_positive_np(A):
    # a = np.eye(A.shape[0])*np.min(np.diag(A))
    # return np.abs(A-a) + a
    return np.abs(A)

def loss_eig_np(A, add):
    if not add:
        for i in range(A.shape[0]):
            A[i] = make_positive_np(A[i])
        return np.sum(np.linalg.eigvalsh(A)[:,-1])
    else:
        B = np.zeros_like(A[0])
        for a in A:
            B+=make_positive_np(a)
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