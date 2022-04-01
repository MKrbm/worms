import numpy as np
from pyrsistent import v
from scipy.linalg import expm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from .optm_method import *


def loss_1(A):
#     A = -A**2
    A = torch.nn.ReLU()(-A)
    B = torch.zeros(1)
    if A.ndim!=3:
        A = A[None,:]
    else:
        for a in A:
            B += - torch.trace(a) + a.sum()
    return B


def loss_eig(A):
#     A = -A**2
    B = torch.zeros(1)
    if A.ndim!=3:
        A = A[None,:]
    else:
        for a in A:
            a_ = torch.abs(a)
            eigs = torch.linalg.eigvalsh(a_)
            B += eigs[-1]
    return B

def loss_eig_np(A):
    A = np.abs(A)
    return np.sum(np.linalg.eigvalsh(A)[:,-1])

def loss_2(M, V):
    assert len(M) == len(V), "inconsistent"
    r = torch.zeros(1)
    for i in range(len(M)):
        m = torch.abs(M[i])
        vs = torch.abs(V[i])
        for v in vs.T:
            r += v @ m @ v
    return r