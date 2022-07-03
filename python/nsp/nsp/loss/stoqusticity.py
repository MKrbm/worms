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
from .base_class import BaseMatirxLoss


class L1(BaseMatirxLoss):
    """
    l1_measure of matrix (U @ X @ U.T) 
    """
    def __init__(self, X, act):
        super().__init__(X, act)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.target = 0

    def forward(self, A):
        return l1_measure(A)

    @staticmethod
    def _inverse(U):
        return U.T.conj()


class L2(BaseMatirxLoss):
    """
    l2_measure of matrix (U @ X @ U.T) 
    """
    def __init__(self, X, act):
        super().__init__(X, act)
        if not np.all(self.act == self.act[0]):
            raise NotImplementedError("the same unitary matrix acts on the given sites")
        self.target = 0

    def forward(self, A):
        return l2_measure(A)

    @staticmethod
    def _inverse(U):
        return U.T.conj()


