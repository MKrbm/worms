import numpy as np
import torch

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
        return _absolute_map(X, sign)
    elif isinstance(X, torch.Tensor):
        ValueError("dimension should be 2") if (X.ndim != 2) else None
        return _absolute_map_torch(X, sign)
        pass
    else:
        TypeError("type of X is inappropriate")