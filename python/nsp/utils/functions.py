import numpy as np
import torch

def absolute_map(X, neg = True):
    if neg:
        sign = -1
    else:
        sign = 1
    
    for i in range(X.shape[0]):
        for j in range(i,X.shape[1]):
            X[i,j] = np.abs(X[i,j])*sign
            X[j,i] = np.abs(X[i,j])*sign
    return X


def absolute_map_torch(X, neg = True):
    if neg:
        sign = -1
    else:
        sign = 1
    
    for i in range(X.shape[0]):
        for j in range(i,X.shape[1]):
            X[i,j] = torch.abs(X[i,j])*sign
            X[j,i] = torch.abs(X[i,j])*sign
    return X