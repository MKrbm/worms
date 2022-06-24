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


import numpy as np
from numba import njit


@njit
def get_nonzero_index(X, thres = 1e-10):
    return np.where(np.abs(X)>thres)


@njit
def num2state(s, L = 4, sps = 2, rev=False):
    state = []
#     S = 1<<nls
    for i in range(L):
        state.append(s%sps)
        s //= sps
    if rev:
        return state
    return state[::-1]

@njit
def state2num(state, sps = 2, rev=False):
    s = np.int32(0)
    sps = np.int32(sps)
    if rev:
        state = state[::-1]
    for i in range(len(state)):
        s *= sps
        s += state[i]
    return s

from scipy import sparse
import numpy as np

@njit
def _l2nl(ham, L, bond = np.array([]), sps = 2, thres=1e-10):
    index = get_nonzero_index(ham, thres)
    size = sps ** len(bond)
    # print(size)
#     print(ham.shape)
    assert ham.shape[0] == size
    
    for b in bond:
        assert (b < L) & (b >= 0), "site is inconsistent with L"
    H = np.zeros((sps**L,sps**L), dtype = ham.dtype)
    L_rem = L-len(bond)
    size_rem = sps ** L_rem
    state = np.zeros(L, dtype = np.int64)
    for i in range(size_rem):
        tmp = num2state(i,L_rem,sps,rev=True)
        cnt = 0
        state = np.zeros(L, dtype=np.int64)
        for j in range(L):
            if not np.any(j == bond):
                state[j] = tmp[cnt]
                cnt+=1
        for a, b in zip(index[0], index[1]):
            ele = ham[a,b]
            a_ = num2state(a, sps=sps, L = len(bond),rev=True)
            b_ = num2state(b, sps=sps, L = len(bond),rev=True)

            for i1 in range(len(bond)):
                state[bond[i1]] = a_[i1]
            # print(state)        
            a_p = state2num(state,sps=sps,rev=True)
            for i1 in range(len(bond)):
                state[bond[i1]] = b_[i1]
            # print(state) 
            b_p = state2num(state,sps=sps,rev=True)
            H[a_p,b_p] = ele
    return H

def l2nl(ham, L, bond = [], sps = 2, thres=1e-10):

    bond = np.array(bond)
    if isinstance(ham, np.ndarray):
        return _l2nl(ham, L, bond , sps, thres)
    elif isinstance(ham, sparse.csr.csr_matrix):
        ham = ham.toarray()
        return sparse.csr_matrix(_l2nl(ham, L, bond , sps, thres))
    else:
        ham = np.array(ham)
        return _l2nl(ham, L, bond , sps, thres)
    

def beauty_array(H_tmp, path = "array.txt"):
    try:
        H_tmp = H_tmp.toarray()
    except:
        pass
    with open(path, 'w') as f:
            f.write("{:>6} ".format(""))
            for j in range(H_tmp.shape[1]):
                    f.write("{:>6}    ".format(str(num2state(j, 2))))

            f.write("\n")
            for i in range(H_tmp.shape[0]):
                    f.write("{:>6}".format(str(num2state(i, 2))))
                    for j in range(H_tmp.shape[1]):        
                            f.write("{:>6.3f}, ".format(H_tmp[i,j]))
                    f.write("\n")


def set_origin(X, return_shift = False):

    if X.shape[0] != X.shape[1]:
        raise ValueError("X is not square matrix!")
    
    x = np.min(np.diag(X)) * np.eye(X.shape[0])
    X -= x
    if return_shift:
        return X, x
    else:
        return X

def exp_energy(E, beta):
    Z = np.exp(-beta*E)
    EZ = E*Z
    return np.sum(EZ)/np.sum(Z)