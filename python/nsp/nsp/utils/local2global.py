from numba import njit
import numpy as np
from scipy import sparse
from .base_conv import num2state, state2num

@njit
def get_nonzero_index(X, thres = 1e-10):
    return np.where(np.abs(X)>thres)


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
    
