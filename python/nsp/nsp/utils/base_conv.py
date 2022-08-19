from numba import njit
import numpy as np

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

def change_order(X, dof_site, order="F"):
    return X.reshape(dof_site*2, order="F").reshape(X.shape)