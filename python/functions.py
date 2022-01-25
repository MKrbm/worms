import numpy as np
from numba import njit

@njit
def num2state(s, L = 4, nls = 2, rev=False):
  state = []
  S = 1<<nls
  for i in range(L):
    state.append(s%S)
    s >>= nls
  if rev:
    return state
  return state[::-1]

@njit
def state2num(state, nls = 2, rev=False):
  s = np.int32(0)
  nls = np.int32(nls)
  if rev:
    state = state[::-1]
  for i in range(len(state)):
    s <<= nls
    s += state[i]
  return s

from scipy import sparse
import numpy as np

def l2nl(ham, L, bond = [], nls = 1):
  index = sparse.find(ham)
  assert ham.shape[0] == 1<<(len(bond)*nls)
  for b in bond:
    assert (b < L) & (b >= 0), "site is inconsistent with L"
  H = sparse.csr_matrix((1<<(L*nls),1<<(L*nls)), dtype = np.float64)
  L_rem = L-len(bond)
  state = np.zeros(L, dtype = np.int64)
  for i in range(1<<(L_rem*nls)):
    tmp = num2state(i,L_rem,nls,rev=True)
    cnt = 0
    state = np.zeros(L, dtype=np.int64)
    for j in range(L):
      if j not in bond:
        # print(j)
        # print(state)
        # print(tmp)
        state[j] = tmp[cnt]
        cnt+=1
    for a, b, ele in zip(index[0], index[1], index[2]):
      a_ = num2state(a, nls=nls, L = len(bond),rev=True)
      b_ = num2state(b, nls=nls, L = len(bond),rev=True)

      for i1 in range(len(bond)):
        state[bond[i1]] = a_[i1]
      a_p = state2num(state,nls=nls,rev=True)
      for i1 in range(len(bond)):
        state[bond[i1]] = b_[i1]
      b_p = state2num(state,nls=nls,rev=True)
      H[a_p,b_p] = ele
  return H