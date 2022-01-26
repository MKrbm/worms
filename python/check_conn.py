import numpy as np
import numpy as np
from scipy import sparse


def num2state(s, L = 4):
    state = []
    for i in range(L):
        state.append(s%4)
        s >>= 2
    return state

def state_update(index, leg, fl):
    d = leg // 2
    l = leg % 2
    index[d] = index[d] ^ (fl << (l * 2))
    return index

J1 = 0.4
J2 = 1


h1 = np.load("array/SS_bond1.npy").round(4)
h2 = np.load("array/SS_bond2.npy").round(4)
ho = np.load("array/SS_onsite.npy").round(4)
H = sparse.csr_matrix(J1*h1 + J2*ho) #this is the hamiltonian you wanna check the connectivity.
shift = -min(np.diagonal(H.toarray()).min(), 0)


H_tmp = H + shift*sparse.identity(H.shape[0])
print(H_tmp)


indices = (np.array((H_tmp).nonzero()).T).tolist()
indices_list = [indices[0]]
index_rem = indices.copy()
index_bag = []
index_rem.remove(indices[0])


while indices_list:
    index = indices_list.pop()
#     print("\n\n")
#     print(index)
    index_bag.append(index)
    for i in range(4):
        for j in range(3):
            for i1 in range(4):
                for j1 in range(3):
                    tmp = state_update(index.copy(), i, j+1)
                    tmp = state_update(tmp, i1, j1+1)
                    if (tmp in indices) and (tmp not in index_bag) and (tmp not in indices_list):
                        indices_list.append(tmp)
                        if tmp in index_rem:
                            index_rem.remove(tmp)
print("remaining : " , index_rem)