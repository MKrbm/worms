from netrc import netrc
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


h1 = np.load("array/SS_bond_test1.npy").round(4)
h2 = np.load("array/SS_bond_test2.npy").round(4)
ho = np.load("array/SS_onsite.npy").round(4)
H = sparse.csr_matrix(J1*h2 + J2*ho) #this is the hamiltonian you wanna check the connectivity.
shift = -min(np.diagonal(H.toarray()).min(), 0)


H_tmp = H + shift*sparse.identity(H.shape[0])
print(H_tmp)

print("\n\n", "-"*20, "\n\n")



indices = (np.array((H_tmp).nonzero()).T).tolist()
index = [14,14]
index_ori = index.copy()
assert index in indices, "index is not listed in non-zero index for the hamiltonian"


indices_list = []
for i in range(4):
    for j in range(3):
        for i1 in range(4):
            for j1 in range(3):
                tmp = state_update(index.copy(), i, j+1)
                tmp = state_update(tmp, i1, j1+1)
                if (tmp in indices and tmp not in indices_list):
                    indices_list.append(tmp)
                    print(tmp, end = " ")
print("\n\n\n")
indices_list = [index]
cnt_list = [0]
index_rem = indices.copy()
index_bag = []
index_rem.remove(index)

mv_cnt = np.infty

prev_dict = {str(index) : index}

while indices_list:
    index = indices_list.pop(0)
    cnt = cnt_list.pop(0)
    index_bag.append(index.copy())
    for i in range(4):
        for j in range(3):
            for i1 in range(4):
                for j1 in range(3):
                    tmp = state_update(index.copy(), i, j+1)
                    tmp = state_update(tmp, i1, j1+1)
                    if (tmp in indices) and (tmp not in index_bag) and (tmp not in indices_list):
                        indices_list.append(tmp.copy())
                        cnt_list.append(cnt+1)
                        if (tmp == [4,4]):
                            print(index)
                        prev_dict[str(tmp)] = index.copy()
                        if H[tmp[0], tmp[1]]  < 0 and tmp[0] != tmp[1]:
                            if mv_cnt > cnt+1:
                                mv_cnt = cnt+1
                                neg_index = tmp

print("# of shortest steps need to reach first negative element is : ", mv_cnt, "\n where index is : ", neg_index)  

index_ = neg_index
while True:
    prev_index = prev_dict[str(index_)]
    if (index_==prev_index):
        break
    print(index_, "<- ", end="")
    index_ = prev_index
print(index_ori)