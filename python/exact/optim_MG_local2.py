import numpy as np
from scipy import sparse
import os
import sys
sys.path.insert(0, "../nsp") 
from nsp.utils.func import *
from nsp.utils.print import beauty_array
from header import *
import argparse
from save_npy import *



Sz = np.zeros([2,2])
Sz[0,0] = 1/2
Sz[1,1] = -1/2
Sx = np.zeros([2,2])
Sx[1,0] = 1/2
Sx[0,1] = 1/2
Sy = np.zeros([2,2], dtype=np.complex64)
Sy[1,0] = 1j/2
Sy[0,1] = -1j/2


SzSz = np.kron(Sz,Sz).real.astype(np.float64)
SxSx = np.kron(Sx,Sx).real.astype(np.float64)
SySy = np.kron(Sy,Sy).real.astype(np.float64)

lh = SzSz + SxSx + SySy

lh = -lh # use minus of local hamiltonian for monte-carlo (exp(-beta H ))
bonds = [[0,1], [0, 2], [1, 2]]
lh2 = sum_ham(lh/2, bonds, 3, 2)
LH = sum_ham(lh2/2, [[0,1,2], [3, 4, 5]], 6, 2) + sum_ham(lh2, [[1, 2, 3], [2, 3, 4]], 6, 2)

D = 8
L = 4
models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(L)]
model = copy.deepcopy(models[0])
loss = nsp.loss.MES(LH, [D,D], inv = model._inv)


# set_seed(33244233)
# X = LH
loss_mes = nsp.loss.MES(LH, [D, D])
t = 0.001
ret_min_grad = 1e10

best_fun = 1E10
for _ in range(1):
    models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(L)]
    cg2 = RiemanNonTransUnitaryCG([(models[i], models[(i+1)%L]) for i in range(L)], [loss]*L)
    solver2 = UnitaryNonTransTs(cg2)
    ret = solver2.run(100, disable_message=False)
    if ret.fun < best_fun:
        print(f"\nbest_fun : {ret.fun}\n")
        best_fun = ret.fun
        best_model = models

# lh = loss_mes._transform([model.matrix()]*loss_mes._n_unitaries).detach().numpy()
LHs = []
LHs = [loss._transform_kron([best_model[i].matrix(), best_model[(i+1)%L].matrix()], original=True).detach().numpy() for i in range(L)]


folder = f"../array/majumdar_ghosh/optimize8_{L}"
co = nsp.utils.base_conv.change_order

save_npy(folder, [co(lh, [8, 8]) for lh in LHs])
# save_npy(folder, LHs)


I_not1 = np.logical_not(np.eye(D))
I_not2 = np.logical_not(np.eye(D))
ML = np.kron(I_not1, np.eye(D))
MR = np.kron(np.eye(D), I_not2)
MI = np.kron(I_not1, I_not2) + np.eye(D*D)
LH_3s = [np.kron(np.eye(D), ML * LHs[(i+1)%L]) + np.kron(MR * LHs[i],np.eye(D)) for i in range(len(LHs))]
LH_2s = [MI * lh for lh in LHs]

LH_3s = [co(lh, [8, 8, 8]) for lh in LH_3s]
LH_2s = [co(lh, [8, 8]) for lh in LH_2s]

folder = f"../array/majumdar_ghosh/optimize8_{L}_af"
save_npy(folder, LH_2s + LH_3s)

# save_npy(folder, LHs)





