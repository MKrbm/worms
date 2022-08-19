import numpy as np
from scipy import sparse
import os
import sys
sys.path.insert(0, "../nsp") 
from nsp.utils.func import *
from nsp.utils.print import beauty_array
from header import *
import argparse



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
lh2 = sum_ham(lh, bonds, 3, 2)
LH = sum_ham(lh2/2, [[0,1,2], [3, 4, 5]], 6, 2) + sum_ham(lh2, [[1, 2, 3], [2, 3, 4]], 6, 2)
# set_seed(33244233)
# X = LH

X = sum_ham(LH, [[0, 1], [1, 2], [3, 4]])
N = 8
loss_mes = nsp.loss.MES(X, [N, N])
t = 0.001
ret_min_grad = 1e10

best_fun = 1E10
for _ in range(2):
    model = nsp.model.UnitaryRiemanGenerator(N, dtype=torch.float64)
    solver = UnitaryTransTs(RiemanUnitaryCG, model, loss_mes, lr = 0.005, momentum=0.1)
    ret = solver.run(10000, False)
    if ret.fun < best_fun:
        print(f"\nbest_fun : {ret.fun}\n")
        best_fun = ret.fun
        best_model = model

lh = loss_mes._transform([best_model.matrix()]*loss_mes._n_unitaries, original = True).detach().numpy()

print(f"\nbest fun was : {best_fun}\n")
path = "../array/majumdar_ghosh/optimize8/0"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path,lh)
print("save : ", path+".npy")
beauty_array(lh,path + ".txt")





