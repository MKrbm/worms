import numpy as np
from scipy import sparse
import os
import sys
sys.path.insert(0, "../../nsp") 
from header import *
from nsp.utils.print import beauty_array
import argparse
sys.path.insert(0, "..") 
from save_npy import *
from datetime import datetime
from random import randint

parser = argparse.ArgumentParser(description='Optimize majumdar gosh')

loss = ["mes", "l1"]


parser.add_argument('-loss','--loss', help='loss_methods', required=True, choices=loss)
parser.add_argument('-N','--num_unit_cells', help='# of independent unit cell', type = int, default = 4)
parser.add_argument('-M','--num_iter', help='# of iterations', type = int, default = 10)

args = vars(parser.parse_args())
M = args["num_iter"]
L = args["num_unit_cells"]
loss_name = args["loss"]

if (args["loss"] == "mes"):
    loss_ = nsp.loss.MES
elif (args["loss"] == "l1"):
    loss_ = nsp.loss.L1

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
models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(L)]
model = copy.deepcopy(models[0])
loss = loss_(LH, [D,D])


# set_seed(33244233)
# X = LH
t = 0.001
ret_min_grad = 1e10

best_fun = 1E10
print(f"iteration : {M}")
for _ in range(M):
    seed = randint(0, 2<<32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(L)]
    cg2 = RiemanNonTransUnitaryCG([(models[i], models[(i+1)%L]) for i in range(L)], [loss]*L, pout = False, lr = 0.005, momentum = 0.05)
    # cg2 = RiemanNonTransUnitaryCG([(models[i], models[(i+1)%L]) for i in range(L)], [loss]*L, pout = False)
    solver2 = UnitaryNonTransTs(cg2, af = True)
    ret = solver2.run(300, disable_message=False)
    print(f"res = {ret.fun} / seed = {seed}")
    if ret.fun < best_fun:
        print(f"\nbest_fun updated : {ret.fun}\n")
        best_fun = ret.fun
        best_model = ret.model

# lh = loss_mes._transform([model.matrix()]*loss_mes._n_unitaries).detach().numpy()
LHs = []

print(f"\nbest fun was : {best_fun}\n")


LHs = [loss._transform_kron([best_model[i].matrix(), best_model[(i+1)%L].matrix()], original=True).detach().numpy() for i in range(L)]
folder = f"../..//array/majumdar_ghosh/optim_{L}_{loss_name}"
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

folder = f"../..//array/majumdar_ghosh/optim_{L}_{loss_name}_af"
save_npy(folder, LH_2s + LH_3s)

# save_npy(folder, LHs)





