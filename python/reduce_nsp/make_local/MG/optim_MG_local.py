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


loss = ["mes", "l1"]


parser = argparse.ArgumentParser(description='Reproduce original paper results')
parser = argparse.ArgumentParser(description='Reproduce original paper results')
parser.add_argument('-af','--add_first', help='adding first', action='store_true')
parser.add_argument('-loss','--loss', help='loss_methods', required=True, choices=loss)
args = parser.parse_args()



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
# set_seed(33244233)

X = LH # for N = 8
N = 8

if (args.loss == "mes"):
    loss = nsp.loss.MES(X, [N, N])
elif (args.loss == "l1"):
    loss = nsp.loss.L1(X, [N, N])
loss_mes = nsp.loss.MES(X, [N, N])
t = 0.001
ret_min_grad = 1e10
best_fun = 1E10
for _ in range(10):
    seed = datetime.now()
    model = nsp.model.UnitaryRiemanGenerator(N, dtype=torch.float64)
    solver = UnitaryTransTs(RiemanUnitaryCG, model, loss, lr = 0.005, momentum=0.1, af = True)
    ret = solver.run(1000, False)
    if ret.fun < best_fun:
        print(f"\nbest_fun : {ret.fun}\n")
        best_fun = ret.fun
        best_model = model
    best_loss = loss_mes([model.matrix()]*loss._n_unitaries).detach().numpy()
    print(f"loss(mes) : {best_loss}")

lh = loss._transform([best_model.matrix()]*loss._n_unitaries, original = True).detach().numpy()
best_loss = loss_mes([model.matrix()]*loss._n_unitaries).detach().numpy()
print(f"\nbest fun was : {best_fun}\n")
print(f"loss(mes) : {best_loss}")
act = loss.act.tolist()
LH = nsp.utils.base_conv.change_order(lh, act)
save_npy(f"../../array/majumdar_ghosh/optim_{args.loss}", [LH])
I_not1 = torch.logical_not(torch.eye(act[0]))
I_not2 = torch.logical_not(torch.eye(act[1]))
ML = torch.kron(I_not1, torch.eye(act[1]))
MR = torch.kron(torch.eye(act[0]), I_not2)
MI = torch.kron(I_not1, I_not2) + torch.eye(np.prod(act))    
LH_3_site = (torch.kron(torch.eye(N),ML * lh) + torch.kron(MR * lh,torch.eye(N)))
LH_2_site = MI * lh
LH_2_site = nsp.utils.base_conv.change_order(LH_2_site, act)
LH_3_site = nsp.utils.base_conv.change_order(LH_3_site, act + [act[0]])

save_npy(f"../../array/majumdar_ghosh/optim_{args.loss}_af", [LH_2_site, LH_3_site])








