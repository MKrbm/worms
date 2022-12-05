import numpy as np
import argparse
import sys
sys.path.append('../../nsp')
from nsp.utils.base_conv import *
from header import *
from nsp.utils.func import *
from nsp.utils.local2global import *
from nsp.utils.print import beauty_array
sys.path.insert(0, "..") 
from save_npy import *
import argparse
from datetime import datetime
from random import randint

lattice = [
    "original",
    "dimer_original",
    "dimer_basis",
    "dimer_optm",
    "plq_original"
]

loss = ["mes", "l1"]
J = [1, 1] #J, J_D
best_fun = 1E10


parser = argparse.ArgumentParser(description='Optimize majumdar gosh')
parser.add_argument('-l','--lattice', help='lattice (model) Name', required=True, choices=lattice)
parser.add_argument('-loss','--loss', help='loss_methods', choices=loss, nargs='?', const='all',default="mes")
parser.add_argument('-M','--num_iter', help='# of iterations', type = int, default = 10)
parser.add_argument('-J','--coupling', help='coupling constant (NN)', type = float, default = 1)
args = parser.parse_args()

M = args.num_iter
loss_name = args.loss
if (loss_name == "mes"):
    loss_f = nsp.loss.MES
elif (loss_name == "l1"):
    loss_f = nsp.loss.L1
lat = args.lattice
J[0] = args.coupling

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

u = np.array([
    [0,1,0,0],
    [1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [0,0,0,1]
])

U = np.kron(u, u)
if lat == "original":
    H = lh
    path = ["../../array/SS/original"]
    save_npy(path[0], [H, H])

if lat == "dimer_original":
    # take lattice as dimer
    H1 = sum_ham(J[0]*lh, [[1,2],[1,3]], 4, 2)
    H1 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

    H2 = sum_ham(J[0]*lh, [[0,2],[0,3]], 4, 2)
    H2 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)
    save_npy(f"../../array/SS/dimer_original_J={J}", [H1, H2])

if lat == "dimer_basis":
    #* dimer lattice and single-triplet basis
    H1 = sum_ham(J[0]*lh, [[1,2],[1,3]], 4, 2)
    H1 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

    H2 = sum_ham(J[0]*lh, [[0,2],[0,3]], 4, 2)
    H2 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)
    save_npy(f"../../array/SS/dimer_basis_J_{J}", [U.T@H1@U, U.T@H2@U]) 

if lat == "dimer_optm":
    # dimer lattice and dimer basis
    H1 = sum_ham(J[0]*lh, [[1,2],[1,3]], 4, 2)
    H1 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

    H2 = sum_ham(J[0]*lh, [[0,2],[0,3]], 4, 2)
    H2 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

    D = 4
    loss1 = loss_f(H1, [D, D], pout = False)
    loss2 = loss_f(H2, [D, D], pout = False)

    for _ in range(M):
        seed = randint(0, 2<<32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
    #     models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(2)]
        model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
        cg = RiemanNonTransUnitaryCG([(model, model)]*2, [loss1, loss2], pout = False)
        solver = UnitaryNonTransTs(cg, af=False)
        ret = solver.run(2000, False)
        print(f"res = {ret.fun} / seed = {seed}")
        if ret.fun < best_fun:
            print(f"\nbest_fun updated : {ret.fun}\n")
            best_fun = ret.fun
            best_model = ret.model
    print("best fun is : ", best_fun)
    U = best_model[0].matrix()
    H1_ = loss1._transform_kron([U, U], original=True).detach().numpy()
    H2_ = loss2._transform_kron([U, U], original=True).detach().numpy()
    save_npy(f"../../array/SS/dimer_optm_J={J}", [H1_, H2_])
    


if lat == "plq_original": #plaquette original
    # dimer lattice and dimer basis
    square1 = [[0,1], [1,2], [2,3], [3,0]]
    square2 = [[4,5], [5,6], [6,7], [7,4]]
    H1 = sum_ham(J[0]*lh/4, square1 + square2, 8, 2)
    H1 += sum_ham(J[1]*lh, [[2,4]], 8, 2)

    H2 = sum_ham(J[0]*lh/4, square1 + square2, 8, 2)
    H2 += sum_ham(J[1]*lh, [[1,7]], 8, 2)
    save_npy(f"../../array/SS/plq_original_J={J}", [H1, H2])
