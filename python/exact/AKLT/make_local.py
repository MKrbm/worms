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


J = [1, 0.5]
J = [float(j) for j in J]
models = [
    "original",
    "optm1",
    "optm2",
    "optm2_nt",
    "optm3"
]

loss = ["mes", "l1"]

parser = argparse.ArgumentParser(description='Optimize majumdar gosh')
parser.add_argument('-m','--model', help='lattice (model) Name', required=True, choices=models)
parser.add_argument('-loss','--loss', help='loss_methods', choices=loss, nargs='?', const='all',default="mes")
parser.add_argument('-L','--num_unit_cells', help='# of independent unit cell', type = int, default = 2)
parser.add_argument('-M','--num_iter', help='# of iterations', type = int, default = 10)
args = vars(parser.parse_args())
L = args["num_unit_cells"]
M = args["num_iter"]
loss_name = args["loss"]
if (loss_name == "mes"):
    loss_f = nsp.loss.MES
elif (loss_name == "l1"):
    loss_f = nsp.loss.L1
lat = args["model"]

Sz = np.zeros([3,3])
Sz[0,0] = 1
Sz[2,2] = -1
Sx = np.zeros([3, 3])
Sx[1,0] = Sx[0,1] = Sx[2,1] = Sx[1,2] = 1/np.sqrt(2)
Sy = np.zeros([3, 3], dtype=np.complex64)
Sy[1,0] = Sy[2,1] = 1j/np.sqrt(2)
Sy[0,1] = Sy[1,2] = -1j/np.sqrt(2)


SzSz = np.kron(Sz,Sz).astype(np.float64)
SxSx = np.kron(Sx,Sx).astype(np.float64)
SySy = np.kron(Sy,Sy).astype(np.float64)

lh = SzSz + SxSx + SySy

t = 0.001
ret_min_grad = 1e10
best_fun = 1E10
lh = -(lh*J[0] + lh@lh*J[1])
if lat == "original":
    H = lh
    save_npy(f"../../array/AKLT/original_J={J[1]:.2}", [H])


elif lat == "optm1":
    H = lh
    D = 3 ** 2
    loss = loss_f(H, [D, D], pout = False)
    for _ in range(M):
        seed = datetime.now()
        model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64, seed = None)
        solver = UnitaryTransTs(RiemanUnitaryCG, model, loss, lr = 0.005, momentum=0.1)
        ret = solver.run(1000, False)
        if ret.fun < best_fun:
            print(f"\nbest_fun : {ret.fun}\n")
            best_fun = ret.fun
            best_model = model
    H = nsp.utils.base_conv.change_order(H, [D, D])
    save_npy(f"../../array/AKLT/optm1", [H])

elif lat == "optm3":
    bonds = [[0, 1], [1, 2], [3, 4], [4, 5]]
    LH = sum_ham(lh/2, bonds, 6, 3)
    LH += sum_ham(lh, [[2, 3]], 6, 3)
    D = 3 ** 3
    loss = loss_f(LH, [D, D], pout = False)
    for _ in range(M):
        seed = randint(0, 2<<32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
        solver = UnitaryTransTs(RiemanUnitarySGD, model, loss, lr = 0.001, momentum=0.1)
        ret = solver.run(10000, False)
        print(f"res = {ret.fun} / seed = {seed}")
        if ret.fun < best_fun:
            print(f"\nbest_fun updated : {ret.fun}\n")
            best_fun = ret.fun
            best_model = model
    lh = loss._transform([best_model.matrix()]*loss._n_unitaries, original = True).detach().numpy()
    H = nsp.utils.base_conv.change_order(lh, [D, D])
    # H = stoquastic(LH)
    save_npy(f"../../array/AKLT/optm3_{loss_name}", [H])


elif lat == "optm2":
    bonds = [[0, 1], [2, 3]]
    LH = sum_ham(lh/2, bonds, 4, 3)
    LH += sum_ham(lh, [[1, 2]], 4, 3)
    D = 3 ** 2
    loss = loss_f(LH, [D, D], pout = False)
    loss_mes = nsp.loss.MES(LH, [D, D], pout = False)
    for _ in range(M):
        seed = randint(0, 2<<32 - 1)
        seed = 692441695
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
        solver = UnitaryTransTs(RiemanUnitarySGD, model, loss, lr = 0.005, momentum=0.1, af = False)
        ret = solver.run(3000, False)
        # if loss_name == "mes":
        #     solver = UnitaryTransTs(RiemanUnitaryCG, model, loss_mes, lr = 0.005, momentum=0.1, af = False)
        #     ret = solver.run(1000, False)
        print(f"res = {ret.fun} / seed = {seed}")
        best_loss = loss_mes([ret.model.matrix()]*loss._n_unitaries).detach().numpy()
        print(f"loss(mes) : {best_loss}")

        if ret.fun < best_fun:
            print(f"\nbest_fun updated : {ret.fun}\n")
            best_fun = ret.fun
            best_model = ret.model
    lh = loss._transform([best_model.matrix()]*loss._n_unitaries, original = True).detach().numpy()
    best_loss = loss([best_model.matrix()]*loss._n_unitaries).detach().numpy()
    print(f"best loss : {best_loss}")
    H = nsp.utils.base_conv.change_order(lh, [D, D])
    save_npy(f"../../array/AKLT/optm2_{loss_name}_J={J[1]:.2}", [H])

    I_not1 = torch.logical_not(torch.eye(D))
    I_not2 = torch.logical_not(torch.eye(D))
    ML = torch.kron(I_not1, torch.eye(D))
    MR = torch.kron(torch.eye(D), I_not2)
    MI = torch.kron(I_not1, I_not2) + torch.eye(D**2)    
    LH_3_site = (torch.kron(torch.eye(D),ML * lh) + torch.kron(MR * lh,torch.eye(D)))
    LH_2_site = MI * lh
    LH_2_site = nsp.utils.base_conv.change_order(LH_2_site, [D, D])
    LH_3_site = nsp.utils.base_conv.change_order(LH_3_site, [D, D, D])
    save_npy(f"../../array/AKLT/optm2_{loss_name}_J={J[1]:.2}_af", [LH_2_site, LH_3_site])


elif lat == "optm2_nt":
    bonds = [[0, 1], [2, 3]]
    LH = sum_ham(lh/2, bonds, 4, 3)
    LH += sum_ham(lh, [[1, 2]], 4, 3)
    D = 3 ** 2
    loss = loss_f(LH, [D, D], pout = False)
    for _ in range(M):
        seed = randint(0, 2<<32 - 1)
        seed = 1561045271
        torch.manual_seed(seed)
        np.random.seed(seed)
        models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(L)]
        cg = RiemanNonTransUnitaryCG([(models[i], models[(i+1)%L]) for i in range(L)], [loss]*L, pout = False, lr = 0.01)
        solver = UnitaryNonTransTs(cg, af=False)
        ret = solver.run(1000, False)
        print(f"res = {ret.fun} / seed = {seed}")
        if ret.fun < best_fun:
            print(f"\nbest_fun updated : {ret.fun}\n")
            best_fun = ret.fun
            best_model = ret.model
    LHs = [loss._transform_kron([best_model[i].matrix(), best_model[(i+1)%L].matrix()], original=True).detach().numpy() for i in range(L)]
    folder = f"../../array/AKLT/optm2_nt{L}_{loss_name}_J={J[1]:.2}"
    co = nsp.utils.base_conv.change_order

    save_npy(folder, [co(lh, [D, D]) for lh in LHs])

    I_not1 = np.logical_not(np.eye(D))
    I_not2 = np.logical_not(np.eye(D))
    ML = np.kron(I_not1, np.eye(D))
    MR = np.kron(np.eye(D), I_not2)
    MI = np.kron(I_not1, I_not2) + np.eye(D*D)
    LH_3s = [np.kron(np.eye(D), ML * LHs[(i+1)%L]) + np.kron(MR * LHs[i],np.eye(D)) for i in range(len(LHs))]
    LH_2s = [MI * lh for lh in LHs]

    LH_3s = [co(lh, [D]*3) for lh in LH_3s]
    LH_2s = [co(lh, [D]*2) for lh in LH_2s]

    save_npy(f"../../array/AKLT/optm2_nt{L}_{loss_name}_J={J[1]:.2}_af", LH_2s + LH_3s)
