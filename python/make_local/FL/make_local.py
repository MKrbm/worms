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


J = [1, 0.8] # Jperpendicular, Jx (divided by Jparallel)
J = [float(j) for j in J]
models = [
    "original",
    "optm",
    "optm_2",
    "optm_nt", # non-trans
]

hams = ["spin", "dimer"]

loss = ["mes", "l1"]

parser = argparse.ArgumentParser(description='Optimize majumdar gosh')
parser.add_argument('-m','--model', help='lattice (model) Name', required=True, choices=models)
parser.add_argument('-loss','--loss', help='loss_methods', choices=loss, nargs='?', const='all',default="mes")
parser.add_argument('-ham','--ham', help='type of hamiltonian', choices=hams, nargs='?', const='all',default="spin")
parser.add_argument('-af','--add_first', help='adding first', action='store_true')
parser.add_argument('-L','--num_unit_cells', help='# of independent unit cell', type = int, default = 2)
parser.add_argument('-M','--num_iter', help='# of iterations', type = int, default = 10)
args = parser.parse_args()

af = args.add_first #add local hams first 
L = args.num_unit_cells
M = args.num_iter
loss_name = args.loss
if (loss_name == "mes"):
    loss_f = nsp.loss.MES
elif (loss_name == "l1"):
    loss_f = nsp.loss.L1
lat = args.model


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
lh = lh # We usually nead to add minus sign. But in the previous research, there is no such a sign, thus I removed it. 
LH = sum_ham(lh*1, [[0,2],[1,3]], 4, 2)
LH += sum_ham(lh*J[0]/2, [[0, 1], [2, 3]], 4, 2)
LH += sum_ham(lh*J[1], [[0, 3], [1, 2]], 4, 2)



jOrth = 1
jParallel = J[0]
jCross = J[1]

Id = np.eye(2)
Id2 = np.eye(4)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])

X1 = np.kron(X,Id)
X2 = np.kron(Id,X)

Y1 = np.kron(Y,Id)
Y2 = np.kron(Id,Y)

Z1 = np.kron(Z,Id)
Z2 = np.kron(Id,Z)

TX = X1 + X2 
TY = Y1 + Y2
TZ = Z1 + Z2

DX = X1 - X2 
DY = Y1 - Y2
DZ = Z1 - Z2

onSiteTerm = np.kron(TX.dot(TX),Id2) \
            + np.kron(TY.dot(TY),Id2) \
            + np.kron(TZ.dot(TZ),Id2)

tTerm = np.kron(TX,TX) + np.kron(TY,TY) + np.kron(TZ,TZ)
dTerm = np.kron(DX,DX) + np.kron(DY,DY) + np.kron(DZ,DZ)

LH2 = .5 * (jCross + jParallel)* tTerm \
                + .5*(jParallel - jCross) * dTerm  \
                + jOrth*(.5 *  onSiteTerm - 3 * np.kron(Id2,Id2))

LH = -LH
LH2 = -LH2.real
ham_type = args.ham
if (ham_type == "spin"):
    H = LH
elif (ham_type  == "dimer"):
    H = LH2


t = 0.001
ret_min_grad = 1e10
best_fun = 1E10
if lat == "original":
    save_npy(f"../../array/FL/original_Jp={(J[0]):.2}_Jx={(J[1]):.2}_{ham_type}", [H])


elif lat == "optm":
    D = 4
    loss = loss_f(H, [D, D], pout = False)
    for _ in range(M):
        seed = datetime.now()
        model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64, seed = None)
        solver = UnitaryTransTs(RiemanUnitaryCG, model, loss, lr = 0.005, momentum=0.1, af = af)
        ret = solver.run(1000, False)
        if ret.fun < best_fun:
            print(f"\nbest_fun : {ret.fun}\n")
            best_fun = ret.fun
            best_model = ret.model
    X = loss._transform([best_model.matrix()]*loss._n_unitaries, original = True).detach().numpy()
    H = nsp.utils.base_conv.change_order(X, [D, D])
    print(f"\nbest_fun : {best_fun}\n")
    save_npy(f"../../array/FL/optm_Jp={(J[0]):.2}_Jx={(J[1]):.2}_{ham_type}_{loss_name}", [H])

    I_not1 = torch.logical_not(torch.eye(D))
    I_not2 = torch.logical_not(torch.eye(D))
    ML = torch.kron(I_not1, torch.eye(D))
    MR = torch.kron(torch.eye(D), I_not2)
    MI = torch.kron(I_not1, I_not2) + torch.eye(D * D)    
    LH_3_site = (torch.kron(torch.eye(D), ML * X) + torch.kron(MR * X,torch.eye(D)))
    LH_2_site = MI * X
    LH_2_site = nsp.utils.base_conv.change_order(LH_2_site, [D, D])
    LH_3_site = nsp.utils.base_conv.change_order(LH_3_site, [D] * 3)

    save_npy(f"../../array/FL/optm_Jp={(J[0]):.2}_Jx={(J[1]):.2}_{ham_type}_{loss_name}_af", [LH_2_site, LH_3_site])
elif lat == "optm_nt":
    D = 4
    loss = loss_f(H, [D, D], pout = False)
    for _ in range(M):
        seed = randint(0, 2<<32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(L)]
        cg = RiemanNonTransUnitaryCG([(models[i], models[(i+1)%L]) for i in range(L)], [loss]*L, pout = False)
        solver = UnitaryNonTransTs(cg, af=False)
        ret = solver.run(2000, False)
        print(f"res = {ret.fun} / seed = {seed}")
        if ret.fun < best_fun:
            print(f"\nbest_fun updated : {ret.fun}\n")
            best_fun = ret.fun
            best_model = ret.model
    LHs = [loss._transform_kron([best_model[i].matrix(), best_model[(i+1)%L].matrix()], original=True).detach().numpy() for i in range(L)]
    folder = f"../../array/FL/optm_nt{L}Jp={(J[0]):.2}_Jx={(J[1]):.2}_{ham_type}_{loss_name}"
    co = nsp.utils.base_conv.change_order

    save_npy(folder, [co(lh, [D, D]) for lh in LHs])
    print(f"\nbest_fun : {best_fun}\n")
    I_not1 = np.logical_not(np.eye(D))
    I_not2 = np.logical_not(np.eye(D))
    ML = np.kron(I_not1, np.eye(D))
    MR = np.kron(np.eye(D), I_not2)
    MI = np.kron(I_not1, I_not2) + np.eye(D*D)
    LH_3s = [np.kron(np.eye(D), ML * LHs[(i+1)%L]) + np.kron(MR * LHs[i],np.eye(D)) for i in range(len(LHs))]
    LH_2s = [MI * lh for lh in LHs]

    LH_3s = [co(lh, [D]*3) for lh in LH_3s]
    LH_2s = [co(lh, [D]*2) for lh in LH_2s]

    save_npy(f"../../array/FL/optm_nt{L}Jp={(J[0]):.2}_Jx={(J[1]):.2}_{ham_type}_{loss_name}_af", LH_2s + LH_3s)



