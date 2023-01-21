import numpy as np
import argparse
import sys
sys.path.append('../../reduce_nsp')
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

# packages for multiprocessing
import psutil
import torch.multiprocessing as mp
import multiprocessing

lattice = [
    "original",
    "dimer_original",
    "singlet",
    "dimer_optim",
    "plq_original"
]

loss = ["mes", "l1"]
J = [1, 1] #J, J_D
message = False
num_processes = psutil.cpu_count(logical=False)

parser = argparse.ArgumentParser(description='Optimize majumdar gosh')
parser.add_argument('-l','--lattice', help='lattice (model) Name', required=True, choices=lattice)
parser.add_argument('-loss','--loss', help='loss_methods', choices=loss, nargs='?', const='all',default="mes")
parser.add_argument('-M','--num_iter', help='# of iterations', type = int, default = 10)
parser.add_argument('-J','--coupling', help='coupling constant (NN)', type = float, default = 1)
parser.add_argument('-P','--num_processes', help='# of parallel process', type = int, default = num_processes)
args = parser.parse_args()
print(args)
num_processes = min(args.num_processes, num_processes)
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
I = np.eye(2)
o = np.kron(I, Sz) + np.kron(Sz, I)

lh = -lh # use minus of local hamiltonian for monte-carlo (exp(-beta H ))

preceision = 5 # elements smaller than 1E-6 in magnitude reduce to zero

u = np.array([
    [0,1,0,0],
    [1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [0,0,0,1]
])
U = np.kron(u, u)

O = np.kron(o, o)
O = np.round(O, preceision) # local order operator  

def dimer_optim(M, queue, i):
    best_fun = 1E10
    torch.set_num_threads(1)
    message = True
    for _ in range(M):
        seed = randint(0, 2<<32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
    #     models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(2)]
        try:
            model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
            cg = RiemanNonTransUnitaryCG([(model, model)]*2, [loss1, loss2], pout = False)
            solver = UnitaryNonTransTs(cg, af=False)
            ret = solver.run(2000,  not message)
            if message:
                print(f"res = {ret.fun} / seed = {seed}")
            if ret.fun < best_fun:
                if message:
                    print("-"*20)
                    print(f"best_fun updated : {ret.fun}")
                    print("-"*20)
                best_res = copy.copy(ret)
        except Exception as e:
            print(f"Failed to optimize : {str(e)}")
            pass
    # print(multiprocessing.current_process(), i )
    queue.put(best_res)
# main part
if __name__ == "__main__":
    J_str = str(J).replace(" ", "")
    print(f"number of process is : {num_processes}")
    if lat == "original":
        H = lh
        path = ["array/original"]
        save_npy(path[0], [H, H])
        save_npy("array/original_z", Sz)

    if lat == "dimer_original":
        # take lattice as dimer
        H1 = sum_ham(J[0]*lh, [[1,2],[1,3]], 4, 2)
        H1 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

        H2 = sum_ham(J[0]*lh, [[0,2],[0,3]], 4, 2)
        H2 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)
        save_npy(f"array/dimer_original_J_{J_str}", [H1, H2])



        O /= 4 # divide by 4 because overwrapped 4 times. 
        try:
            save_npy(f"array/dimer_original_J_{J_str}_z", [O/(H1 + 1E-8), O/(H2+1E-8)]) 
        except:
            print("support of operator is not covered by a support of Hamiltonian")


    if lat == "singlet":
        #* dimer lattice and single-triplet basis
        H1 = sum_ham(J[0]*lh, [[1,2],[1,3]], 4, 2)
        H1 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

        H2 = sum_ham(J[0]*lh, [[0,2],[0,3]], 4, 2)
        H2 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)
        save_npy(f"array/singlet_J_{J_str}", [U.T@H1@U, U.T@H2@U]) 
        
        O /= 4 # divide by 4 because overwrapped 4 times. 

        try:
            save_npy(f"array/singlet_J_{J_str}_z",  [O/(H1 + 1E-8), O/(H2+1E-8)]) 
        except:
            print("support of operator is not covered by a support of Hamiltonian")

    if lat == "dimer_optim":
        # dimer lattice and dimer basis
        H1 = sum_ham(J[0]*lh, [[1,2],[1,3]], 4, 2)
        H1 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

        H2 = sum_ham(J[0]*lh, [[0,2],[0,3]], 4, 2)
        H2 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

        D = 4
        loss1 = loss_f(H1, [D, D], pout = False)
        loss2 = loss_f(H2, [D, D], pout = False)
        res = []
        M_par_cpu = int(M / num_processes) + 1
        for i in range(num_processes):
            manager = mp.Manager()
            q = manager.Queue()
            p = mp.Process(target=dimer_optim, args=(M_par_cpu,q, i))
            p.start()
            res.append((p, q))
        best_fun = 1E10
        for p, q in res:
            p.join()
            res = q.get()
            if res.fun < best_fun:
                best_fun = res.fun
                best_model = res.model
        print("best fun is : ", best_fun)
        U = best_model[0].matrix()
        H1 = loss1._transform_kron([U, U], original=True).detach().numpy()
        H2 = loss2._transform_kron([U, U], original=True).detach().numpy()
        U = U.detach().numpy()
        save_npy(f"array/dimer_optim_J_{J_str}_M_{M}", [H1, H2])
        save_npy(f"array/U_dimer_optim_J_{J_str}_M_{M}", [U])
        U = np.kron(U, U)

        O /= 4 # divide by 4 because overwrapped 4 times.

        try:
            save_npy(f"array/dimer_optim_J_{J_str}_z_M_{M}",  [O/(H1 + 1E-8), O/(H2+1E-8)])
        except:
            print("support of operator is not covered by a support of Hamiltonian")


        

        
    if lat == "plq_original": #plaquette original
        # dimer lattice and dimer basis
        square1 = [[0,1], [1,2], [2,3], [3,0]]
        square2 = [[4,5], [5,6], [6,7], [7,4]]
        H1 = sum_ham(J[0]*lh/4, square1 + square2, 8, 2)
        H1 += sum_ham(J[1]*lh, [[2,4]], 8, 2)

        H2 = sum_ham(J[0]*lh/4, square1 + square2, 8, 2)
        H2 += sum_ham(J[1]*lh, [[1,7]], 8, 2)
        save_npy(f"array/plq_original_J_{J_str}_M_{M}", [H1, H2])
