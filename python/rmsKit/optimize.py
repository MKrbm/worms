from make_local import KH
from make_local import save_npy
# from jax import numpy as jnp
import argparse
import sys
import copy
from random import randint
import torch
import numpy as np
sys.path.append('../../reduce_nsp')
import nsp
from nsp.solver import SymmSolver, UnitaryTransTs, UnitaryNonTransTs
from nsp.optim import *
import psutil
num_processes = psutil.cpu_count(logical=False)

import torch.multiprocessing as mp


models = ["KH", "HXYZ", ]
loss = ["mes", "l1", "bes", "none"]

parser = argparse.ArgumentParser(
    description="exact diagonalization of shastry_surtherland"
)
parser.add_argument(
    "-m", "--model", help="model (model) Name", required=True, choices=models
)
parser.add_argument(
    "-Jz", "--coupling_z", help="coupling constant (Jz)", type=float, default=1
)  # SxSx + SySy +
parser.add_argument("-Jx", "--coupling_x", help="coupling constant (Jx)", type=float)
parser.add_argument("-Jy", "--coupling_y", help="coupling constant (Jy)", type=float)
parser.add_argument("-Hx", "--mag_x", help="magnetic field", type=float, default=0)
parser.add_argument("-Hz", "--mag_z", help="magnetic field", type=float, default=0)
parser.add_argument("-T", "--temperature", help="temperature", type=float)
parser.add_argument('-M','--num_iter', help='# of iterations', type = int, default = 10)
parser.add_argument('-P','--num_processes', help='# of parallel process', type = int, default = num_processes)

parser.add_argument(
    "-u",
    "--unitary_algorithm",
    help="algorithm determine local unitary matrix",
    default="original",
)
parser.add_argument('-loss','--loss', help='loss_methods', choices=loss, nargs='?', const='all',default="none")

# parser.add_argument('-N','--n_samples', help='# of samples', type = int, default = 50)
# parser.add_argument('-m','--n_Krylov', help='dimernsion of Krylov space', type = int, default = 50)


def mpi_symm_optim( sps, loss_list, M, queue, i, seed):
    # lr = 0.002
    # momentum = 0.1
    best_fun = 1E10
    torch.set_num_threads(2)
    message = True
    best_res = None
    torch.manual_seed(seed)
    np.random.seed(seed)
    for _ in range(M):
        momentum = np.random.lognormal(np.log(0.05), 0.5)
        lr = np.random.lognormal(np.log(0.003), 2)
        print(seed, lr, momentum)
        print(momentum)
    #     models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(2)]
        try:
            model = nsp.model.UnitaryRiemanGenerator(sps, dtype=torch.float64)
            cg = RiemanNonTransUnitarySGD([(model, model)]*len(loss_list), loss_list, lr = lr, momentum = momentum)
            solver = UnitaryNonTransTs(cg, af=False)
            ret = solver.run(10000,  i != 0, cut_off_cnt = 500)
            if message:
                print(f"res = {ret.fun} / seed = {seed}")
            if ret.fun < best_fun:
                if message:
                    print("-"*20)
                    print(f"best_fun updated : {ret.fun} / process : {i}")
                    print("-"*20)
                    print("start optimization with settings : ", f"lr = {lr} / momentum = {momentum} ")
                best_res = copy.copy(ret)
                best_fun = ret.fun
        except Exception as e:
            print(f"Failed to optimize : {str(e)}")
            pass
    # print(multiprocessing.current_process(), i )
    queue.put(best_res)


if __name__ == "__main__":
    args = parser.parse_args()
    num_processes = min(args.num_processes, num_processes)
    M = args.num_iter
    p = dict(
        Jx=args.coupling_x if args.coupling_x is not None else args.coupling_z,
        Jy=args.coupling_y if args.coupling_y is not None else args.coupling_z,
        Jz=args.coupling_z,
        hx=args.mag_x,
        hz=args.mag_z,
    )
    a = ""
    for k, v in p.items():
        v = float(v)
        a += f"{k}_{v:.4g}_"
    params_str = a[:-1]
    ua = args.unitary_algorithm
    path = f"array/{args.model}/{ua}/{args.loss}/{params_str}"

    h_list = []
    sps = 2
    if args.model == "KH":
        h_list, sps = KH(ua, p)
        pass

    if args.loss != "none" and h_list:
        loss_name = args.loss
        if (loss_name == "mes"):
            loss_f = nsp.loss.MES
        elif (loss_name == "l1"):
            loss_f = nsp.loss.L1
        elif (loss_name == "bes"):
            loss_f = nsp.loss.BES
        else:
            raise ValueError("Not implemented yet")
        message = True

        loss_list = [
            loss_f(h, [sps, sps], pout = False) for h in h_list
        ]
        res = []
        M_par_cpu = int(M / num_processes) 
        best_model = []
        print("Calculation start multiprocessing : ", num_processes)
        seed_ = []
        for i in range(num_processes):
            seed = randint(0, 2<<32 - 1)
            seed_.append(seed)
            manager = mp.Manager()
            q = manager.Queue()
            p = mp.Process(target=mpi_symm_optim, args=(sps, loss_list, M_par_cpu, q, i, seed))
            p.start()
            res.append((p, q))
        best_fun = 1E10
        for i, (p, q) in enumerate(res):
            p.join()
            res = q.get()
            print(f"res = {res.fun} / process = {i} / seed = {seed_[i]}")
            if res.fun < best_fun:
                best_fun = res.fun
                best_model = res.model
        print("best fun is : ", best_fun)
        u = best_model[0].matrix()
        h_list = [
            l._transform_kron([u, u], original=True).detach().numpy() for l in loss_list
        ]
        u = u.detach().numpy()
        save_npy(f"{path}/M_{M}/H", [h for h in h_list])
        save_npy(f"{path}/M_{M}/u", [u])
        
        # for _ in range(args.num_iter):
        #     seed = randint(0, 2<<32 - 1)
        #     torch.manual_seed(seed)
        #     np.random.seed(seed)
        #     best_fun = 1E10
        # #     models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(2)]
        #     try:
        #         model = nsp.model.UnitaryRiemanGenerator(sps, dtype=torch.float64)
        #         cg = RiemanNonTransUnitaryCG([(model, model)]*3, loss_list, pout = False)
        #         solver = UnitaryNonTransTs(cg, af=False)
        #         ret = solver.run(2000,  not message)
        #         if message:
        #             print(f"res = {ret.fun} / seed = {seed}")
        #         if ret.fun < best_fun:
        #             if message:
        #                 print("-"*20)
        #                 print(f"best_fun updated : {ret.fun}")
        #                 print("-"*20)
        #             best_res = copy.copy(ret)
        #     except Exception as e:
        #         print(f"Failed to optimize : {str(e)}")
        #         pass
        pass
    else:
        # n* if no optimization
        if not h_list:
            raise ValueError("h_list is empty")
        save_npy(f"{path}/H", [h for h in h_list])
