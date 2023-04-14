from lattice import KH, HXYZ
from lattice import save_npy
import argparse
from random import randint
import numpy as np
import rms
import subprocess
import jax
import jax.numpy as jnp
import math

import logging
import os
import datetime


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"optimizer_output/{now}_optimizer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename,
    # handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)


models = [
    "KH",
    "HXYZ",
]
loss = ["none", "mes", "qes", "smel", "sel"]  # minimum energy solver, quasi energy solver

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
parser.add_argument("-hx", "--mag_x", help="magnetic field", type=float, default=0)
parser.add_argument("-hz", "--mag_z", help="magnetic field", type=float, default=0)
parser.add_argument("-T", "--temperature", help="temperature", type=float)
parser.add_argument("-M", "--num_iter", help="# of iterations", type=int, default=10)
parser.add_argument("-r", "--seed", help="random seed", type=int, default=None)
parser.add_argument(
    "-u",
    "--unitary_algorithm",
    help="algorithm determine local unitary matrix",
    default="original",
)
parser.add_argument(
    "-loss",
    "--loss",
    help="loss_methods",
    choices=loss,
    nargs="?",
    const="all",
)

parser.add_argument(
    "-p",
    "--platform",
    help="cput / gpu",
    choices=["cpu", "gpu"],
    nargs="?",
    const="all",
    default="cpu",
)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if args.platform == "gpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# u0 = np.load("array/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_1/u/0.npy")
if __name__ == "__main__":
    logging.info("args: {}".format(args))
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
    folder = f"array/{args.model}/{ua}/{args.loss}/{params_str}"
    h_list = []
    sps = 2
    x = None
    groundstate_path = None
    if args.model == "KH":
        h_list, sps = KH.local(ua, p)
        model_name = "KH" + f"_2x2"
        if args.loss == "qes":
            groundstate_path = f"out/{model_name}/{ua}/{params_str}/groundstate.npy"
            if not os.path.exists(groundstate_path):
                command = [
                    "python",
                    "solver_jax.py",
                    "-m",
                    "KH",
                    "-u",
                    ua,
                    "-L1",
                    "2",
                    "-L2",
                    "2",
                    "-gs",
                ]
                for k, v in p.items():
                    command += [f"-{k}", str(v)]
                logging.info("Eigenvector is not found. Run solver_jax.py with command")
                logging.info(" ".join(command))
                out = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = out.communicate()

        if args.loss != "none" and "3site" not in ua:
            raise ValueError("optimizer is supported only for 3site unitary algorithm")
    elif args.model == "HXYZ":
        h_list, sps = HXYZ.local(ua, p)
    def scheduler(lr):
        def wrapper(step):
            r = step / 10
            return 1 / math.sqrt(1 + r) * lr

        return wrapper

    path = f"array/{args.model}/{ua}/{args.loss}/{params_str}"
    seed = randint(0, 100000) if args.seed is None else args.seed
    np.random.seed(seed)
    logging.info("seed: %s", seed)
    ur = rms.unitary.UnitaryRiemanGenerator(8, jax.random.PRNGKey(seed), np.float64)
    best_lv = 1e10
    best_u = None
    if args.loss == "none":
        h_list = [-np.array(h) for h in h_list]
        save_npy(f"{path}/H", h_list)

    elif args.loss == "mes" and h_list:

        state_list = [rms.loss.init_loss(jnp.array(h), 8, np.float64, "mes") for h in h_list]
        mesLoss = rms.loss.mes_multi
        lion_solver = rms.solver.lionSolver(mesLoss, state_list)
        momentum_solver = rms.solver.momentumSolver(mesLoss, state_list)
        cg_solver = rms.solver.cgSolver(mesLoss, state_list)
        logging.info("D           : %s", momentum_solver.D)
        logging.info("upper_bound : %s", momentum_solver.upper_bound)

        for _ in range(M):
            u = ur.reset_matrix()
            # u = U
            u, lv = lion_solver(
                u, 5000, scheduler(0.01), cout=True, cutoff_cnt=100, mass1=0.9, mass2=0.98
            )
            # u, lv = cg_solver(u, 500, 0.001, cutoff_cnt=50, cout=True, mass=0.1)
            # u, lv = momentum_solver(u, 1000, 0.1, 0.3, cout=True, cutoff_cnt=10)
            # u, lv = cg_solver(u, 500, 0.001, 0.1, cutoff_cnt=10, cout=True)
            if lv < best_lv:
                best_lv = lv
                best_u = (u).copy()
    elif args.loss == "qes" and h_list:
        if groundstate_path:
            x = np.load(groundstate_path)
        else:
            raise RuntimeError("groundstate is not found")
        x0 = x.reshape([8] * 4)
        x0 = x0.transpose([0, 2, 1, 3]).reshape(-1)
        x0 = jnp.array(x0)
        x1 = jnp.array(x)
        x2 = x.reshape([8] * 4)
        x2 = x2.transpose([0, 3, 1, 2]).reshape(-1)
        x2 = jnp.array(x2)
        x_list = [x0, x1, x2]

        state_list = [
            rms.loss.init_loss(jnp.array(_h), 8, np.float64, "qes", X=jnp.array(_x))
            for _h, _x in zip(h_list, x_list)
        ]
        qesLoss = rms.loss.qes_multi
        print(qesLoss(state_list, jnp.array(u0)))
        lion_solver = rms.solver.lionSolver(qesLoss, state_list)
        momentum_solver = rms.solver.momentumSolver(qesLoss, state_list)
        cg_solver = rms.solver.cgSolver(qesLoss, state_list)
        best_lv = 1e10
        best_u = None

        logging.info("D           : %s", momentum_solver.D)
        logging.info("upper_bound : %s", momentum_solver.upper_bound)

        # def scheduler(step):
        #     r = step / 10
        #     return 1 / math.sqrt(1 + r) * 0.01

        for _ in range(M):
            u = ur.reset_matrix()
            u = jnp.array(u0)
            
            # u = jnp.eye(8, dtype=np.float64)
            # u = U
            mass1 = np.random.lognormal(np.log(0.6), 0.5)
            mass2 = np.random.lognormal(np.log(0.5), 0.5)
            lr = np.random.lognormal(np.log(0.001), 2)
            # print(f"mass1: {mass1}, mass2: {mass2}, lr: {lr}")
            logging.info(f"mass1: {mass1}, mass2: {mass2}, lr: {lr}")
            u, lv = lion_solver(
                u,
                5000,
                scheduler(lr),
                cout=True,
                cutoff_cnt=100,
                mass1=0.9,
                mass2=0.6,
                offset=0.0001,
            )
            # u, lv = momentum_solver(
            #     u, 1000, scheduler2, mass=0.5, cout=True, cutoff_cnt=10
            # )
            # u, lv = cg_solver(
            #     u, 500, 0.001, cutoff_cnt=10, cout=True, mass=0.1, offset=0.01
            # )
            # u, lv = cg_solver(u, 500, 0.001, 0.1, cutoff_cnt=10, cout=True)
            if lv < best_lv:
                best_lv = lv
                best_u = (u).copy()
    elif args.loss == "smel" and h_list:
        H = jnp.array(KH.system([2, 2], "3site", p))
        state = rms.loss.init_loss(H, 8, np.float64, "smel")
        state_list = [state]
        qesLoss = rms.loss.system_mel_multi
        lion_solver = rms.solver.lionSolver(qesLoss, state_list)
        momentum_solver = rms.solver.momentumSolver(qesLoss, state_list)
        # cg_solver = rms.solver.cgSolver(qesLoss, state_list)
        best_lv = 1e10
        best_u = None

        logging.info("D           : %s", momentum_solver.D)
        logging.info("upper_bound : %s", momentum_solver.upper_bound)

        # def scheduler(step):
        #     r = step / 10
        #     return 1 / math.sqrt(1 + r) * 0.01

        for _ in range(M):
            u = ur.reset_matrix()
            
            # u = jnp.eye(8, dtype=np.float64)
            # u = U
            mass1 = np.random.lognormal(np.log(0.6), 0.5)
            mass2 = np.random.lognormal(np.log(0.5), 0.5)
            lr = np.random.lognormal(np.log(0.0001), 2)
            logging.info("mass1: %s, mass2: %s, lr: %s", mass1, mass2, lr)
            u, lv = momentum_solver(
                u,
                1000,
                scheduler(lr),
                cout=True,
                cutoff_cnt=100,
                mass=mass1,
                offset=0.01,
            )
            if lv < best_lv:
                best_lv = lv
                best_u = (u).copy()

    elif args.loss == "sel" and h_list:
        H = jnp.array(KH.system([2, 2], "3site", p))
        state = rms.loss.init_loss(H, 8, np.float64, "sel", beta=1.0)
        state_list = [state]
        qesLoss = rms.loss.system_el_multi
        lion_solver = rms.solver.lionSolver(qesLoss, state_list)
        momentum_solver = rms.solver.momentumSolver(qesLoss, state_list)
        # cg_solver = rms.solver.cgSolver(qesLoss, state_list)
        best_lv = 1e10
        best_u = None

        logging.info("D           : %s", momentum_solver.D)
        logging.info("upper_bound : %s", momentum_solver.upper_bound)

        for _ in range(M):
            u = ur.reset_matrix()
            mass1 = np.random.lognormal(np.log(0.3), 0.5)
            mass2 = np.random.lognormal(np.log(0.5), 0.5)
            lr = np.random.lognormal(np.log(0.01), 2)
            logging.info("iter : %s mass1: %s, mass2: %s, lr: %s", _, mass1, mass2, lr)

            # u = jnp.eye(8, dtype=np.float64)
            # u = U
            u, lv = momentum_solver(
                u,
                1000,
                scheduler(lr),
                cout=True,
                cutoff_cnt=10,
                mass=mass1,
                offset=0.01,
            )
            if lv < best_lv:
                best_lv = lv
                best_u = (u).copy()
            logging.info("loss value: %s", best_lv)
            
    else:
        raise RuntimeError("loss function is not found")
    logging.info("loss value: %s", best_lv)
    save_npy(f"{path}/M_{M}/u", [np.array(best_u)])
