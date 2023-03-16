from jax_lattice import KH
from jax_lattice import save_npy
import argparse
from random import randint
import numpy as np
import rms
import os
import subprocess
import jax
import jax.numpy as jnp
import math

models = [
    "KH",
    "HXYZ",
]
loss = ["none", "mes", "qes"]  # minimum energy solver, quasi energy solver

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

# parser.add_argument('-N','--n_samples', help='# of samples', type = int, default = 50)
# parser.add_argument('-m','--n_Krylov', help='dimernsion of Krylov space', type = int, default = 50)
args = parser.parse_args()
# jax.config.update("jax_platform_name", args.platform)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if args.platform == "gpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if __name__ == "__main__":
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
                print("Eigenvector is not found. Run solver_jax.py with command")
                print(" ".join(command))
                out = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = out.communicate()

        if args.loss != "none" and "3site" not in ua:
            raise ValueError("optimizer is supported only for 3site unitary algorithm")

    path = f"array/{args.model}/{ua}/{args.loss}/{params_str}"
    seed = randint(0, 100000)
    print("seed: ", seed)
    ur = rms.unitary.UnitaryRiemanGenerator(8, jax.random.PRNGKey(seed), np.float64)
    if args.loss == "mes" and h_list:

        mesLoss_list = [
            rms.loss.MinimumEnergy(jnp.array(h), 8, np.float64) for h in h_list
        ]
        mesLoss = rms.loss.MeanMultiLoss(mesLoss_list)
        lion_solver = rms.solver.lionSolver(mesLoss)
        momentum_solver = rms.solver.momentumSolver(mesLoss)
        cg_solver = rms.solver.cgSolver(mesLoss)
        best_lv = 1e10
        best_u = None

        def scheduler(step):
            r = step / 10
            return 1 / math.sqrt(1 + int(r)) * 0.01

        for _ in range(M):
            u = ur.reset_matrix()
            # u = U
            u, lv = lion_solver(
                u, 5000, scheduler, cout=True, cutoff_cnt=100, mass1=0.9, mass2=0.98
            )
            u, lv = cg_solver(u, 500, 0.001, cutoff_cnt=50, cout=True, mass=0.1)
            # u, lv = momentum_solver(u, 1000, 0.1, 0.3, cout=True, cutoff_cnt=10)
            # u, lv = cg_solver(u, 500, 0.001, 0.1, cutoff_cnt=10, cout=True)
            if lv < best_lv:
                best_lv = lv
                best_u = u
        print("loss value: ", best_lv)
        save_npy(f"{path}/M_{M}/u", [np.array(best_u)])

    if args.loss == "qes" and h_list:
        if groundstate_path:
            x = np.load(groundstate_path)
        else:
            raise RuntimeError("groundstate is not found")
        x0 = x.reshape([8] * 4)
        x0 = x0.transpose([0, 2, 1, 3]).reshape(-1)
        x0 = jnp.array(x0)

        x1 = x
        x1 = jnp.array(x0)

        x2 = x.reshape([8] * 4)
        x2 = x2.transpose([0, 3, 1, 2]).reshape(-1)
        x2 = jnp.array(x2)
        x_list = [x0, x1, x2]

        qesLoss_list = [
            rms.loss.QuasiEnergy(jnp.array(h), _x, 8, np.float64)
            for h, _x in zip(h_list, x_list)
        ]
        qesLoss = rms.loss.MeanMultiLoss(qesLoss_list)
        lion_solver = rms.solver.lionSolver(qesLoss)
        momentum_solver = rms.solver.momentumSolver(qesLoss)
        cg_solver = rms.solver.cgSolver(qesLoss)
        best_lv = 1e10
        best_u = None

        def scheduler(step):
            r = step / 10
            return 1 / math.sqrt(1 + int(r)) * 0.01

        def scheduler2(step):
            r = step / 10
            return 1 / (1 + int(r)) * 0.1

        for _ in range(M):
            u = ur.reset_matrix()
            # u = jnp.eye(8, dtype=np.float64)
            # u = U
            u, lv = lion_solver(
                u,
                5000,
                scheduler,
                cout=True,
                cutoff_cnt=100,
                mass1=0.9,
                mass2=0.6,
                offset=0.001,
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
                best_u = u
        print("loss value: ", best_lv)
        save_npy(f"{path}/M_{M}/u", [np.array(best_u)])
