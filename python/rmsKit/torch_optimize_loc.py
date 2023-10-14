import torch
from torch import Tensor
from lattice import KH, FF
from lattice import save_npy, list_unitaries

import argparse
from random import randint
import random
import numpy as np
import rms_torch
import logging
import datetime


models = [
    "KH",
    "HXYZ",
    "HXYZ2D",
    "FF1D",
    "FF2D",
]
loss_val = ["mes", "none"]  # minimum energy solver, quasi energy solver

parser = argparse.ArgumentParser(description="exact diagonalization of shastry_surtherland")
parser.add_argument("-m", "--model", help="model (model) Name", required=True, choices=models)
parser.add_argument("-Jz", "--coupling_z", help="coupling constant (Jz)", type=float, default=1)  # SxSx + SySy +
parser.add_argument("-Jx", "--coupling_x", help="coupling constant (Jx)", type=float)
parser.add_argument("-Jy", "--coupling_y", help="coupling constant (Jy)", type=float)
parser.add_argument("-hx", "--mag_x", help="magnetic field", type=float, default=0)
parser.add_argument("-hz", "--mag_z", help="magnetic field", type=float, default=0)
parser.add_argument("-T", "--temperature", help="temperature", type=float)
parser.add_argument("-M", "--num_iter", help="# of iterations", type=int, default=10)
parser.add_argument("-r", "--seed", help="random seed", type=int, default=None)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.01)
parser.add_argument("-schedule", help="Use scheduler if given", action="store_true")
parser.add_argument("-f_path", help="Path to fine tuning unitaries", type=str, default="")
parser.add_argument("-e", "--epoch", help="epoch", type=int, default=100)
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
    choices=loss_val,
    nargs="?",
    const="all",
    required=True,
)

parser.add_argument(
    "-o",
    "--optimizer",
    help="optimizer",
    choices=["LION", "Adam"],
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


def lr_lambda(epoch: int) -> float:
    f = lambda x: np.exp(-4.5 * np.tanh(x * 0.02))
    epoch = (epoch // 10) * 10
    return f(epoch)



args = parser.parse_args()

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
args_str = "args: {}".format(args)
log_filename = f"optimizer_output/{now}_optimizer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename,
    # handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
print(f"logging to file: {log_filename}")
logging.info(args_str)


device = torch.device("cuda") if args.platform == "gpu" else torch.device("cpu")
logging.info("device: {}".format(device))

logging.info("args: {}".format(args))
M = args.num_iter
seed_list = [randint(0, 1000000) for i in range(M)]

if __name__ == "__main__":
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

    H = None
    P = None

    if "KH" in args.model:
        h_list, sps = KH.local(ua, p)
    elif "HXYZ" in args.model:
        pass
    elif "FF" in args.model:
        if args.model == "FF1D":
            d = 1
        elif args.model == "FF2D":
            d = 2
        p = dict(
                sps = 3,
                rank = 2,
                length = 6,
                dimension = d,
                seed = 0,
                )
        h_list, sps = FF.local(ua, p)
        # params_str = f"{d}D_{p["sps"]}s_{p["rank"]}r_{p["length"]}" 
        params_str = f's_{p["sps"]}_r_{p["rank"]}_l_{p["length"]}_seed_{p["seed"]}'

    path = f"array/torch/{args.model}_loc/{ua}/{args.loss}/{params_str}"
    logging.info(f"operators ***will* be saved to {path}")

    # decide the loss function
    if args.loss == "none":
        h_list = [-np.array(h) for h in h_list]
        save_npy(f"{path}/H", h_list)
        exit(0)
    elif args.loss == "mes":
        loss = rms_torch.MinimumEnergyLoss(h_list, device=device)

