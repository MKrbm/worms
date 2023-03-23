import torch
from torch import Tensor
from lattice import KH
from lattice import save_npy

import argparse
from random import randint
import numpy as np
import rms
import subprocess
import math
import rms_torch
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
print(f"logging to file: {log_filename}")


models = [
    "KH",
    "HXYZ",
]
loss_val = ["sel", "sqel"]  # minimum energy solver, quasi energy solver

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
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.01)
parser.add_argument("-schedule", help = "Use scheduler if given", action = "store_true")
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
device = torch.device("cuda") if args.platform == "gpu" else torch.device("cpu")
u0 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/0.npy")
u1 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/1.npy")
u2 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/2.npy")
u3 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/3.npy")
u_list = [u0, u1, u2, u3]
u0 = np.load("array/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_1/u/0.npy")


#d* set seed
seed = args.seed if args.seed is not None else randint(0, 1000000)


# logging.info(f"seed: {seed}")
# torch.manual_seed(seed)
# np.random.seed(seed)

if __name__ == "__main__":
    logging.info("args: {}".format(args))
    M = args.num_iter
    seed_list = [args.seed if args.seed else randint(0, 1000000) for i in range(M)]
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
    path = f"array/torch/{args.model}/{ua}/{args.loss}/{params_str}"

    H = None
    if args.model == "KH":  
        if ua == "3site":
            H = KH.system([2, 2], ua, p)
        else:
            raise ValueError("not implemented")

    if args.loss == "sel" and H is not None: #* system energy loss
        loss_ = rms_torch.SystemEnergyLoss(H, device=device)
    elif args.loss == "sqel" and H is not None:
        loss_ = rms_torch.SystemQuasiEnergyLoss(H, N = 3, device=device)
        logging.info("Pre-calculated ground state and energy")
    else :
        raise ValueError("not implemented")
    model_ = rms_torch.UnitaryRieman(H.shape[0], 8, device=device).to(device)
    model = torch.compile(model_, dynamic = False, fullgraph=True)
    loss = torch.compile(loss_, dynamic = False, fullgraph=True)

    best_loss = 1e10
    best_us = None
    for i, seed in enumerate(seed_list):
        logging.info(f"iteration: {i+1}/{M}, seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        local_best_loss = 1e10
        local_best_us = []
        # model.reset_params() if i > 0 else None

        model.reset_params()
        optimizer = rms_torch.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
        def lr_lambda(epoch : int) -> float:
            lr = args.learning_rate
            if epoch < 5:
                return lr
            elif epoch < 10:
                return 0.8 * lr
            elif epoch < 15:
                return 0.5 * lr
            elif epoch < 20:
                return 0.1 * lr
            elif epoch < 30:
                return 0.07 * lr
            elif epoch < 80:
                return 0.05 * lr
            else:
                return 0.01 * lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda) if args.schedule else None
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) if args.schedule else None
        epochs = args.epoch
        for t in range(epochs):
            optimizer.zero_grad()
            output = model()
            loss_val = loss(output, 100)
            if loss_val.item() < local_best_loss:
                with torch.no_grad():
                    local_best_loss = loss_val.item()
                    local_best_us = [p.data.detach().cpu().numpy() for p in model.parameters()]
            loss_val.backward()
            for p in model.parameters():
                grad = p.grad  # Get the gradient from the compiled model
                if grad is not None:
                    grad.data[:] = rms_torch.riemannian_grad_torch(p.data, grad)
                else:
                    raise RuntimeError("No gradient for parameter")
            optimizer.step()
            scheduler.step() if scheduler is not None else None
            logging.info(f"Epoch: {t+1}/{epochs}, Loss: {loss_val.item()}")

        logging.info(f"best local loss: {local_best_loss} quasiEnergy = {loss(model())}", )
        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_us = [np.copy(u) for u in local_best_us]
    

    logging.info("loss value: %s", best_loss)
    save_npy(f"{path}/M_{M}_e_{args.epoch}_lr_{args.learning_rate}/u", best_us)
