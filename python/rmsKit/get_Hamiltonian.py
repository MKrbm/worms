import torch
from torch import Tensor
from lattice import KH
from lattice import save_npy, list_unitaries

import argparse
from random import randint
import random
import numpy as np
import rms_torch
import logging
import datetime
import os




models = [
    "KH",
    "HXYZ",
]

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
<<<<<<< HEAD:python/rmsKit/torch_optimize.py
parser.add_argument("-T", "--temperature", help="temperature", type=float)
parser.add_argument("-M", "--num_iter", help="# of iterations", type=int, default=10)
parser.add_argument("-s", "--seed", help="random seed", type=int, default=None)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.01)
parser.add_argument("-schedule", help = "Use scheduler if given", action = "store_true")
parser.add_argument("-e", "--epoch", help="epoch", type=int, default=100)
parser.add_argument(
    "-ua",
    "--unitary_algorithm",
    help="algorithm determine local unitary matrix",
    default="original",
)
parser.add_argument(
    "-l",
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
=======
parser.add_argument("-f_path", help = "Path to fine tuning unitaries", type=str, default="")
>>>>>>> origin/MKrbm/issue17:python/rmsKit/get_Hamiltonian.py

    
args = parser.parse_args()
<<<<<<< HEAD:python/rmsKit/torch_optimize.py
device = torch.device("cuda") if args.platform == "gpu" else torch.device("cpu")
# u0 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/0.npy")
# u1 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/1.npy")
# u2 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/2.npy")
# u3 = np.load("array/torch/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_200/u/3.npy")
# u_list = [u0, u1, u2, u3]
# u0 = np.load("array/KH/3site/sel/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_1/u/0.npy")
=======

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
args_str = "args: {}".format(args)
log_filename = f"optimizer_output/{now}_{args_str}_optimizer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename,
    # handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
print(f"logging to file: {log_filename}")
>>>>>>> origin/MKrbm/issue17:python/rmsKit/get_Hamiltonian.py





logging.info("args: {}".format(args))


#d* set fine tuning unitaries
f_path = args.f_path + "/u"
loss_folders = [
    entry.path for entry in os.scandir(f_path) if entry.name.endswith(".npy")
]


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
    path = f"array/torch/{args.model}/{ua}/{args.loss}/{params_str}"

    H = None
    if args.model == "KH":  
        if ua == "3site":
            H = KH.system([2, 2], ua, p)
        else:
            raise ValueError("not implemented")

    if args.loss == "sel" and H is not None: #* system energy loss
        loss = rms_torch.SystemEnergyLoss(H, device=device)
    elif args.loss == "sqel" and H is not None:
<<<<<<< HEAD:python/rmsKit/torch_optimize.py
        loss = rms_torch.SystemQuasiEnergyLoss(H, N = 3, device=device)
=======
        loss = rms_torch.SystemQuasiEnergyLoss(H, N = 5, device=device)
>>>>>>> origin/MKrbm/issue17:python/rmsKit/get_Hamiltonian.py
        logging.info("Pre-calculated ground state and energy")
    elif args.loss == "smel" and H is not None:
        loss = rms_torch.SystemMinimumEnergyLoss(H, device=device).to(device)
    else :
        raise ValueError("not implemented")
    model_ = rms_torch.UnitaryRieman(H.shape[0], 8, device=device).to(device)
    model = torch.compile(model_, dynamic = False, fullgraph=True)
<<<<<<< HEAD:python/rmsKit/torch_optimize.py
    # loss = torch.compile(loss_, dynamic = False, fullgraph=True)
=======
>>>>>>> origin/MKrbm/issue17:python/rmsKit/get_Hamiltonian.py

    best_loss = 1e10
    best_us = None
    good_seeds = []
    unitaries = []
    for i, seed in enumerate(seed_list):
        logging.info(f"iteration: {i+1}/{M}, seed: {seed}")
        if ft_unitaries:
            logging.info("fine tuning unitaries are used")
        torch.manual_seed(seed)
        np.random.seed(seed)
        local_best_loss = 1e10
        local_best_us = []
        model.reset_params(ft_unitaries[i]) if ft_unitaries else model.reset_params()

        if args.optimizer == "LION":
            optimizer = rms_torch.LION(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == "Adam":
            optimizer = rms_torch.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
        else:
            raise ValueError("not implemented")
        

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda) if args.schedule else None
        epochs = args.epoch
        if args.loss == "smel":
            loss.initializer(model())
        for t in range(epochs):
            optimizer.zero_grad()
            output = model()
            loss_val = loss(output, 100) if args.loss == "sqel" else loss(output)
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

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_us = [np.copy(u) for u in local_best_us]
        good_seeds.append([seed, local_best_loss])
        logging.info(f"best local loss: {local_best_loss} quasiEnergy = {loss(model())}", )
        save_npy(f"{path}/e_{args.epoch}_lr_{args.learning_rate}/loss_{local_best_loss:5f}/u", local_best_us)
    

    logging.info("loss value: %s", best_loss)
    logging.info("good seeds: %s", good_seeds)
