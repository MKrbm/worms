import os
import re
import shutil
import torch
from lattice import KH, FF
from lattice import save_npy
from utils import get_local_best_loss_from_tensorboard
import time

import argparse
from random import randint


# for tensorboard
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import rms_torch
import logging
import datetime
import platform

models = [
    "KH",
    "HXYZ",
    "HXYZ2D",
    "FF1D",
    "FF2D",
]
# minimum energy solver, quasi energy solver
loss_val = ["mel", "none", "Adam"]

parser = argparse.ArgumentParser(
    description="exact diagonalization of shastry_surtherland")
parser.add_argument("-m", "--model", help="model (model) Name",
                    required=True, choices=models)
parser.add_argument("-Jz", "--coupling_z", help="coupling constant (Jz)",
                    type=float, default=1)  # SxSx + SySy +
parser.add_argument("-Jx", "--coupling_x",
                    help="coupling constant (Jx)", type=float)
parser.add_argument("-Jy", "--coupling_y",
                    help="coupling constant (Jy)", type=float)
parser.add_argument("-hx", "--mag_x", help="magnetic field",
                    type=float, default=0)
parser.add_argument("-hz", "--mag_z", help="magnetic field",
                    type=float, default=0)
parser.add_argument("-T", "--temperature", help="temperature", type=float)
parser.add_argument("-M", "--num_iter",
                    help="# of iterations", type=int, default=10)
parser.add_argument("-r", "--seed", help="random seed", type=int, default=None)
parser.add_argument("--sps", help="sps", type=int, default=3)
parser.add_argument("-lr", "--learning_rate",
                    help="learning rate", type=float, default=0.01)
parser.add_argument(
    "-schedule", help="Use scheduler if given", action="store_true")
parser.add_argument(
    "--print", help="print out the result", action="store_true")
parser.add_argument(
    "-f_path", help="Path to fine tuning unitaries", type=str, default="")
parser.add_argument("-e", "--epoch", help="epoch", type=int, default=100)
parser.add_argument(
    "-lt",
    "--lattice_type",
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
    default="LION",
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

parser.add_argument(
    "-n",
    "--num_threads",
    help="number of threads",
    type=int,
    default=-1,
)

def list_arrays(path):
    array_files = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith('.npy')]:
            # print(re.search(r'u\/\d\.npy', os.path.join(dirpath, filename)))
            # matches both / and \ separators
            if re.search(r'u\/\d\.npy', os.path.join(dirpath, filename)):
                array_files.append(os.path.join(dirpath, filename))

    return array_files


# specified_path = "array/torch/FF1D_loc/s_3_r_2_us_1_d_1_seed_11/original_mel_LION/lr_0.005_epoch_100_M_10000/"
specified_path = ""
arrays_path = list_arrays(specified_path)
arrays_path.sort()
# for arr in arrays_path:
#     print("load array: ", arr)
arrays = [np.load(arr) for arr in arrays_path]
# print(arrays[0])


def lr_lambda(epoch: int) -> float:
    def f(x): return np.exp(-4.5 * np.tanh(x * 0.02))
    epoch = (epoch // 10) * 10
    return f(epoch)


args = parser.parse_args()
p = dict(
    Jx=args.coupling_x if args.coupling_x is not None else args.coupling_z,
    Jy=args.coupling_y if args.coupling_y is not None else args.coupling_z,
    Jz=args.coupling_z,
    hx=args.mag_x,
    hz=args.mag_z,
)

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
logging.info("filename : torch_optimize_loc.py")
logging.info(args_str)

# if machine is osx then use mps backend instead of cuda
if args.platform == "gpu":
    # check if os is osx
    if platform.system() == "Darwin":
        device = torch.device("mps")
    else:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")
    if args.num_threads == -1:
        args.num_threads = torch.get_num_threads()
    torch.set_num_threads(args.num_threads)

# device = torch.device("") if args.platform == "gpu" else torch.device("cpu")
logging.info("device: {}".format(device))

num_threads = torch.get_num_threads()
print("Number of CPU threads used: {}".format(num_threads))
logging.info("args: {}".format(args))
M = args.num_iter if not arrays else len(arrays)
seed = args.seed
# `-r` is for random seed determine randomness of local hamiltonian instead of optimization
seed_list = [randint(0, 1000000) for i in range(M)]


if __name__ == "__main__":

    a = ""
    for k, v in p.items():
        v = float(v)
        a += f"{k}_{v:.4g}_"
    params_str = a[:-1]
    lt = args.lattice_type

    # Get the current number of threads used by PyTorch
    num_threads = torch.get_num_threads()

    # print(f'Number of CPU threads used: {num_threads}')
    logging.info(f"Number of CPU threads used: {num_threads}")
    H = None
    P = None

    custom_dir = "out/tensorboard"  # Adjust this to your desired directory
    loss_name = f"{lt}_{args.loss}_{args.optimizer}"
    setting_name = f"lr_{args.learning_rate}_epoch_{args.epoch}"

    if "KH" in args.model:
        h_list, sps = KH.local(lt, p)
        model_name = f"{args.model}_loc/Jz_{p['Jz']}_Jx_{p['Jx']}_Jy_{p['Jy']}_hx_{p['hx']}_hz_{p['hz']}"
    elif "HXYZ" in args.model:
        model_name = f"{args.model}_loc/Jz_{p['Jz']}_Jx_{p['Jx']}_Jy_{p['Jy']}_hx_{p['hx']}_hz_{p['hz']}"
        pass
    elif "FF" in args.model:
        if args.model == "FF1D":
            d = 1
        elif args.model == "FF2D":
            d = 2
        p = dict(
            sps=args.sps,
            rank=2,
            dimension=d,
            # lattice type (number of sites in unit cell)
            lt=1 if lt == "original" else int(lt),
            seed=1 if seed is None else seed,
        )
        h_list, sps = FF.local(p)
        params_str = f's_{sps}_r_{p["rank"]}_lt_{p["lt"]}_d_{p["dimension"]}_seed_{p["seed"]}'
        model_name = f"{args.model}_loc/{params_str}"

    base_name = f"{model_name}/{loss_name}/{setting_name}"
    path = f"array/torch/{model_name}/{loss_name}/{setting_name}"
    ham_path = f"array/torch/{model_name}/{loss_name}/{setting_name}/H.npy"

    save_npy(f"{path}/H", [-np.array(h) for h in h_list])
    print(f"logging to tensorboard under: {custom_dir}/{base_name}")
    logging.info("tensorboard under : {custom_dir}/{base_name}")
    logging.info(f"unitary will be saved to {path}")
    logging.info(f"hamiltonian saved to {ham_path}")

    if args.loss == "none":
        exit()
    if args.loss == "mel":
        loss = rms_torch.MinimumEnergyLoss(h_list, device=device)

    optimizer_func: torch.optim.Optimizer = None
    if args.optimizer == "LION":
        optimizer_func = rms_torch.LION
        learning_params = dict(
            lr=args.learning_rate,
        )
    elif args.optimizer == "Adam":
        optimizer_func = rms_torch.Adam
        learning_params = dict(
            lr=args.learning_rate,
            betas=(0.3, 0.5),
        )
    else:
        raise ValueError("not implemented")
    model = rms_torch.UnitaryRieman(
        h_list[0].shape[1], sps, device=device).to(device)

    best_loss = 1e10
    best_us = None
    good_seeds = []
    unitaries = []

    model.reset_params(torch.eye(sps))
    logging.info("initial loss: %s", loss(model()).item())

    if arrays:
        print("Fine tuning unitaries given")
    # if f_path is given, initialize unitaries with the given unitaries and number of iteration is the number of unitaries in f_path
    for i, seed in enumerate(seed_list):
        now = time.time()
        # now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tb_name = f"{custom_dir}/{base_name}/{seed}"
        writer = SummaryWriter(tb_name)
        if args.loss == "smel":
            loss.initializer(model())
        logging.info(f"iteration: {i+1}/{M}, seed: {seed}")
        # original_dir_name = f"{custom_dir}/{base_name}/seed_{seed}_{now}"
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.loss == "smel":
            loss.initializer(model())
        local_best_loss = 1e10
        local_best_us = []
        model.reset_params() if not arrays else model.reset_params(
            torch.from_numpy(arrays[i]))

        optimizer = optimizer_func(
            model.parameters(), **learning_params)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda) if args.schedule else None
        epochs = args.epoch

        loss_list = []
        for t in range(epochs):
            optimizer.zero_grad()
            output = model()
            loss_val = loss(output)
            loss_val_item = loss_val.item()
            if loss_val_item < local_best_loss:
                with torch.no_grad():
                    local_best_loss = loss_val_item
                    local_best_us = [p.data.detach().cpu().numpy()
                                     for p in model.parameters()]
            loss_val.backward()
            for p in model.parameters():
                grad = p.grad  # Get the gradient from the compiled model
                if grad is not None:
                    grad.data[:] = rms_torch.riemannian_grad_torch(
                        p.data, grad)
                else:
                    raise RuntimeError("No gradient for parameter")
            if args.print:
                logging.info(f"Epoch: {t+1}/{epochs}, Loss: {loss_val.item()}")
                print(f"Epoch: {t+1}/{epochs}, Loss: {loss_val.item()}")
            optimizer.step()
            scheduler.step() if scheduler is not None else None
            loss_list.append(loss_val_item)
            writer.add_scalar('Loss', loss_val_item, t)

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_us = [np.copy(u) for u in local_best_us]
        good_seeds.append([seed, local_best_loss])

        time_elapsed = time.time() - now
        logging.info(
            f"best loss at epoch {epochs}: {local_best_loss}, best loss so far: {best_loss} time elapsed: {time_elapsed:.4f} seconds"
        )
        save_npy(f"{path}/loss_{local_best_loss:.5f}/u", local_best_us)
        writer.close()

    logging.info("best loss value: %s", best_loss)
