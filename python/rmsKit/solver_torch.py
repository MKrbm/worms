import numpy as np
import torch
import os
from lattice import save_npy, get_model
from utils.parser import get_parser, get_params_parser
from utils import NOW, get_logger
import logging


parser = get_parser(length=True)
args, params, hash_str = get_params_parser(parser)


log_filename = f"optimizer_output/{NOW}_{hash_str}.log"
logger = get_logger(log_filename, level=logging.INFO, stdout=args.stdout)

if __name__ == "__main__":
    device = torch.device(
        "cuda") if args.platform == "gpu" else torch.device("cpu")
    logging.info("device: {}".format(device))

    L_list = []
    L_list.append(args.length1)
    L_list.append(args.length2) if args.length2 is not None else None
    N = np.prod(L_list)
    if args.model == "SS2D":
        N = 4*N
    elif args.model == "KH2D":
        N = 3*N

    Hnp, sps, path = get_model(args.model, params, L=L_list)
    if "1D" in args.model:
        path += f"_obc_{int(args.obc)}"
    elif "2D" in args.model:
        pass
    print(path)
    H = torch.from_numpy(Hnp).to(device)
    E, V = torch.linalg.eigh(H)
    E = E.cpu().numpy()
    V = V.cpu().numpy()

    logging.info(f"finish diagonalization of {path}")
    logging.info(f"output to {path}")

    path = "out/" + path
    file = f"{path}/groundstate"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    save_npy(file, V[:, 0])
    file = f"{path}/groundstate.csv"
    os.makedirs(os.path.dirname(file), exist_ok=True)

    logging.info("groundstate file: {}".format(file))
    with open(file, "w") as dat_file:
        dat_file.write("index,value\n")
        for i, v in enumerate(V[:, 0]):
            dat_file.write(f"{i},{v:.60g}\n")

    beta = np.linspace(0, 10, 1001).reshape(1, -1)
    B = np.exp(-beta * E[:, None])
    Z = B.sum(axis=0)
    E_mean = (E[:, None] * B).sum(axis=0) / Z
    E_square_mean = ((E * E)[:, None] * B).sum(axis=0) / Z
    beta = beta.reshape(-1)
    C = (E_square_mean - E_mean**2) * (beta**2)

    # * save calculated data
    file = f"{path}/eigenvalues"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    save_npy(file, E)

    file = f"{path}/eigenvalues.csv"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    logging.info("eigenvalues file: {}".format(file))
    with open(file, "w") as dat_file:
        dat_file.write("index,value\n")
        # logging.info("index,value") if args.stdout else None
        for i, e in enumerate(E):
            dat_file.write(f"{i},{e:.60g}\n")
            # logging.info(f"{i},{e:.5g}") if args.stdout else None

    file = f"{path}/statistics.csv"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    logging.info("stat file: {}".format(file))
    with open(file, "w") as dat_file:
        dat_file.write("beta,energy_per_site,specific_heat\n")
        for (
            b,
            e,
            c,
        ) in zip(beta, E_mean, C):
            dat_file.write(f"{b},{e/N},{c/N}\n")

    # print(f"output to {path}")
