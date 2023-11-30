import torch
import numpy as np
import rms_torch
import time
import platform
import logging
from typing import List, Tuple, Any, Dict
from numpy._typing import NDArray

import lattice
from lattice import save_npy
from utils.parser import args, params,  hash_str
from utils import now, get_logger


def get_model_ham(model: str, params: Dict[str, str]) -> Tuple[List[NDArray[Any]], int, str]:
    lt = params["lt"]
    sps = params["sps"]
    seed = params["seed"]
    if "FF" in model:
        if model == "FF1D":
            d = 1
        elif model == "FF2D":
            d = 2
        p = dict(
            sps=sps,
            rank=2,
            dimension=d,
            # lattice type (number of sites in unit cell)
            lt=1 if lt == "original" else int(lt),
            seed=1 if seed is None else seed,
        )
        h_list, sps = lattice.FF.local(p)
        # h_list: List[NDArray[Any]] = [h.numpy() for h in _h_list]
        params_str = f's_{sps}_r_{p["rank"]}_lt_{p["lt"]}_d_{p["dimension"]}_seed_{p["seed"]}'
        model_name = f"{model}_loc/{params_str}"

    return h_list, sps, model_name


if __name__ == "__main__":

    sps = args.sps
    epochs = args.epochs
    seed_list = [np.random.randint(0, 1000000) for i in range(args.num_iter)]

    # logging.basicConfig(
    #     stream=sys.stdout,
    #     level=logging.DEBUG,  # This can be adjusted to the desired level.
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )

    log_filename = f"optimizer_output/{now}_{hash_str}.log"
    logger = get_logger(log_filename, level=logging.INFO, stdout=args.stdout)

    if args.platform == "gpu":
        # check if os is osx
        if platform.system() == "Darwin":
            device = torch.device("mps")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        # print("Info : please specify the number of threads with OMP_NUM_THREADS and MKL_NUM_THREADS")
        logging.info(
            "Please specify the number of threads with OMP_NUM_THREADS and MKL_NUM_THREADS")

    # print(f"Info : device = {device}")
    logging.info(f"device = {device}")

    h_list, sps, model_name = get_model_ham(args.model, params)

    custom_dir = "out/tensorboard"  # Adjust this to your desired directory
    loss_name = f"{params['lt']}_{args.loss}_{args.optimizer}"
    setting_name = f"lr_{args.learning_rate}_epoch_{epochs}"
    base_name = f"{model_name}/{loss_name}/{setting_name}"
    path = f"array/torch/{model_name}/{loss_name}/{setting_name}"
    ham_path = f"array/torch/{model_name}/{loss_name}/{setting_name}/H.npy"

    # print(f"Info : Unitary will be saved to {path}")
    # print(f"Info : Hamiltonian saved to {ham_path}")
    logging.info(f"Unitary will be saved to {path}")
    logging.info(f"Hamiltonian saved to {ham_path}")
    save_npy(f"{path}/H", [-np.array(h) for h in h_list])

    loss = rms_torch.MinimumEnergyLoss(h_list, device=device)
    optimizer_func: type[torch.optim.Optimizer] = rms_torch.LION
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
        raise ValueError("Invalid optimizer")

    model = rms_torch.UnitaryRieman(
        h_list[0].shape[1], sps, device=device).to(device)
    model.reset_params(torch.eye(sps))

    loss_val = loss(model())
    # print(f"Info : initial loss = {loss_val.item()}")
    logging.info(f"initial loss = {loss_val.item()}")

    best_loss = 1e10
    best_us = None
    num_print = 10
    for i, seed in enumerate(seed_list):
        start = time.time()
        tb_name = f"{custom_dir}/{base_name}/{seed}"
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.loss == "smel":
            loss.initializer(model())
        local_best_loss = 1e10
        local_best_us = []
        model.reset_params()
        optimizer = optimizer_func(
            model.parameters(), **learning_params)

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
                grad = p.grad
                if grad is not None:
                    grad.data[:] = rms_torch.riemannian_grad_torch(
                        p.data, grad)
                else:
                    raise RuntimeError("No gradient for parameter")
                if (t+1) % (epochs // num_print) == 0 or t == 0:
                    logging.info(
                        f"I: {i + 1}/{len(seed_list)} : Epoch: {t+1}/{epochs}, Loss: {loss_val.item()}")
            optimizer.step()
            loss_list.append(loss_val_item)

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_us = [np.copy(u) for u in local_best_us]

        time_elapsed = time.time() - start
        logging.info(
            f"""
            best loss at epoch {epochs}: {local_best_loss},
            best loss so far: {best_loss} time elapsed: {time_elapsed:.4f} seconds
            """
        )
        save_npy(f"{path}/loss_{local_best_loss:.5f}/u", local_best_us)
