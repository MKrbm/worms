import torch
import numpy as np
import rms_torch
import time
import platform
import logging
from pathlib import Path
import os

from lattice import save_npy, get_model
from utils.parser import get_parser, get_params_parser
from utils import now, get_logger

parser = get_parser()

# add symbolic link to generated hamiltonian and unitary
parser.add_argument(
    "--symoblic_link",
    type=Path,  # Changed to Path for direct pathlib support
    default=None,
    help="The symbolic link to the directory the generated hamiltonian and unitary are stored.",
)
args, params, hash_str = get_params_parser(parser)

if __name__ == "__main__":

    sps = args.sps
    epochs = args.epochs
    seed_list = [np.random.randint(0, 1000000) for i in range(args.num_iter)]

    log_dir = Path("optimizer_output")
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    log_filename = log_dir / f"{now}_{hash_str}.log"
    logger = get_logger(log_filename.resolve().as_posix(), level=logging.INFO, stdout=args.stdout)

    if args.platform == "gpu":
        if platform.system() == "Darwin":
            device = torch.device("mps")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.info("Please specify the number of threads with OMP_NUM_THREADS and MKL_NUM_THREADS")

    logging.info(f"device = {device}")

    h_list, sps, model_name = get_model(args.model, params)

    custom_dir = Path("out/tensorboard")
    custom_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    loss_name = f"{params['lt']}_{args.loss}_{args.optimizer}"
    setting_name = f"lr_{args.learning_rate}_epoch_{epochs}"
    base_name = Path(model_name) / loss_name / setting_name
    h_path = Path("array/torch") / model_name
    h_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    u_path = h_path / loss_name / setting_name
    u_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    logging.info(f"Unitary will be saved to {u_path.resolve()}")
    logging.info(f"Hamiltonian saved to {h_path}/H/")
    save_npy(h_path / "H", [-np.array(h) for h in h_list])  # minus for - beta * H

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

    initial_loss = loss(model()).item()
    logging.info(f"initial loss = {initial_loss}")

    best_loss = 1e10
    best_us = None
    num_print = 10
    for i, seed in enumerate(seed_list):
        start = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.loss == "smel":
            loss.initializer(model())
        local_best_loss = 1e10
        local_best_us = []
        model.reset_params()
        optimizer = optimizer_func(model.parameters(), **learning_params)

        for t in range(epochs):
            optimizer.zero_grad()
            output = model()
            loss_val = loss(output)
            loss_val_item = loss_val.item()
            if loss_val_item < local_best_loss:
                with torch.no_grad():
                    local_best_loss = loss_val_item
                    local_best_us = [p.data.detach().cpu().numpy() for p in model.parameters()]
            loss_val.backward()
            for p in model.parameters():
                grad = p.grad
                if grad is not None:
                    grad.data[:] = rms_torch.riemannian_grad_torch(p.data, grad)
                else:
                    raise RuntimeError("No gradient for parameter")
                if (t+1) % (epochs // num_print) == 0 or t == 0:
                    logging.info(
                        f"I: {i + 1}/{len(seed_list)} : Epoch: {t+1}/{epochs}, Loss: {loss_val.item()}")
            optimizer.step()

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_us = [np.copy(u) for u in local_best_us]

        time_elapsed = time.time() - start
        logging.info(
            f"best loss at epoch {epochs}: {local_best_loss}, best loss so far: {best_loss}, time elapsed: {time_elapsed:.4f} seconds"
        )
        u_path_epoch = u_path / f"loss_{local_best_loss:.5f}/u"
        u_path_epoch.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        save_npy(u_path_epoch, local_best_us)

    logging.info(f"best loss: {best_loss} / initial loss: {initial_loss}")
    logging.info(f"best loss was saved to {u_path}/loss_{best_loss:.5f}/u")
    logging.info(f"hamiltonian was saved to {h_path}/H")

    if args.symoblic_link is not None:
        symb_path = args.symoblic_link
        logging.info(f"Create symbolic link to {h_path.resolve()}")
        logging.info(f"Link is {symb_path.resolve()}")
        if symb_path.exists():
            logging.info(f"Remove existing link {symb_path}")
            symb_path.unlink()
        os.symlink(h_path.resolve(), symb_path)
