import numpy as np
import os
import re
import logging
from typing import Any, List, Union


def proj_symm(x):
    s = int(np.sqrt(x.shape[0]))
    x = x.reshape(s, s, s, s)
    return ((x + np.einsum("ijkl->jilk", x))/2).reshape(s*s, s*s)


def save_npy(folder: str, hams: List[np._typing.NDArray[Any]]) -> None:
    if isinstance(hams, list):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, ham in enumerate(hams):
            name = f"{i}"
            np.save(folder + f"/{i}", (ham.real).astype(np.float64))
            log = f"save matrix ({ham.shape}): " + folder + "/" + name + ".npy"
            logging.info(log)
    else:
        if not os.path.exists(os.path.dirname(folder)):
            os.makedirs(os.path.dirname(folder))
        name = "0"
        np.save(folder, hams.real.astype(np.float64))
        log = f"save matrix ({hams.shape}): " + folder + ".npy"
        logging.info(log)


def list_unitaries(path, n_percent=None, n_top=None, thres=None):
    loss_folders = [
        entry.path for entry in os.scandir(path) if entry.is_dir() and entry.name.startswith("loss_")
    ]

    def get_folder_number(folder_name):
        match = re.search(r"loss_(\d+\.\d+)", folder_name)
        if match:
            return float(match.group(1))
        return float("inf")

    loss_folders.sort(key=get_folder_number)
    selected_folders = [loss_folders for _ in range(3)]

    if n_percent is not None:
        if isinstance(n_percent, int) and 0 <= n_percent <= 100:
            num_folders = int(len(loss_folders) * n_percent / 100)
            selected_folders[0] = loss_folders[:num_folders]
        else:
            raise ValueError(
                "Invalid n_percent value. Please enter an integer value between 0 and 100.")

    if n_top is not None:
        if isinstance(n_top, int) and 0 <= n_top <= len(loss_folders):
            selected_folders[1] = loss_folders[:n_top]
        else:
            raise ValueError(
                "Invalid n_top value. Please enter an integer value between 0 and the total number of folders.")

    if thres is not None:
        if isinstance(thres, (int, float)):
            selected_folders[2] = [
                folder for folder in loss_folders if get_folder_number(folder) < thres]
        else:
            raise ValueError(
                "Invalid threshold value. Please enter a valid number.")

    # 3つのリストのうち要素数が最も少ないものを選択
    result = min(selected_folders, key=len)
    return result
