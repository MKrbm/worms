import numpy as np
import datetime
import logging
import sys
from .functions import sum_ham, stoquastic, get_files_in_order, files_to_dataframe, \
    get_local_best_loss_from_tensorboard, bfs_search_and_get_files, filter_dataframe, \
    get_seed_and_loss, get_canonical_form
from .run_sim import run_ff, extract_loss, path_with_lowest_loss, run_worm


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_logger(log_filename: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logging.info("Log file will be saved to " + log_filename)
    logging.info("filename : torch_optimize_loc.py")
    return logger


__all__ = [
    "run_ff",
    "sum_ham",
    "stoquastic",
    "get_local_best_loss_from_tensorboard",
    "extract_loss",
    "path_with_lowest_loss",
    "run_worm",
    "get_files_in_order",
    "bfs_search_and_get_files",
    "filter_dataframe",
    "files_to_dataframe",
    "get_seed_and_loss",
    "get_canonical_form",
    "get_version",
    "get_numpy_version",
    "now",
    "get_logger",
]
