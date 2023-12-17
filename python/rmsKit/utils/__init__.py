"""This module contains utility functions for the rmsKit package.

The functions are divided into the following categories:
    - Functions for running the simulation
    - Functions for extracting information from the simulation result
    - Some other utility functions
"""
import datetime
from .functions import sum_ham, stoquastic, get_files_in_order, files_to_dataframe, \
    get_local_best_loss_from_tensorboard, bfs_search_and_get_files, filter_dataframe, \
    get_seed_and_loss, get_canonical_form, extract_info_from_file, result_to_dataframe, \
    extract_parameters_from_path, param_dict_normalize, get_loss

from .run_sim import run_ff, extract_loss, path_with_lowest_loss, run_worm
from .get_logger import get_logger
from . import path


NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# __all__ = [
#     "run_ff",
#     "sum_ham",
#     "stoquastic",
#     "get_local_best_loss_from_tensorboard",
#     "extract_loss",
#     "path_with_lowest_loss",
#     "run_worm",
#     "get_files_in_order",
#     "bfs_search_and_get_files",
#     "filter_dataframe",
#     "files_to_dataframe",
#     "get_seed_and_loss",
#     "get_canonical_form",
#     "now",
#     "get_logger",
#     "extract_info_from_file",
#     "result_to_dataframe",
#     "extract_parameters_from_path",
# ]
