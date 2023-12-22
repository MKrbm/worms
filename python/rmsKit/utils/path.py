"""Utilities for path manipulation."""
import re
from pathlib import Path
import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from .functions import get_loss

logger = logging.getLogger(__name__)


def extract_to_rmskit(path: Path):
    """Extract the path to rmsKit directory from the given path."""
    for parent in path.parents:
        if parent.name == 'rmsKit':
            return str(parent)
    raise ValueError("The given path is not under 'rmsKit'.")


def find_info_txt_files(directory_path):
    """Find all info.txt files under the given directory."""
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise ValueError(f"The given path {directory_path} is not a directory.")
    info_txt_files = list(dir_path.rglob('info.txt'))
    return [str(file) for file in info_txt_files]


def find_summary_files(directory_path: Path) -> List[Dict[str, Path]]:
    """Find all info.txt files under the given directory."""
    dir_path = Path(directory_path).resolve()
    if not dir_path.is_dir():
        raise ValueError(f"The given path {directory_path} is not a directory.")
    sim_res_folder = [path.parent for path in dir_path.rglob('summary/')]
    res = []
    for sim_res in sim_res_folder:
        summary_folder = sim_res / "summary"
        info_file = sim_res / "info.txt"
        assert summary_folder.is_dir()
        assert info_file.exists()
        for path in summary_folder.rglob('*.csv'):
            dic = dict()
            dic["summary"] = path
            dic["info"] = info_file
            res.append(dic)
    return res


def get_df_from_summary_files(summary_files: str, N: int) -> pd.DataFrame:
    """Get a dataframe from the given summary files.

    summary files must be csv files.
    """
    df = None
    for res_dict in summary_files:
        sum_file = res_dict["summary"]
        info_file = res_dict["info"]
        info = extract_info_from_txt(info_file)
        if "sweeps_{}".format(N) in str(sum_file):
            try:
                df1 = pd.read_csv(sum_file)
                if not (df1.u_path.isnull().values.any()):
                    df1 = get_loss(df1)
                    if np.abs(df1.loss.values - info["best_loss"]).mean() > 1E-5:
                        print(df1.loss, info["best_loss"])
                        raise ValueError("best_loss is not consistent with summary data")
            except BaseException as e:
                logging.warning("Could not concate or read {}".format(sum_file))
                logging.warning(e)

            df1["init_loss"] = info["initial_loss"]

            if df is not None:
                df = pd.concat([df, df1])
            else:
                df = df1
    df = df.reset_index(drop=True)
    return df


def get_sim_result(directory_path: Path, N: int) -> pd.DataFrame:
    """Get the simulation result from the given directory."""
    summary_files = find_summary_files(directory_path)
    df = get_df_from_summary_files(summary_files, N)
    return df


def extract_info_from_txt(file_path: Path) -> Dict[str, Path]:
    """Extract information from the given info.txt file."""
    best_loss_pattern = r"best loss: ([\d.e-]+)"
    initial_loss_pattern = r"initial loss: ([\d.e-]+)"
    hamiltonian_path_pattern = r"hamiltonian was saved to ([/\w._-]+)"
    unitary_path_pattern = r"best loss was saved to ([/\w._-]+)"

    # Read the file and extract information
    with open(file_path, 'r') as file:
        content = file.read()

        # Extract best loss
        best_loss_match = re.search(best_loss_pattern, content)
        best_loss = best_loss_match.group(1) if best_loss_match else None

        # Extract initial loss
        initial_loss_match = re.search(initial_loss_pattern, content)
        initial_loss = initial_loss_match.group(1) if initial_loss_match else None

        # Extract hamiltonian path
        hamiltonian_path_match = re.search(hamiltonian_path_pattern, content)
        hamiltonian_path = hamiltonian_path_match.group(1) if hamiltonian_path_match else None

        # Extract unitary path
        unitary_path_match = re.search(unitary_path_pattern, content)
        unitary_path = unitary_path_match.group(1) if unitary_path_match else None

    res = {
        "best_loss": float(best_loss),
        "initial_loss": float(initial_loss),
        "hamiltonian_path": Path(hamiltonian_path),
        "unitary_path": Path(unitary_path)
    }

    if None in res.values():
        raise ValueError("""
        Could not extract all required information from the given file.
        The file should contain the following information:
        - best loss
        - initial loss
        - hamiltonian path
        - unitary path
        The file content is:
        {}
        """.format(content))

    return res


def get_worm_path(search_path: Path, return_info_path: bool = False):
    """Extract the path to model hamiltonian and optimized unitary from the given path."""
    if not search_path.exists():
        raise ValueError("The given search path does not exist.")
    if not search_path.is_dir():
        raise ValueError("The given path is not a directory.")

    # Get info.txt files under the hamiltonian path
    info_txt_files = find_info_txt_files(search_path)
    if len(info_txt_files) == 0:
        raise ValueError("No info.txt file found under the search path : {}".format(search_path))
    if len(info_txt_files) > 1:
        logger.warning("""
                       Multiple info.txt files found under the search path.
                       The {} will be used.
            """.format(info_txt_files[0]))
    info_txt_file = Path(info_txt_files[0])
    extracted_info = extract_info_from_txt(info_txt_file)

    # Get the path to the optimized unitary
    unitary_path = extracted_info["unitary_path"]
    if not Path(unitary_path).exists():
        raise ValueError("The path to the optimized unitary does not exist.")
    hamiltonian_path = extracted_info["hamiltonian_path"]
    if not Path(hamiltonian_path).exists():
        raise ValueError("The path to the model hamiltonian does not exist.")

    loss = extracted_info["best_loss"]
    initial_loss = extracted_info["initial_loss"]

    if return_info_path:
        return loss, initial_loss, unitary_path, hamiltonian_path, info_txt_file
    else:
        return loss, initial_loss, unitary_path, hamiltonian_path
