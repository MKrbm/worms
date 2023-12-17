"""Utilities for path manipulation."""
import re
from pathlib import Path
import logging
from typing import Dict

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


def get_worm_path(search_path: Path, get_unitary: bool = True):
    """Extract the path to model hamiltonian and optimized unitary from the given path."""
    if not search_path.exists():
        raise ValueError("The given search path does not exist.")
    if not search_path.is_dir():
        raise ValueError("The given path is not a directory.")

    # Find the path to the model hamiltonian
    directories_named_H = list(search_path.rglob('H/'))
    if len(directories_named_H) == 0:
        raise ValueError("No directory for hamiltonian found under the given path.")
    if len(directories_named_H) > 1:
        logger.warning("""
                       Multiple directories named 'H' found under the given path.
                       The {} will be used.
            """.format(directories_named_H[0]))
    hamiltonian_path = directories_named_H[0]

    if not get_unitary:
        return hamiltonian_path

    # Get info.txt files under the hamiltonian path
    info_txt_files = find_info_txt_files(search_path)
    if len(info_txt_files) == 0:
        raise ValueError("No info.txt file found under the search path : {}".format(search_path))
    if len(info_txt_files) > 1:
        logger.warning("""
                       Multiple info.txt files found under the search path.
                       The {} will be used.
            """.format(info_txt_files[0]))
    info_txt_file = info_txt_files[0]
    extracted_info = extract_info_from_txt(info_txt_file)

    # Get the path to the optimized unitary
    unitary_path = extracted_info["unitary_path"]
    if not Path(unitary_path).exists():
        raise ValueError("The path to the optimized unitary does not exist.")
    hamiltonian_path_ = extracted_info["hamiltonian_path"]
    if not Path(hamiltonian_path_).exists():
        raise ValueError("The path to the model hamiltonian does not exist.")

    if hamiltonian_path_.resolve().as_posix() != hamiltonian_path.resolve().as_posix():
        logger.warning(
            """ The path to the model hamiltonian extracted from the info.txt file : \n{} 
            does not match the path to the model hamiltonian found under the given path : \n{}
            The path extracted from the info.txt file will be used.
            """.format(
                hamiltonian_path_,
                hamiltonian_path))
        hamiltonian_path = hamiltonian_path_

    loss = extracted_info["best_loss"]
    initial_loss = extracted_info["initial_loss"]

    return loss, initial_loss, unitary_path, hamiltonian_path
