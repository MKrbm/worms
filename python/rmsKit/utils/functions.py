from typing import Dict
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os
import glob
from collections import deque
import re
import logging

logger = logging.getLogger(__name__)


def sum_ham(h: np.ndarray, bonds: List, L: int, sps: int, stoquastic_=False, out=None):
    local_size = h.shape[0]
    h = np.kron(h, np.eye(int((sps**L) / h.shape[0])))
    ori_shape = h.shape
    H = np.zeros_like(h)
    h = h.reshape(L * 2 * (sps,))
    for bond in bonds:
        if sps ** len(bond) != local_size:
            raise ValueError("local size is not consistent with bond")
        trans = np.arange(L)
        for i, b in enumerate(bond):
            trans[b] = i
        l_tmp = len(bond)
        for i in range(L):
            if i not in bond:
                trans[i] = l_tmp
                l_tmp += 1
        trans = np.concatenate([trans, trans + L])
        H += h.transpose(trans).reshape(ori_shape)
    if stoquastic_:
        return stoquastic(H)
    return H


def stoquastic(X: np.ndarray) -> np.ndarray:
    a = np.eye(X.shape[0]) * np.max(np.diag(X))
    return -np.abs(X - a) + a


def get_local_best_loss_from_tensorboard(tensorboard_dir):
    """Get a list of local best loss values and their corresponding directory names."""
    dir_names = glob.glob(f"{tensorboard_dir}/*")
    loss_values_and_dir_names = []
    for dir_name in dir_names:
        # Assuming directory name format contains "local_best_loss_{value}"
        try:
            loss_value = float(dir_name.split("loss_")[-1])
            loss_values_and_dir_names.append(loss_value)
        except BaseException:
            pass  # directory name might not match expected format
    return sorted(loss_values_and_dir_names)


def get_files_in_order(folder_path):
    # Get all text files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Function to convert file names to datetime for sorting
    def file_to_datetime(filename):
        try:
            split_filename = re.split(r'[_\.]', filename)
            return datetime.strptime("_".join(split_filename[:2]), "%Y%m%d_%H%M%S")
        except Exception as e:
            print(f"Warning: {filename} does not match the expected format.")
            print(e)
            return None

    # Sort files based on datetime
    sorted_files = sorted(files, key=file_to_datetime, reverse=True)

    # Return the full relative paths to the sorted files
    return [os.path.join(folder_path, f) for f in sorted_files]


# Column names
TOTAL_SWEEPS = "sweeps"
U_PATH = "u_path"
NUMBER_OF_SITES = "n_sites"
TEMPERATURE = "T"
ALPHA = "alpha"
ENERGY_P_SITE_MEAN = "e"
ENERGY_P_SITE_ERROR = "e_error"
AVERAGE_SIGN_MEAN = "as"
AVERAGE_SIGN_ERROR = "as_error"
MODEL_NAME = "model_name"
HAM_PATH = "ham_path"
SPECIFIC_HEAT_P_SITE_MEAN = "c"
SPECIFIC_HEA_P_SITET_ERROR = "c_error"
MAGNETIZATION_MEAN = "m"
MAGNETIZATION_ERROR = "m_error"
SUSCEPTIBILITY_MEAN = "chi"
SUSCEPTIBILITY_ERROR = "chi_error"
BOUNDARY_CONDITION = "bc"


def extract_info_from_file(file_path, allow_missing: bool = False,
                           warning: bool = True) -> Tuple[Dict, None]:
    # Dictionary to store extracted info
    info_dict = {}

    with open(file_path, 'r') as file:
        content = file.readlines()
        for line in content:
            if "temperature" in line:
                info_dict[TEMPERATURE] = float(line.split(":")[1].strip())
            elif "hamiltonian is read from" in line:
                info_dict[HAM_PATH] = line.split("from")[1].strip()
            elif "unitary" in line:
                if "unitary matrix is read from" in line:
                    info_dict[U_PATH] = line.split("from")[1].strip()
                else:
                    info_dict[U_PATH] = ""
            elif "sweeps(in total)" in line:
                info_dict[TOTAL_SWEEPS] = int(
                    line.split(":")[1].replace(",", "").strip())
            elif "number of sites" in line:
                info_dict[NUMBER_OF_SITES] = int(line.split(":")[1].strip())
            elif "alpha" in line:
                info_dict[ALPHA] = float(line.split(":")[1].strip())
            elif "Energy per site" in line:
                values = line.split("=")[1].split("+-")
                info_dict[ENERGY_P_SITE_MEAN] = float(values[0].strip())
                info_dict[ENERGY_P_SITE_ERROR] = float(values[1].strip())
            elif "Average sign" in line:
                values = line.split("=")[1].split("+-")
                info_dict[AVERAGE_SIGN_MEAN] = float(values[0].strip())
                info_dict[AVERAGE_SIGN_ERROR] = float(values[1].strip())
            elif "Specific heat" in line:
                values = line.split("=")[1].split("+-")
                info_dict[SPECIFIC_HEAT_P_SITE_MEAN] = float(values[0].strip())
                info_dict[SPECIFIC_HEA_P_SITET_ERROR] = float(
                    values[1].strip())
            elif "magnetization" in line:
                values = line.split("=")[1].split("+-")
                info_dict[MAGNETIZATION_MEAN] = float(values[0].strip())
                info_dict[MAGNETIZATION_ERROR] = float(values[1].strip())
            elif "susceptibility" in line:
                values = line.split("=")[1].split("+-")
                info_dict[SUSCEPTIBILITY_MEAN] = float(values[0].strip())
                info_dict[SUSCEPTIBILITY_ERROR] = float(values[1].strip())
            elif "model name is" in line:
                info_dict[MODEL_NAME] = line.split(":")[1].strip()
            elif "boundary condition" in line:
                info_dict[BOUNDARY_CONDITION] = line.split(":")[1].strip()
                if info_dict[BOUNDARY_CONDITION] != "pbc" and info_dict[BOUNDARY_CONDITION] != "obc":
                    raise ValueError("boundary condition should be either periodic or open")

    if U_PATH not in info_dict.keys() or HAM_PATH not in info_dict.keys():
        print(
            f"Info: {file_path} is missing necessary information. Ignored this file") if warning else None
        return None

    if ENERGY_P_SITE_ERROR not in info_dict.keys():
        print(
            f"Info: {file_path} has not yet finished. Ignored this file") if warning else None
        return None

    if len(info_dict.keys()) != 18 and not allow_missing:
        print(
            f"Warning: {file_path} is missing some information.") if warning else None
        return None

    return info_dict


def filter_dataframe(df, sweeps=None, u_path=None, n_sites=None, ham_path=None):

    # Conditions
    condition1 = df[TOTAL_SWEEPS] == sweeps if sweeps is not None else True
    condition2 = ((u_path) ^ df[U_PATH].isna())if u_path is not None else True
    condition3 = df[NUMBER_OF_SITES] == n_sites if n_sites is not None else True
    condition4 = df[HAM_PATH].str.contains(
        ham_path) if ham_path is not None else True

    # Filtering
    filtered_df = df[condition1 & condition2 & condition3 & condition4]

    return filtered_df


def bfs_search_and_get_files(base_folder):
    # Initialize a queue for BFS
    queue = deque([base_folder])
    all_files = []

    while queue:
        current_dir = queue.popleft()

        # Check if the current directory exists
        if not os.path.isdir(current_dir):
            continue

        # Use get_files_in_order to get sorted files in the current directory
        all_files.extend(get_files_in_order(current_dir))

        # Add all sub-directories to the queue
        for sub_dir in os.listdir(current_dir):
            full_sub_dir = os.path.join(current_dir, sub_dir)
            if os.path.isdir(full_sub_dir):
                queue.append(full_sub_dir)

    if len(all_files) == 0:
        # print(f"No files found in {base_folder}")
        raise RuntimeError("No files found""")

    return all_files


def files_to_dataframe(file_list, **kwargs):
    # Define a list to store the data
    data = []

    # Read each file and extract the necessary information
    for file_path in file_list:
        data_dict = extract_info_from_file(
            file_path, **kwargs)
        data.append(data_dict) if data_dict is not None else None

    # Convert the data list to a DataFrame
    df = pd.DataFrame(data)

    return df


def get_loss(df):
    """Get the loss from the file path."""
    df['loss'] = df['u_path'].str.extract(
        r'loss_([+|-]?[0-9]*\.[0-9]+|[0-9]+\.[0-9]*$)').astype(float)
    return df


def result_to_dataframe(base_folder, allow_missing=True, warning=False):
    """Get the result from the base folder."""
    df = files_to_dataframe(bfs_search_and_get_files(
        base_folder), allow_missing=allow_missing, warning=warning)
    return df


def get_seed_and_loss(file_path):
    """Get the seed and loss from the file path.

    Use get_loss to get the loss from the file path.
    """
    df = files_to_dataframe(bfs_search_and_get_files(
        file_path), allow_missing=True, warning=False)
#     df = df[df.model_name.str.contains("FF_1D")]
    df['seed'] = df['ham_path'].str.extract(r'seed_(\d+)').astype(int)
    return get_loss(df)


def extract_parameters_from_path(path: str) -> Dict[str, float]:
    """This function extracts the parameters and their values from the path.

    Here suppose path includes the parameters in the following format:
    {parent}/{model_name}_{par1}_{par1_value}_{par2}_{par2_value} /{setting_name}
    or
    {parent}/{par1}_{par1_value}_{par2}_{par2_value} /{setting_name}
    """
    # Split the path and get the relevant part with parameters
    parts = path.split('/')
    params_part = parts[-4]  # The third from the last part contains the parameters
    params_split = params_part.split('_')

    params = dict()
    if len(params_split) % 2 != 0:
        """
        When model data is included
        """
        params["modelname"] = params_split[0]
        params_split = params_split[1:]
        for i in range(0, len(params_split), 2):
            try:
                params[params_split[i]] = float(params_split[i + 1])
            except ValueError as e:
                logging.error("""
                The parameter value is not a float: {}
                path : {}
                Error : {}
                return None
                """.format(params_split[i + 1], path, e))
                return None

    else:
        for i in range(0, len(params_split), 2):
            try:
                params[params_split[i]] = float(params_split[i + 1])
            except ValueError as e:
                logging.error("""
                The parameter value is not a float: {}
                path : {}
                Error : {}
                return None
                """.format(params_split[i + 1], path, e))
                return None

    return params


def param_dict_normalize(param_dicts: List[Dict[str, float]]) -> pd.DataFrame:
    """Assuming 'param_dicts' is a list of dictionaries returned from extract_parameters_from_path.

    I found this function is almost doing the same thing as pd.json_normalize
    """
    params_df = pd.json_normalize(param_dicts)
    if params_df.isnull().sum().iloc[1:].sum():
        for col in params_df.columns:
            if params_df[col].isnull().sum():
                logger.warning(f"Warning: {col} has null values")
        raise ValueError("Parameters are missing")
    return params_df


def get_canonical_form(A):

    check_validity = True
    if A.ndim != 3:
        raise ValueError("A must be a 3-rank tensor")
    if A.shape[0] != A.shape[2]:
        raise ValueError(
            "middle index should represent physical index and the side indices should be virtual indices")

    s = A.shape[0]
    A = A.transpose(1, 0, 2)
    A_tilde = np.einsum("ijk,ilm->jlkm", A, A)
    A_tilde = A_tilde.reshape(s**2, s**2)
    e, V = np.linalg.eigh(A_tilde)
    rho = e[-1]
    A_tilde = A_tilde / rho

    e, V = np.linalg.eigh(A_tilde)
    x = V[:, -1].reshape(s, s)

    e, U = np.linalg.eigh(x)
    x_h = U @ np.diag(np.sqrt(e + 0j)) @ U.T
    x_h_inv = U @ np.diag(1/np.sqrt(e + 0j)) @ U.T

    B = x_h_inv @ A @ x_h / np.sqrt(rho)  # canonical form
    B = B.transpose(1, 0, 2)

    if check_validity:
        check_cano = np.einsum("jik, lik->jl", B, B)
        if np.linalg.norm(np.eye(check_cano.shape[0]) - check_cano) > 1E-8:
            raise ValueError("B is not a canonical")
        B_ = B.transpose(1, 0, 2)
        B_tilde = np.einsum("ijk,ilm->jlkm", B_, B_).reshape(4, 4)
        Eb = np.sort(np.linalg.eigvals(B_tilde))
        Ea = np.sort(np.linalg.eigvals(A_tilde))
        if np.linalg.norm(Ea.real - Eb.real) > 1E-8:
            raise ValueError("B is not a canonical")
    return B
