import subprocess
import os
import glob
import re
from datetime import datetime
import contextlib
from typing import Union, List, Tuple
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_loss(dir_name):
    # Use regex to extract the loss value from the directory name
    match = re.search(r"loss_(\d+\.\d+)", dir_name)
    if match:
        return float(match.group(1))
    return None


def path_with_lowest_loss(parent_dir, return_ham=False, absolute_path=False, only_ham=False):
    # Get all directory names under the parent directory using glob (not only direct children)
    parent_dir = os.path.abspath(parent_dir)
    dir_names = glob.glob(os.path.join(parent_dir, "**"), recursive=True)
    dir_names = [path for path in dir_names if "loss" in path]

    if only_ham:
        # find the file path include "/H/" in the dir_names
        ham_path = [path for path in dir_names if "/H" in path][0]
        if ham_path is None:
            raise ValueError("No hamiltonian file found.")
        elif len(ham_path) > 1:
            print("Warning: Multiple hamiltonian files found : ", ham_path)

        if absolute_path:
            # get absolute path to parent_dir
            ham_path = os.path.abspath(ham_path)
            return ham_path

        return ham_path

    # Extract loss values and associate them with their paths
    losses = [(path, extract_loss(path)) for path in dir_names]
    # Filter out any paths where loss couldn't be extracted
    valid_losses = [(path, loss) for path, loss in losses if loss is not None]

    # Find the path with the lowest loss
    min_path, min_loss = min(valid_losses, key=lambda x: x[1])
    min_path = min_path + "/u"
    print("min_path: ", min_path)

    if return_ham:
        # find the file path include "/H/" in the dir_names
        ham_path = [path for path in dir_names if path.endswith("/H")][0]
        print(ham_path)

        logger.info("ham_path: {}".format(ham_path))

        if ham_path is None:
            raise ValueError("No hamiltonian file found.")
        elif len(ham_path) > 1:
            print("Warning: Multiple hamiltonian files found : ", ham_path)

        if absolute_path:
            # get absolute path to parent_dir
            ham_path = os.path.abspath(ham_path)
            min_path = os.path.abspath(min_path)
            return min_path, min_loss, ham_path

        return min_path, min_loss, ham_path

    if absolute_path:
        min_path = os.path.abspath(min_path)
        return min_path, min_loss

    return min_path, min_loss


def run_ff(
    seed: int,
    rank: int,
    sps: int,
    lt: int,
    dimension: int = 1,
    M: int = 100,
    e: int = 1000,
    gpu: bool = False,
    o: str = "LION",
    lr: float = 0.01,
    output_dir: str = "out/run_sim/",
    output: bool = False,
) -> Union[subprocess.CompletedProcess, str, None]:
    # Set your parameters
    params = {
        "sps": sps,
        "rank": rank,
        "dimension": dimension,  # Currently, only 1D is available
        "lt": lt,
        "seed": seed,
        "M": M,
        "e": e,
        "o": o,
        "lr": lr,
    }

    # check if executable exists
    if not os.path.isfile("torch_optimize_loc.py"):
        print("torch_optimize_loc.py not found. Please check the path.")
        return None
    if dimension not in [1, 2]:
        raise ValueError("Dimension must be 1 or 2.")

    # Construct the command to run the optimization script
    command = f"python torch_optimize_loc.py -m FF{dimension}D -loss mel -o LION -e {params['e']} -M {params['M']} -lr {params['lr']} -r {params['seed']}" \
        + f" -lt {params['lt']} -p {'gpu' if gpu else 'cpu'} "

    command += f"-o {o}"

    # Construct a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the output file path
    output_filename = (
        f"output_seed_{params['seed']}_rank_{params['rank']}_sps_{params['sps']}_lt_{params['lt']}_{timestamp}.txt"
    )
    output_filepath = os.path.join(output_dir, output_filename)

    if output:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Run the command and write to the output file
        with open(output_filepath, "w") as output_file:
            process = subprocess.run(
                command, shell=True, stdout=output_file, stderr=subprocess.STDOUT, text=True)

        # Provide feedback
        if process.returncode == 0:
            print("Optimization successfully.")
        else:
            print("Optimization failed.")
            print(process.stderr)
        print(
            f"""Command: {command}
                Output: {output_filepath}\n
                settings = {params} \n """
        )
    else:
        # Run the command and capture the output
        process = subprocess.run(
            command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Optimization successfully.")
        else:
            print("Optimization failed.")
            print(process.stderr)

        print(
            f"""Command: {command}
                settings = {params} \n """
        )
    return process


@contextlib.contextmanager
def change_directory(directory):
    original_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_directory)


def find_executable(grandparent_dir: str) -> Tuple[str, str]:
    pattern = os.path.join(grandparent_dir, "**", "Release")
    release_dirs = glob.glob(pattern, recursive=True)

    for dir in release_dirs:
        executable_path = os.path.join(dir, "main_MPI")
        if os.path.isfile(executable_path):
            return dir, "main_MPI"

    return "", ""


def run_worm(
        model_name: str,
        ham_path: Path,
        u_path: Union[Path, None],
        L: List[int],
        T: float,
        N: int,
        n: int = 1,
        obc: bool = False,
        project_dir: Union[Path, None] = None,
        logging: bool = True):

    if not isinstance(ham_path, Path):
        ham_path = Path(ham_path)
        if not ham_path.is_dir():
            raise ValueError("ham_path :{} must be a existing directory.".format(ham_path))
        else:
            if ham_path.name != "H":
                raise ValueError(
                    "ham_path : {} must be a directory with name 'H'.".format(ham_path))
    if u_path and not isinstance(u_path, Path):
        u_path = Path(u_path)
        if not u_path.is_dir():
            raise ValueError("u_path : {} must be a existing directory.".format(u_path))
        else:
            if u_path.name != "u":
                raise ValueError("u_path : {} must be a directory with name 'u'.".format(u_path))
    if project_dir is not None and not isinstance(project_dir, Path):
        project_dir = Path(project_dir)

    # 1. Get the current directory
    if project_dir is not None:
        if not project_dir.is_dir():
            raise ValueError("project_dir : {} must be a existing directory.".format(project_dir))
        release_dir = project_dir / "build"
    else:
        release_dir = Path(os.getcwd()) / "build"

    if not release_dir.is_dir():
        raise ValueError("release_dir : {} must be a existing directory.".format(release_dir))

    if not isinstance(L, list):
        raise ValueError("L must be a list of integers.")

    # 2. Find the executable
    executable_name = "./main_MPI"

    h = np.load(ham_path / "0.npy")
    sps = np.round(np.sqrt(h.shape[0])).astype(np.int64)

    # 3. Prepare the command arguments
    T = round(T, 5)
    cmd = [
        "mpirun",
        "-n" if n >= 1 else "",
        str(n) if n >= 1 else "",
        executable_name,
        "-m",
        model_name,
        "-T",
        str(T),
        "-ham",
        ham_path.resolve().as_posix(),
        "-unitary" if u_path else "",
        u_path.resolve().as_posix() if u_path else "",
        "-N",
        str(N),
        "--sps",
        str(sps),
        "--output" if logging else "",
        "--split-sweeps",
        "--obc" if obc else ""
    ]

    cmd = [str(arg) for arg in cmd if arg != ""]

    # Add -L arguments
    for i, length in enumerate(L, start=1):
        cmd.append(f"-L{i}")
        cmd.append(str(length))

    command = " ".join(cmd)
    env = os.environ.copy()
    with change_directory(release_dir):
        if not os.path.isdir(ham_path):
            logger.error("ham_path : %s not found. Please check the path.", ham_path)
            logger.error("current dir is: %s", os.getcwd())
            return
        # print("command: \n", command)
        logger.debug("command: %s", command)

        out = subprocess.run(
            command,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            env=env)

        # Check if command executed successfully
        if out.returncode != 0:
            error_message = f"Command execution failed with return code {out.returncode}"
            # print("Error output:", out.stdout.decode())
            logger.error("Error output: %s", out.stdout.decode())
            logger.error("command: %s", command)
            raise RuntimeError(error_message)

    return out
