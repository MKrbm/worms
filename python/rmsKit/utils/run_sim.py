import subprocess
import os
import glob
import re
from datetime import datetime
import contextlib
from typing import Union, List, Tuple


def extract_loss(dir_name):
    # Use regex to extract the loss value from the directory name
    match = re.search(r"loss_(\d+\.\d+)", dir_name)
    if match:
        return float(match.group(1))
    return None


def path_with_lowest_loss(parent_dir):
    # Get all directory names under the parent directory using glob
    dir_names = glob.glob(f"{parent_dir}/*")

    # Extract loss values and associate them with their paths
    losses = [(path, extract_loss(path)) for path in dir_names]
    # Filter out any paths where loss couldn't be extracted
    valid_losses = [(path, loss) for path, loss in losses if loss is not None]

    # Find the path with the lowest loss
    min_path, min_loss = min(valid_losses, key=lambda x: x[1])

    return min_path, min_loss


def run_ff_1d(
    seed: int,
    rank: int,
    sps: int,
    lt: int,
    dimension: int = 1,
    M: int = 100,
    e: int = 1000,
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
    }

    # check if executable exists
    if not os.path.isfile("torch_optimize_loc.py"):
        print("torch_optimize_loc.py not found. Please check the path.")
        return None

    # Construct the command to run the optimization script
    command = f"python torch_optimize_loc.py -m FF1D -loss mel -o LION -e {params['e']} -M {params['M']} -lr 0.005 --seed {params['seed']}"

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
            process = subprocess.run(command, shell=True, stdout=output_file, stderr=subprocess.STDOUT, text=True)

        # Provide feedback
        if process.returncode == 0:
            print("Optimization successfully.")
        else:
            print("Optimization failed.")
        print(
            f"""Command: {command}
                Output: {output_filepath}\n
                settings = {params} \n """
        )
    else:
        # Run the command and capture the output
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            print("Optimization successfully.")
        else:
            print("Optimization failed.")

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


def run_worm(model_name: str, ham_path: str, u_path: str, L: List[int], T: float, N: int, n: int = 1):
    # 1. Get the current directory
    current_dir = os.getcwd()

    # 2. Move up two directories and find the executable
    grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    # release_dir, executable_name = find_executable(grandparent_dir)
    executable_name = "main_MPI"
    release_dir = os.path.join(grandparent_dir, "Release")

    if not release_dir:
        print("Executable not found.")
        return

    # 3. Prepare the command arguments
    T = round(T, 4)
    cmd = [
        "mpirun",
        "-n",
        str(n),
        executable_name,
        "-m",
        model_name,
        "-T",
        str(T),
        "-ham",
        ham_path,
        "-unitary" if u_path else "",
        u_path if u_path else "",
        "-N",
        str(N),
        "-L1",
        str(L[0]),
        "-L2" if len(L) > 1 else "",
        str(L[1]) if len(L) > 1 else "",
        "--output",
    ]

    # Add -L arguments
    for i, length in enumerate(L, start=1):
        cmd.append(f"-L{i}")
        cmd.append(str(length))

    with change_directory(release_dir):
        # check if folder for ham_path and u_path exist (they are path to the folder)
        if not os.path.isdir(ham_path):
            print("current dir is: ", os.getcwd())
            print("ham_path : ", ham_path, " not found. Please check the path.")
            return

        # if not os.path.isdir(u_path):
        #     print("current dir is: ", os.getcwd())
        #     print("u_path : ", u_path, " not found. Please check the path.")
        #     return
        out = subprocess.run(cmd, stderr=subprocess.STDOUT)

    return out
