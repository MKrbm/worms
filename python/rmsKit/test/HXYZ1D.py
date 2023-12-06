import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.parser import get_parser
import numpy as np
import argparse
import os
import subprocess
import logging
import contextlib
import datetime
import re
import pandas as pd


NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"out"


@contextlib.contextmanager
def change_directory(directory):
    original_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_directory)


# Get the full path of the current file
current_file = os.path.abspath(__file__)

# Extract the filename
filename = os.path.basename(current_file)

# Extract the parent directory
parent_directory = os.path.dirname(current_file)
output_file = os.path.join(parent_directory, output_dir, f"{{}}_{NOW}_{filename}.txt")
rmsKit_directory = os.path.dirname(parent_directory)

args, params, hash_str = get_parser(length=True, model="HXYZ1D")

if __name__ == "__main__":

    params["L1"] = args.length1

    #
    # with change_directory(rmsKit_directory):
    #     # Add the rmsKit directory to the path
    #     cmd = [
    #         "python",
    #         "solver_torch.py",
    #         "-m",
    #         "HXYZ1D",
    #         "-L1",
    #         str(params["L1"]),
    #         "-Jz",
    #         str(params["Jz"]),
    #         "-Jx",
    #         str(params["Jx"]),
    #         "-Jy",
    #         str(params["Jy"]),
    #         "-hx",
    #         str(params["hx"]),
    #         "-hz",
    #         str(params["hz"]),
    #         "--stdout",
    #         # ">>",
    #         # output_file_temp.format("exact"),
    #     ]
    #
    #     command = " ".join(cmd)
    #     out = subprocess.run(command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    #
    # stdout = out.stdout.decode("utf-8")
    # # print(stdout)
    # # redirect stdout to output_file
    # with open(output_file.format("exact"), "w") as f:
    #     f.write(stdout)
    #
    # # Regex pattern to match the line with the file path
    # pattern = re.compile(r'stat file: (\S+\.csv)')
    #
    # csv_path: str = ""
    #
    # lines = stdout.split('\n')
    # for line in lines:
    #     # print(line, end='')  # Optional: print the output line by line
    #     match = pattern.search(line)
    #     if match:
    #         csv_path = match.group(1)
    #         break  # Stop reading more lines if the path is found
    #
    # df_stat = pd.read_csv(csv_path)
    #
    #

    with change_directory(rmsKit_directory):
        # Add the rmsKit directory to the path
        cmd = [
            "python",
            "new_optimize_loc.py",
            "-m",
            "HXYZ1D",
            "-Jz",
            str(params["Jz"]),
            "-Jx",
            str(params["Jx"]),
            "-Jy",
            str(params["Jy"]),
            "-hx",
            str(params["hx"]),
            "-hz",
            str(params["hz"]),
            "-e 1000",
            "-M 10",
            "--stdout",
        ]
        # Get a copy of the current environment variables
        env = os.environ.copy()

        # Modify the MKL_THREADING_LAYER environment variable
        # env["MKL_THREADING_LAYER"] = "GNU"
        command = " ".join(cmd)
        out = subprocess.run(
            command,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            env=env)
    stdout = out.stdout.decode("utf-8")
    print(stdout)
    # redirect stdout to output_file
    with open(output_file.format("optimize"), "w") as f:
        f.write(stdout)


    # Define regular expressions
    best_unitary_pattern = re.compile(r'best loss was saved to (\S+)')
    hamiltonian_pattern = re.compile(r'hamiltonian was saved to (\S+)')

    # Extract paths
    best_unitary_path = None
    hamiltonian_path = None

    lines = stdout.split('\n')
    for line in lines:
        if best_unitary_match := best_unitary_pattern.search(line):
            best_unitary_path = best_unitary_match.group(1)
        elif hamiltonian_match := hamiltonian_pattern.search(line):
            hamiltonian_path = hamiltonian_match.group(1)


    if best_unitary_path is None:
        raise ValueError("Best unitary path not found.")
    if hamiltonian_path is None:
        raise ValueError("Hamiltonian path not found.")


