from typing import Dict, List, Tuple
import pandas as pd
import re
import datetime
import contextlib
import logging
import subprocess
import os

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import run_worm  # noqa: E402
from utils.functions import extract_info_from_file, get_loss  # noqa: E402

NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def change_directory(directory):
    original_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_directory)


def run(params: Dict,
        rmsKit_directory: Path,
        output_dir: Path,
        model_name: str,
        cmd_solver: str,
        cmd_optimize: str,
        N: int,
        beta: List[float],
        ) -> Tuple[pd.DataFrame,
                   pd.DataFrame,
                   pd.DataFrame,
                   pd.DataFrame]:

    output_file = (output_dir / f"{{}}_{NOW}_{model_name}.txt").resolve().as_posix()

    logger.info("command sovler: {}".format(cmd_solver))
    logger.info("command optimization: {}".format(cmd_optimize))
    with change_directory(rmsKit_directory):
        # Add the rmsKit directory to the path

        command = cmd_solver
        out = subprocess.run(command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)

        stdout = out.stdout.decode("utf-8")
        # redirect stdout to output_file
        with open(output_file.format("exact"), "w") as f:
            f.write(stdout)

        # Regex pattern to match the line with the file path
        pattern = re.compile(r'stat file: (\S+\.csv)')
        pattern2 = re.compile(r'eigenvalues file: (\S+\.csv)')

        csv_path: str = ""
        eig_csv_path: str = ""

        # print(stdout)
        lines = stdout.split('\n')
        for line in lines:
            # print(line, end='')  # Optional: print the output line by line
            match = pattern.search(line)
            match2 = pattern2.search(line)
            if match:
                csv_path = match.group(1)
            elif match2:
                eig_csv_path = match2.group(1)

        df_stat = pd.read_csv(csv_path)
        df_eig = pd.read_csv(eig_csv_path)

        command = cmd_optimize

        env = os.environ.copy()
        out = subprocess.run(
            command,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            env=env)
        stdout = out.stdout.decode("utf-8")
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
            logging.error("\n{}".format(stdout))
            raise ValueError("Best unitary path not found.")
        if hamiltonian_path is None:
            logging.error("\n{}".format(stdout))
            raise ValueError("Hamiltonian path not found.")

        # get absolute path
        best_unitary_path = os.path.abspath(best_unitary_path)
        hamiltonian_path = os.path.abspath(hamiltonian_path)

        # run normal worm

        L_list = [params["L1"], params["L2"]] if "L2" in params else [params["L1"]]

        rmsKit_directory = Path(rmsKit_directory)

        run_res = []

        for b in beta:
            t = 1 / b
            output = run_worm(
                model_name,
                hamiltonian_path,
                best_unitary_path,
                L_list,
                t,
                N,
                project_dir=rmsKit_directory.parent.parent.resolve(),
                logging=False)

            with open(output_file.format("worm"), "w") as f:
                f.write(output.stdout.decode("utf-8"))

            dic = extract_info_from_file(output_file.format("worm"))
            dic["beta"] = b
            run_res.append(dic)

        df_worm = get_loss(pd.DataFrame(run_res))
        df_worm = df_worm.drop(columns=["u_path", "ham_path"])

        # Merge the two DataFrames on beta
        merged_df = pd.merge(df_stat, df_worm, on='beta', how='inner')

        merged_df['e_diff'] = merged_df['energy_per_site'] - merged_df['e']
        merged_df['c_diff'] = merged_df['specific_heat'] - merged_df['c']

        merged_df["e_test"] = merged_df["e_diff"].abs() < 3 * merged_df["e_error"]
        merged_df["c_test"] = merged_df["c_diff"].abs() < 3 * merged_df["c_error"]

        merged_df_u = merged_df

        run_res = []
        for b in beta:
            t = 1 / b
            output = run_worm(model_name, hamiltonian_path, "", L_list, t, N,
                              project_dir=rmsKit_directory.parent.parent.resolve(), logging=False)

            with open(output_file.format("worm"), "w") as f:
                f.write(output.stdout.decode("utf-8"))

            dic = extract_info_from_file(output_file.format("worm"))
            dic["beta"] = b
            run_res.append(dic)

        df_worm = get_loss(pd.DataFrame(run_res))
        df_worm = df_worm.drop(columns=["u_path", "ham_path"])

        # Merge the two DataFrames on beta
        merged_df = pd.merge(df_stat, df_worm, on='beta', how='inner')

        merged_df['e_diff'] = merged_df['energy_per_site'] - merged_df['e']
        merged_df['c_diff'] = merged_df['specific_heat'] - merged_df['c']

        merged_df["e_test"] = merged_df["e_diff"].abs() < 3 * merged_df["e_error"]
        merged_df["c_test"] = merged_df["c_diff"].abs() < 3 * merged_df["c_error"]

        merged_df_h = merged_df
    # return optimized unitary, identity, eigenvalues, statistics
    return merged_df_u, merged_df_h, df_eig, df_stat
