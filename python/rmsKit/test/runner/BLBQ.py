from . import run
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def run_BLBQ1D(params: Dict,
               rmsKit_directory,
               output_dir) -> Tuple[pd.DataFrame,
                                    pd.DataFrame,
                                    pd.DataFrame,
                                    pd.DataFrame]:

    output_dir = Path(output_dir)
    cmd = [
        "python",
        "solver_torch.py",
        "-m",
        "BLBQ1D",
        "-L1",
        str(params["L1"]),
        "-J0",
        str(params["J0"]),
        "-J1",
        str(params["J1"]),
        "-hx",
        str(params["hx"]),
        "-hz",
        str(params["hz"]),
        "-lt",
        str(params["lt"]),
        "--stdout",
    ]

    cmd_solver = " ".join(cmd)

    cmd = [
        "python",
        "new_optimize_loc.py",
        "-m",
        "BLBQ1D",
        "-J0",
        str(params["J0"]),
        "-J1",
        str(params["J1"]),
        "-hx",
        str(params["hx"]),
        "-hz",
        str(params["hz"]),
        "-lt",
        str(params["lt"]),
        "--stdout",
        "-e 2000",
        "-M 20",
        "-lr 0.01",
        "--stdout",
    ]
    cmd_optimize = " ".join(cmd)

    return run.run(params, rmsKit_directory, output_dir, "BLBQ1D", cmd_solver, cmd_optimize)
